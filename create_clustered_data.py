"""
Calling this file creates a ground truth position not from the basic 2D Gaussian prior, but as a Gaussian mixture model creating a number of different clusters.
"""

## Basics
import os
import numpy as np
import jax.numpy as jnp
import jax.scipy.stats as jstats
import jax
import matplotlib.pyplot as plt
import pickle

## Typing
from jax._src.prng import PRNGKeyArray
from jax._src.typing import ArrayLike
from typing import Callable, Tuple

## Self-made functions
from binary_euclidean_LSM import sample_observation as bin_euc_sample_observation
from binary_hyperbolic_LSM import sample_observation as bin_hyp_sample_observation
from continuous_euclidean_LSM import sample_observation as con_euc_sample_observation
from continuous_hyperbolic_LSM import sample_observation as con_hyp_sample_observation
from helper_functions import get_cmd_params, set_GPU, open_taskfile, get_safe_folder, get_filename_with_ext, create_task_file, triu2mat

### Create cmd argument list (arg_name, var_name, type, default[OPT], nargs[OPT]).
###  - arg_name is the name of the argument in the command line.
###  - var_name is the name of the variable in the returned dictionary (which we re-use as variable name here).
###  - type is the data-type of the variable.
###  - default is the default value it takes if nothing is passed to the command line. This argument is only optional if type is bool, where the default is always False.
###  - nargs is the number of arguments, where '?' (default) is 1 argument, '+' concatenates all arguments to 1 list. This argument is optional.
_n_clusters = 3
arguments = [('-bdf', 'base_data_filename', str, 'clustered_data'), # the most basic version of the filename of the saved data
             ('-datfol', 'data_folder', str, 'Data'), # folder where the data is stored
             ('-of', 'output_folder', str, 'Data/cluster_sim'), # folder where to dump data
             ('-fof', 'figure_output_folder', str, 'Figures/cluster_sim'), # folder where to dump figures
             ('-tf', 'task_filename', str, 'cluster_tasklist'), # filename of the list of task names
             ('-tdelim', 'task_delimiter', str, ','), # delimiter of the tasks in the task file
             ('-nsub', 'n_subjects', int, 5), # number of "subjects" (= observations per GT)
             ('-ntasks', 'n_tasks', int, 1),  # number of "tasks" (= ground truths)
             ('-nc', 'n_clusters', int, _n_clusters), # number of clusters
             ('--diveven', 'divide_evenly', bool), # whether to divide the nodes evenly over the clusters
             ('-ppc', 'prob_per_cluster', float, [1/_n_clusters]*_n_clusters, '+'), # list of probabilities, describing the chance of a node being assigned to each cluster
             ('-mcd', 'min_cluster_dist', float, 0.), # minimum distance between the clusters
             ('-alpha', 'alpha', float, [3.]*_n_clusters, '+'), # shape parameters of the gamma distribution to sample cluster means
             ('-theta', 'theta', float, [1.]*_n_clusters, '+'), # scale parameters of the gamma distribution to sample cluster means
             ('-sigmus', 'sigma_mus', float, [1.]*_n_clusters, '+'), # standard deviations for the clusters' normal distributions
             ('-sbt', 'sigma_beta_T', float, 0.), # logit transformed standard deviation of the beta distribution
             ('-et', 'edge_type', str, 'con'), # edge type of the generated network
             ('-geo', 'geometry', str, 'euc'), # geometry of the latent space
             ('-N', 'N', int, 50), # number of nodes
             ('-D', 'D', int, 2), # dimensionality of the latent space
             ('-seed', 'seed', int, 0), # PRNGKey seed
             ('-seedfile', 'seed_file', str, 'seed.txt'), # save file for the seed
             ('--plot', 'make_plot', bool), # whether to make a plot
             ('-cm', 'cmap', str, 'bwr'), # colormap for correlation figures (divergent)
             ('-ccmap', 'cluster_cmap', str, 'jet'), # colormap for assigning colors to clusters
             ('-lfs', 'label_fontsize', float, 20),  # fontsize of labels (and legend)
             ('-tfs', 'tick_fontsize', float, 16),  # fontsize of the tick labels
             ('-gpu', 'gpu', str, ''), # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]
## Get arguments from command line.
global_params = get_cmd_params(arguments)
set_GPU(global_params['gpu']) ## MUST BE RUN FIRST
base_data_filename = global_params['base_data_filename']
data_folder = global_params['data_folder']
N = global_params['N']
output_folder = get_safe_folder(f"{global_params['output_folder']}")
figure_output_folder = get_safe_folder(f"{global_params['figure_output_folder']}")
task_filename = get_filename_with_ext(global_params['task_filename'], ext='txt', folder=data_folder)
task_delimiter = global_params['task_delimiter']
n_subjects = global_params['n_subjects']
n_clusters = global_params['n_clusters']
prob_per_cluster = global_params['prob_per_cluster']
assert np.sum(prob_per_cluster) == 1.
min_cluster_dist = global_params['min_cluster_dist']
alpha = global_params['alpha']
theta = global_params['theta']
sigma_mus = global_params['sigma_mus']
sigma_beta_T = global_params['sigma_beta_T']
M = N*(N-1)//2
D = global_params['D']
seed = global_params['seed']
seed_file = global_params['seed_file']
make_plot = global_params['make_plot']
divide_evenly = global_params['divide_evenly']
cmap = global_params['cmap']
cluster_cmap = global_params['cluster_cmap']
label_fontsize = global_params['label_fontsize']
tick_fontsize = global_params['tick_fontsize']
edge_type = global_params['edge_type']
geometry = global_params['geometry']
n_tasks = global_params['n_tasks']

## Deal with wrong length sigmas
if len(sigma_mus) is not n_clusters:
    sigma_mus = [1.]*n_clusters

if seed is None:
    with open(seed_file, 'r') as f:
        seed = int(f.read())
## Use JAX functions only after setting the GPU, otherwise it will use all GPUs by default.
key = jax.random.PRNGKey(seed)

## Define a number of variables based on geometry or edge type
latpos = '_z' if geometry == 'hyp' else 'z'
sample_obs_dict = {
                   'bin_euc':bin_euc_sample_observation,
                   'bin_hyp':bin_hyp_sample_observation,
                   'con_euc':con_euc_sample_observation,
                   'con_hyp':con_hyp_sample_observation,
                  }
sample_obs_func = sample_obs_dict[f"{edge_type}_{geometry}"]

def get_mus(key:PRNGKeyArray, n_clusters:int=n_clusters, min_cluster_dist:float=min_cluster_dist, alpha:float=alpha, theta:float=theta, D:int=D) -> Tuple[PRNGKeyArray, jnp.array]:
    """
    Samples means of the clusters by sampling a random direction and a (minimum-bounded) distance
    PARAMS:
    key : random key for JAX functions
    n_clusters : number of clusters
    min_cluster_dist : the minimum distance between two cluster means
    alpha : shape parameter of the gamma distribution 
    theta : scale parameter of the gamma distribution 
    D : dimensionality of the latent space
    """
    mus = jnp.zeros((n_clusters, D))
    for ci in range(n_clusters):
        key, phi_key, gamma_key = jax.random.split(key, 3)
        ## Sample the direction from [0, 2pi]
        phi = jax.random.uniform(phi_key, minval=0, maxval=2*np.pi)
        ## Sample the distance from a gamma distribution, then add the minimum distance
        d = min_cluster_dist + jax.random.gamma(gamma_key, alpha)*theta
        ## Transform polar to cartesian
        mu_x = jnp.cos(phi)*d
        mu_y = jnp.sin(phi)*d
        mus = mus.at[ci,:].set([mu_x, mu_y])
    return key, mus

def divide_nodes_evenly(N:int=N, n_clusters:int=n_clusters) -> np.array:
    """
    Divides the nodes over the clusters as evenly as possible
    PARAMS:
    N : total number of latent positions
    n_clusters : number of clusters
    """
    N_per_cluster = np.zeros(n_clusters, dtype=int)
    ci = 0
    cluster_index = [ci]
    while N > 0:
        N_per_cluster[ci] += 1
        N -= 1
        ci = (ci+1)%n_clusters
        cluster_index.append(ci)
    return N_per_cluster, cluster_index

def divide_nodes_by_prob(N:int=N, prob_per_cluster:ArrayLike=prob_per_cluster) -> Tuple[np.array, np.array]:
    """
    Divides the nodes over the clusters probabilisticly. Returns both the number of nodes per cluster and the cluster index per node.
    PARAMS:
    N : total number of latent positions
    prob_per_cluster : (n_clusters,) contains the probability of a node being assigned to each cluster
    """
    n_clusters = len(prob_per_cluster)
    ## Give eaach node a cluster index
    cluster_index = np.random.choice(n_clusters, (N,), p=prob_per_cluster)
    ## Then count how many nodes are assigned to each cluster
    N_per_cluster = np.array([np.sum(cluster_index == c) for c in range(n_clusters)])
    return N_per_cluster, cluster_index

def get_clustered_latent_positions(key:PRNGKeyArray, mus:ArrayLike, sigmas:ArrayLike, N_per_cluster:ArrayLike, D:int=D) -> Tuple[PRNGKeyArray, jnp.array]:
    """
    Samples latent position in clusters as samples from normal distributions described by the given mus and sigmas.
    PARAMS:
    key : random key for JAX functions
    mus : means of the clusters
    sigmas : standard deviations of the clusters
    N_per_cluster : number of nodes per cluster
    D : dimensionality of the latent space
    """
    assert len(mus) == len(sigmas) == len(N_per_cluster), f"mus, sigmas, N_per_cluster must all be of same length but are {len(mus)}, {len(sigmas)} and {len(N_per_cluster)} respectively"
    N = jnp.sum(N_per_cluster)
    z = jnp.zeros((N, D))
    n_clusters = len(mus)
    start_i, end_i = 0, N_per_cluster[0]
    for ci in range(n_clusters):
        key, subkey = jax.random.split(key)
        zci = sigmas[ci]*jax.random.normal(subkey, (N_per_cluster[ci], D))+mus[ci,:]
        z = z.at[start_i:end_i, :].set(zci)
        ## Update indices
        start_i = jnp.sum(N_per_cluster[:ci+1])
        end_i = jnp.sum(N_per_cluster[:ci+2])
    return key, z

## Create the task file, and immediately re-read it, to also get the properly formatted "encodings". 
create_task_file(task_filename, n_tasks, 1, task_delimiter)
tasks, encs = open_taskfile(task_filename)

cluster_colors = np.array([plt.get_cmap(cluster_cmap)(i) for i in np.linspace(0,1,n_clusters)])

obs = {}
for ti, task in enumerate(tasks):
    ## Take one GT per task, each subject is a noisy observation of the GT
    N_per_cluster, cluster_index = divide_nodes_evenly(N, n_clusters) if divide_evenly else divide_nodes_by_prob(N, prob_per_cluster)
    key, mus = get_mus(key)
    key, z = get_clustered_latent_positions(key, mus, sigma_mus, N_per_cluster)

    ## Save ground truth
    ground_truth = {latpos: z}
    if edge_type == 'con':
        ground_truth['sigma_beta_T'] = sigma_beta_T
    ground_truth_filename = get_filename_with_ext(f"gt_{task}", folder=output_folder)
    with open(ground_truth_filename, 'wb') as f:
        pickle.dump(ground_truth, f)

    if make_plot:
        ## Getting the colors to line up, convoluted to deal with the fact that each color is also an array.
        node_colors = np.array([])
        for ci in range(n_clusters):
            node_colors = np.hstack([node_colors, np.tile(cluster_colors[ci], N_per_cluster[ci])])
        ## Reshape to make each element an individual color array again.
        node_colors = node_colors.reshape((N, len(cluster_colors[0])))

        ## Scatter plot of the latent positions, including cluster means.
        plt.figure(figsize=(7,7))
        plt.scatter(mus[:, 0], mus[:, 1], c=cluster_colors, marker='*', label=r'$\mu$')
        plt.scatter(z[:, 0], z[:, 1], c=node_colors, s=5, label='z')
        plt.legend(fontsize=tick_fontsize)
        savetitle = get_filename_with_ext(f"gt_{task}_{n_clusters}_clusters", ext='png', folder=figure_output_folder)
        plt.savefig(savetitle, bbox_inches='tight')
        plt.close()

    for n_sub in range(n_subjects):
        ## We see each subject as a noisy observation of the ground truth, and so per subject we take 1 observation.
        key, A = sample_obs_func(key, ground_truth, 1)
        A = A[0] # Unpack the one sample
        obs[f"S{n_sub}_{task}_{encs[0]}"] = A

        ## Plot correlations
        if make_plot:
            plt.figure(figsize=(10,10))
            plt.imshow(triu2mat(A), vmin=-1, vmax=1, cmap=cmap)
            c_start_index = np.cumsum(N_per_cluster)
            plt.xticks(c_start_index, labels=range(n_clusters), fontsize=tick_fontsize)
            plt.yticks(c_start_index, labels=range(n_clusters), fontsize=tick_fontsize)
            filename = get_filename_with_ext(f"S{n_sub}_{task}_correlations", ext='png', folder=figure_output_folder)
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

## Save observations
observations_filename = get_filename_with_ext('cluster_sim_observations', folder=output_folder)
with open(observations_filename, 'wb') as f:
    pickle.dump(obs, f)

    