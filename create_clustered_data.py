import os

import numpy as np
import jax.numpy as jnp
import jax.scipy.stats as jstats
import jax
import matplotlib.pyplot as plt
import pickle

from continuous_hyperbolic_LSM import sample_observation
from helper_functions import get_cmd_params, set_GPU, open_taskfile, get_safe_folder
from plotting_functions import plot_correlations

from jax._src.prng import PRNGKeyArray
from jax._src.typing import ArrayLike
from typing import Callable, Tuple

arguments = [('-df', 'base_data_filename', str, 'clustered_data'),  # the most basic version of the filename of the saved data
             ('-of', 'output_folder', str, 'Data/cluster_sim'), # folder where to dump data
             ('-fof', 'figure_output_folder', str, 'Figures/cluster_sim'), # folder where to dump figures
             ('-tf', 'task_filename', str, 'task_list.txt'), # filename of the list of task names
             ('-s1', 'subject1', int, 1),  # first subject to be used
             ('-sn', 'subjectn', int, 5),  # last subject to be used
             ('-nc', 'n_clusters', int, 5), # number of clusters
             ('-mcd', 'min_cluster_dist', float, 3.), # minimum distance between the clusters
             ('-alpha', 'alpha', float, 3.), # shape parameter of the gamma distribution to sample cluster means
             ('-theta', 'theta', float, 1.), # scale parameter of the gamma distribution to sample cluster means
             ('-sigmus', 'sigma_mus', float, 1.), # standard deviation for the cluster's normal distribution
             ('-sbt', 'sigma_beta_T', float, 0.), # logit transformed standard deviation of the beta distribution
             ('-N', 'N', int, 164), # number of nodes
             ('-D', 'D', int, 2), # dimensionality of the latent space
             ('-seed', 'seed', int, 0), # PRNGKey seed
             ('--plot', 'make_plot', bool), # whether to make a plot
             ('-cm', 'cmap', str, 'bwr'), # colormap
             ('-gpu', 'gpu', str, ''),  # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]

global_params = get_cmd_params(arguments)
base_data_filename = global_params['base_data_filename']
N = global_params['N']
output_folder = get_safe_folder(f"{global_params['output_folder']}/{N}")
figure_output_folder = get_safe_folder(f"{global_params['figure_output_folder']}/{N}")
task_filename = global_params['task_filename']
subject1 = global_params['subject1']
subjectn = global_params['subjectn']
n_clusters = global_params['n_clusters']
min_cluster_dist = global_params['min_cluster_dist']
alpha = global_params['alpha']
theta = global_params['theta']
sigma_mus = global_params['sigma_mus']
sigma_beta_T = global_params['sigma_beta_T']
M = N*(N-1)//2
D = global_params['D']
seed = global_params['seed']
make_plot = global_params['make_plot']
cmap = global_params['cmap']
set_GPU(global_params['gpu'])

key = jax.random.PRNGKey(seed)

def get_mus(key:PRNGKeyArray, n_clusters:int=n_clusters, min_cluster_dist:float=min_cluster_dist, alpha:float=alpha, theta:float=theta, D:int=D) -> Tuple[PRNGKeyArray, jnp.array]:
    """
    Samples 2D means of the clusters by sampling a random direction uniformly in [0, 2pi] and a (minimum-bounded) distance from the origin from a gamma distribution.
    PARAMS:
    key : random key for JAX functions
    n_clusters : number of clusters
    min_cluster_dist : the minimum distance between two cluster means
    alpha : shape parameter of the gamma distribution 
    theta : scale parameter of the gamma distribution 
    D : dimensionality of the latent space
    """
    assert D==2, f"D must be 2 because my code sucks but D is {D} instead."
    mus = jnp.zeros((n_clusters, D)) # D=2 always, otherwise getting the mus doesn't work
    for ci in range(n_clusters):
        key, phi_key, gamma_key = jax.random.split(key, 3)
        phi = jax.random.uniform(phi_key, minval=0, maxval=2*np.pi)
        d = min_cluster_dist + jax.random.gamma(gamma_key, alpha)*theta
        mu_x = jnp.cos(phi)*d
        mu_y = jnp.sin(phi)*d
        mus = mus.at[ci,:].set([mu_x, mu_y])
    return key, mus

def divide_nodes(N:int=N, n_clusters:int=n_clusters) -> np.array:
    """
    Divides the nodes over the clusters as evenly as possible
    PARAMS:
    N : total number of latent positions
    n_clusters : number of clusters
    """
    N_per_cluster = np.zeros(n_clusters, dtype=int)
    ci = 0
    while N > 0:
        N_per_cluster[ci] += 1 ###N_per_cluster.at[ci].set(N_per_cluster[ci]+1)
        N -= 1
        ci = (ci+1)%n_clusters
    return N_per_cluster

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
    _z = jnp.zeros((N, D))
    n_clusters = len(mus)
    start_i, end_i = 0, N_per_cluster[0]
    for ci in range(n_clusters):
        key, subkey = jax.random.split(key)
        _zci = jax.random.normal(subkey, (N_per_cluster[ci], D)) # Gives standard N(0,1)
        _zci = _zci.at[:,0].set(sigmas[ci]*_zci[:,0]+mus[ci,0]) # Transform N(0,1) to N(mu, sigma) in x
        _zci = _zci.at[:,1].set(sigmas[ci]*_zci[:,1]+mus[ci,1]) # Transform N(0,1) to N(mu, sigma) in y
        # print("i:{}, index size: {}, start: {}, end: {}, length input: {}, cluster size: {}".format(ci, end_i-start_i, start_i, end_i, len(_zci), N_per_cluster[ci]))
        _z = _z.at[start_i:end_i, :].set(_zci)
        # Update _z indices
        if ci < n_clusters-1:
            start_i = jnp.sum(N_per_cluster[:ci+1]) # ci+1 (because end is exclusive)
            end_i = jnp.sum(N_per_cluster[:ci+2]) # ci+1 (because we want to 'see' the next one) +1 (because end is exclusive)
    return key, _z

sigmas = jnp.array([sigma_mus]*n_clusters)
N_per_cluster = divide_nodes()

tasks, encs = open_taskfile(task_filename)
obs = {}

for si, n_sub in enumerate(range(subject1, subjectn+1)):
    for ti, task in enumerate(tasks):
        # Create latent positions
        key, mus = get_mus(key)
        key, _z = get_clustered_latent_positions(key, mus, sigmas, N_per_cluster)

        if make_plot:
            plt.figure(figsize=(6, 6))
            plt.scatter(mus[:, 0], mus[:, 1], c='r', label=u'cluster $\mu$')
            plt.scatter(_z[:, 0], _z[:, 1], c='k', s=1,
                        label=u'\u200c_z')  # Add zero width unicode character, labels that start with underscores are ignored in plt.legend.
            plt.legend()
            plt.title(f"Ground truth for S{n_sub}_{task} with {n_clusters} clusters")
            plt.tight_layout()
            savetitle = f"{figure_output_folder}/gt_S{n_sub}_{task}_{n_clusters}_clusters.png"
            plt.savefig(savetitle)
            plt.close()
            
        # Save ground truth
        ground_truth = {'_z': _z, 'sigma_beta_T': sigma_beta_T}
        ground_truth_filename = f"{output_folder}/gt_S{n_sub}_{task}.pkl"
        with open(ground_truth_filename, 'wb') as f:
            pickle.dump(ground_truth, f)

        for ei, enc in enumerate(encs): # Do it this way to use the right dictionary keys
            # THIS MIGHT BE WEIRD, BECAUSE SAMPLE_OBSERVATIONS CALLS get_det_params WHICH ASSUMES _z WAS DRAWN FROM A SINGLE NORMAL DISTRIBUTION I THINK?
            # But because we use mu=0 (default), we don't actually do anything with _z because we're not parallel transporting, we're really only exponential wrapping.
            key, A = sample_observation(key, ground_truth) # key, 1 x M
            A = A[0] # 1 x M --> M
            obs[f"S{n_sub}_{task}_{enc}"] = A

            # Plot correlations
            if make_plot:
                vmax = np.max(np.abs(A))
                vmin = -vmax
                plt.figure(figsize=(6, 6))
                ax = plt.gca()
                ax = plot_correlations(A, ax, cmap=cmap, add_colorbar=True, vmin=vmin, vmax=vmax)
                plt.title(f"S{n_sub} {task} {enc}")
                plt.xlabel('Node index')
                plt.ylabel('Node index')
                plt.tight_layout()
                plt.savefig(f"{figure_output_folder}/S{n_sub}_{task}_{enc}_corr.png")
                plt.close()

# Save observations
observations_filename = f"{output_folder}/clustered_sim_observations.pkl"
with open(observations_filename, 'wb') as f:
    pickle.dump(obs, f)

    