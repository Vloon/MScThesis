import os

import numpy as np
import jax.numpy as jnp
import jax.scipy.stats as jstats
import jax
import matplotlib.pyplot as plt
import pickle

from continuous_hyperbolic_LSM import sample_observation
from helper_functions import get_cmd_params, set_GPU, open_taskfile, get_safe_folder, get_filename_with_ext, invlogit, is_valid
from plotting_functions import plot_correlations

from jax._src.prng import PRNGKeyArray
from jax._src.typing import ArrayLike
from typing import Callable, Tuple

from continuous_hyperbolic_LSM import sample_prior as con_hyp_sample_prior, sample_observation as con_hyp_sample_observation
from binary_euclidean_LSM import sample_prior as bin_euc_sample_prior, sample_observation as bin_euc_sample_observation
# sample_prior(key:PRNGKeyArray, shape:tuple, mu:float = mu, sigma:float = sigma, eps:float = eps)


arguments = [('-df', 'base_data_filename', str, 'prior_data'),  # the most basic version of the filename of the saved data
             ('-of', 'output_folder', str, 'Data/prior_sim'), # folder where to dump data
             ('-fof', 'figure_output_folder', str, 'Figures/prior_sim'), # folder where to dump figures
             ('-tf', 'task_filename', str, 'task_list_prior_sim'), # filename of the list of task names WITHOUT EXTENSION
             ('-tdelim', 'task_delimiter', str, ','), # delimiter of the tasks in the task file
             ('-nsub', 'n_subjects', int, 1), # number of "subjects"
             ('-ntasks', 'n_tasks', int, 1), # number of "tasks"
             ('-nobs', 'n_observations', int, 2), # number of observations sampled
             ('-s', 'sigma', float, 1.), # standard deviation for the cluster's normal distribution
             ('-musig', 'mu_sigma', float, 0.), # mean of the logit-transformed std of the beta distribution
             ('-sigsig', 'sigma_sigma', float, 1.), # standard deviation of the logit-transformed std of the beta distribution
             ('-sbt', 'sigma_beta_T', float, None), # logit transformed standard deviation of the beta distribution. If this one is given, sigma is taken from here.
             ('-N', 'N', int, 164), # number of nodes
             ('-D', 'D', int, 2), # dimensionality of the latent space
             ('-et', 'edge_type', str, 'con'), # edge type ('con' or 'bin')
             ('-geo', 'geometry', str, 'hyp'), # LS geometry ('hyp' or 'euc')
             ('-seed', 'seed', int, 0), # PRNGKey seed
             ('--plot', 'make_plot', bool), # whether to make plots
             ('-cm', 'cmap', str, 'bwr'), # colormap for the observation plots
             ('-gpu', 'gpu', str, ''),  # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]

global_params = get_cmd_params(arguments)
base_data_filename = global_params['base_data_filename']
N = global_params['N']
M = N*(N-1)//2
D = global_params['D']
output_folder = get_safe_folder(f"{global_params['output_folder']}")
figure_output_folder = get_safe_folder(f"{global_params['figure_output_folder']}/{N}")
task_filename = global_params['task_filename']
task_delimiter = global_params['task_delimiter']
n_subjects = global_params['n_subjects']
n_tasks = global_params['n_tasks']
n_observations = global_params['n_observations']
sigma = global_params['sigma']
mu_sigma = global_params['mu_sigma']
sigma_sigma = global_params['sigma_sigma']
sigma_beta_T = global_params['sigma_beta_T']
edge_type = global_params['edge_type']
geometry = global_params['geometry']
make_plot = global_params['make_plot']
cmap = global_params['cmap']
set_GPU(global_params['gpu'])
# After GPU is set
key = jax.random.PRNGKey(global_params['seed'])

sample_prior_dict = {'con_hyp':con_hyp_sample_prior}
sample_prior_func = sample_prior_dict[f"{edge_type}_{geometry}"]

sample_obs_dict = {'con_hyp':con_hyp_sample_observation}
sample_obs_func = sample_obs_dict[f"{edge_type}_{geometry}"]

def create_task_file(filename:str, n_tasks:int=n_tasks, n_observations:int=n_observations, delim:str=task_delimiter) -> None:
    """
    Creates a task file which can later be used by load_observations.
    PARAMS:
    filename : name of the task file
    n_tasks : number of "tasks"
    n_observations : number of observations
    delim : delimiter between the tasks and observations in the task file
    """
    task_str = delim.join([f'T{i}' for i in range(n_tasks)])
    obs_str = delim.join([f'obs{i}' for i in range(n_observations)])
    with open(filename, 'w') as f:
        f.write(f'{task_str}\n')
        f.write(obs_str)

def create_data(key:PRNGKeyArray, shape:tuple=(N,D), n_observations:int=n_observations, sigma:float=sigma, mu_sigma:float=mu_sigma, sigma_sigma:float=sigma_sigma, sigma_beta_T:float=sigma_beta_T, sample_prior_func:Callable=sample_prior_func, sample_obs_func:Callable=sample_obs_func) -> Tuple[PRNGKeyArray, dict, jnp.array]:
    """
    Samples a prior as ground truth and samples some observations
    PARAMS:
    key : random key for jax functions
    shape : NxD nodes by latent space dimensions
    n_observations : number of observations per ground truth
    sigma : std of the latent space positions
    mu_sigma : mean of the shared (transformed) sigma over the beta distributions
    sigma_sigma : std of the shared (transformed) sigma over the beta distributions
    sigma_beta_T : if given, sets sigma_beta_T in the prior to the given value
    sample_prior_func : function to sample the prior
    sample_obs_func : function to sample observations from the prior
    """
    ### TODO: make sure it works with other prior functions that need other parameters (e.g. bin euc) (something with dictionaries?)
    # Sample prior
    params = {}
    key, gt_prior = sample_prior_func(key, shape, sigma, mu_sigma, sigma_sigma)
    # Overwrite SBT
    if sigma_beta_T is not None:
        gt_prior['sigma_beta_T'] = sigma_beta_T
    # Sample observations
    key, observations = sample_obs_func(key, gt_prior, n_observations)
    return key, gt_prior, observations

# Create correct filenames
sbt_txt = f"_sbt_{sigma_beta_T:.1f}" if sigma_beta_T is not None else ''
task_filename = f"{task_filename}_N_{N}_n_obs_{n_observations}{sbt_txt}"
output_task_filename = get_filename_with_ext(task_filename, ext='txt', folder=output_folder)
data_filename = f"{base_data_filename}_N_{N}_n_obs_{n_observations}{sbt_txt}"
output_data_filename = get_filename_with_ext(data_filename, folder=output_folder)

create_task_file(output_task_filename)

latpos = '_z' if geometry == 'hyp' else 'z'

sim_data = {}

for si in range(1,n_subjects+1):
    for ti in range(n_tasks):
        key, gt_prior, observations = create_data(key)
        gt_filename = get_filename_with_ext(f"gt_prior_S{si}_T{ti}_N_{N}_n_obs_{n_observations}{sbt_txt}", folder=output_folder)
        with open(gt_filename, 'wb') as f:
            pickle.dump(gt_prior, f)

        flat_observations = np.zeros((n_observations, M), dtype=float)
        for oi, obs in enumerate(observations): # Observations has length n_observations.
            valid, idc = is_valid(obs)
            if not valid:
                print(f"  bad idc: {idc}")
            valid = np.all(obs > 0) and np.all(obs < 1)
            if not valid:
                print(f"  bad idc <= 0: {np.where(obs <= 0)}")
                print(f"  bad idc >= 1: {np.where(obs >= 1)}")
            dict_key = f"S{si}_T{ti}_obs{oi}"
            sim_data[dict_key] = obs
            flat_observations[oi,:] = obs

        if make_plot:
            plt.figure()
            plt.imshow(flat_observations, cmap=plt.cm.plasma)
            plt.xlabel('Edge')
            plt.ylabel('Observation')
            title = f"flat observation comparison"
            if sigma_beta_T is not None:
                title = f"{title} for sigma/bound={invlogit(sigma_beta_T):.2f}"
            plt.title(title)
            plt.colorbar()
            plt.tight_layout()
            savefile = get_filename_with_ext(f"flat_obs_S{si}_T{ti}_N{N}_n_obs{n_observations}{sbt_txt}", ext='png', folder=figure_output_folder)
            plt.savefig(savefile)
            plt.close()

        ## Make scatter plot of prior to show original network
        if make_plot:
            plt.figure()
            plt.scatter(gt_prior[latpos][:,0], gt_prior[latpos][:,1], c='k', s=0.5)
            title = f"Ground truth prior for S{si} T{ti}\n{N} nodes, {n_observations} observations"
            if sigma_beta_T is not None:
                title = f"{title}, sigma/bound {invlogit(sigma_beta_T):.2f}"
            plt.title(title)
            plt.tight_layout()
            savefile = get_filename_with_ext(f"gt_prior_S{si}_T{ti}_N_{N}_n_obs_{n_observations}{sbt_txt}", ext='png', folder=figure_output_folder)
            plt.savefig(savefile)
            plt.close()

with open(output_data_filename, 'wb') as f:
    pickle.dump(sim_data, f)
