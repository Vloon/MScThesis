"""
Calling this file creates simulated data by sampling a ground truth prior, together with observations from that prior. These are both saved, and plotted. 
"""

## Basics
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import pickle

## Typing
from jax._src.prng import PRNGKeyArray
from jax._src.typing import ArrayLike
from typing import Callable, Tuple

## Self-made functions
from helper_functions import get_cmd_params, set_GPU, open_taskfile, get_safe_folder, get_filename_with_ext, invlogit, is_valid, create_task_file
from plotting_functions import plot_correlations

from binary_euclidean_LSM import sample_prior as bin_euc_sample_prior, sample_observation as bin_euc_sample_observation, get_det_params as bin_euc_det_params
from binary_hyperbolic_LSM import sample_prior as bin_hyp_sample_prior, sample_observation as bin_hyp_sample_observation, get_det_params as bin_hyp_det_params
from continuous_euclidean_LSM import sample_prior as con_euc_sample_prior, sample_observation as con_euc_sample_observation, get_det_params as con_euc_det_params
from continuous_hyperbolic_LSM import sample_prior as con_hyp_sample_prior, sample_observation as con_hyp_sample_observation, get_det_params as con_hyp_det_params

### Create cmd argument list (arg_name, var_name, type, default[OPT], nargs[OPT]).
###  - arg_name is the name of the argument in the command line.
###  - var_name is the name of the variable in the returned dictionary (which we re-use as variable name here).
###  - type is the data-type of the variable.
###  - default is the default value it takes if nothing is passed to the command line. This argument is only optional if type is bool, where the default is always False.
###  - nargs is the number of arguments, where '?' (default) is 1 argument, '+' concatenates all arguments to 1 list. This argument is optional.
arguments = [('-df', 'base_data_filename', str, 'prior_data'), # the most basic version of the filename of the saved data
             ('-of', 'output_folder', str, 'Data/prior_sim'), # folder where to dump data
             ('-fof', 'figure_output_folder', str, 'Figures/prior_sim'), # folder where to dump figures
             ('-tf', 'task_filename', str, 'task_list_prior_sim'), # filename of the list of task names WITHOUT EXTENSION
             ('-tdelim', 'task_delimiter', str, ','), # delimiter of the tasks in the task file
             ('-nsub', 'n_subjects', int, 1), # number of "subjects"
             ('-ntasks', 'n_tasks', int, 1), # number of "tasks"
             ('-nobs', 'n_observations', int, 2), # number of observations sampled
             ('-eps', 'eps', float, 1e-5), # clipping value in a number of functions
             ('-obseps', 'obs_eps', float,1e-7), # clipping value for the observations in the continuous models
             ('-mu', 'mu', float, 0.), # mean for the positions' normal distribution
             ('-sig', 'sigma', float, 1.), # standard deviation for the positions' normal distribution
             ('-musig', 'mu_sigma', float, 0.), # mean of the logit-transformed std of the beta distribution
             ('-sigsig', 'sigma_sigma', float, 1.), # standard deviation of the logit-transformed std of the beta distribution
             ('-sbt', 'sigma_beta_T', float, None), # logit transformed standard deviation of the beta distribution. If this one is given, sigma is taken from here.
             ('-N', 'N', int, 164), # number of nodes
             ('-D', 'D', int, 2), # dimensionality of the latent space
             ('-et', 'edge_type', str, 'con'), # edge type ('con' or 'bin')
             ('-geo', 'geometry', str, 'hyp'), # LS geometry ('hyp' or 'euc')
             ('-seed', 'seed', int, 0), # PRNGKey seed
             ('--print', 'do_print', bool), # whether to print (mostly errors)
             ('--plot', 'make_plot', bool), # whether to make plots
             ('-cm', 'cmap', str, 'bwr'), # colormap for the observation plots
             ('-gpu', 'gpu', str, ''),  # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]

## Get arguments from command line.
global_params = get_cmd_params(arguments)
set_GPU(global_params['gpu']) ## MUST BE RUN FIRST
base_data_filename = global_params['base_data_filename']
edge_type = global_params['edge_type']
assert edge_type in ['bin', 'con'], f"Edge type must be 'bin' (binary) or 'con' (continuous) but is {edge_type}"
geometry = global_params['geometry']
assert geometry in ['euc', 'hyp'], f"Geometry must be 'euc' (Euclidean) or 'hyp' (hyperbolic) but is {geometry}"
N = global_params['N']
M = N*(N-1)//2
D = global_params['D']
output_folder = get_safe_folder(f"{global_params['output_folder']}/{edge_type}_{geometry}")
figure_output_folder = get_safe_folder(f"{global_params['figure_output_folder']}/{edge_type}_{geometry}")
task_filename = global_params['task_filename']
task_delimiter = global_params['task_delimiter']
n_subjects = global_params['n_subjects']
n_tasks = global_params['n_tasks']
n_observations = global_params['n_observations']
eps = global_params['eps']
obs_eps = global_params['obs_eps']
mu = global_params['mu']
sigma = global_params['sigma']
mu_sigma = global_params['mu_sigma']
sigma_sigma = global_params['sigma_sigma']
sigma_beta_T = global_params['sigma_beta_T']
do_print = global_params['do_print']
make_plot = global_params['make_plot']
cmap = global_params['cmap']
## Use JAX functions only after setting the GPU, otherwise it will use all GPUs by default.
key = jax.random.PRNGKey(global_params['seed'])

## Define a number of variables based on geometry or edge type
latpos = '_z' if geometry == 'hyp' else 'z'
sample_prior_dict = {
                     'bin_euc':bin_euc_sample_prior,
                     'bin_hyp':bin_hyp_sample_prior,
                     'con_euc':con_euc_sample_prior,
                     'con_hyp':con_hyp_sample_prior,
                    }
sample_prior_func = sample_prior_dict[f"{edge_type}_{geometry}"]

sample_obs_dict = {
                   'bin_euc':bin_euc_sample_observation,
                   'bin_hyp':bin_hyp_sample_observation,
                   'con_euc':con_euc_sample_observation,
                   'con_hyp':con_hyp_sample_observation,
                  }
sample_obs_func = sample_obs_dict[f"{edge_type}_{geometry}"]

get_det_params_dict = {
                        'bin_euc':bin_euc_det_params,
                        'bin_hyp':bin_hyp_det_params,
                        'con_euc':con_euc_det_params,
                        'con_hyp':con_hyp_det_params,
                       }
get_det_params = get_det_params_dict[f"{edge_type}_{geometry}"]

## Dictionary containing a number of universally used variables
default_params = {'mu':mu, 'sigma':sigma, 'mu_sigma':mu_sigma, 'sigma_sigma':sigma_sigma, 'eps':eps, 'obs_eps':obs_eps}

## Allow overwriting of certain parameters. If value is None, it will be ignored.
overwrite_param_dict = {
                        'bin_euc': {},
                        'bin_hyp': {},
                        'con_euc': {'overwrite_sigma_val':sigma_beta_T},
                        'con_hyp': {'overwrite_sigma_val':sigma_beta_T},
                       }
overwrite_params = overwrite_param_dict[f"{edge_type}_{geometry}"]

# def create_task_file(filename:str, n_tasks:int=n_tasks, n_observations:int=n_observations, delim:str=task_delimiter) -> None:
#     """
#     Creates a task file which can later be used by load_observations.
#     PARAMS:
#     filename : name of the task file
#     n_tasks : number of "tasks"
#     n_observations : number of observations
#     delim : delimiter between the tasks and observations in the task file
#     """
#     task_str = delim.join([f'T{i}' for i in range(n_tasks)])
#     obs_str = delim.join([f'obs{i}' for i in range(n_observations)])
#     with open(filename, 'w') as f:
#         f.write(f'{task_str}\n')
#         f.write(obs_str)

def create_data(key:PRNGKeyArray,
                shape:tuple = (N,D),
                n_observations:int = n_observations,
                sample_prior_func:Callable = sample_prior_func,
                sample_obs_func:Callable = sample_obs_func,
                params:dict = default_params,
                overwrite_params:dict = overwrite_params
                ) -> Tuple[PRNGKeyArray, dict, jnp.array]:
    """
    Samples a prior as ground truth and samples observations.
    PARAMS:
    key : random key for jax functions
    shape : number of nodes by latent space dimensions
    n_observations : number of observations per ground truth
    sample_prior_func : function to sample the prior
    sample_obs_func : function to sample observations from the prior
    params : dictionary containing the parameters for the sampling functions
    overwrite_params : dictionary containing all parameters you want to overwrite
    """
    ## Sample prior
    key, gt_prior = sample_prior_func(key, shape, **params, **overwrite_params)
    ## Sample observations
    key, observations = sample_obs_func(key, gt_prior, n_observations, **params)
    return key, gt_prior, observations

## Create correct filenames
sbt_txt = f"_sbt_{sigma_beta_T:.1f}" if sigma_beta_T is not None else ''
task_filename = f"{task_filename}_N_{N}_n_obs_{n_observations}{sbt_txt}"
output_task_filename = get_filename_with_ext(task_filename, ext='txt', folder=output_folder)
data_filename = f"{base_data_filename}_N_{N}_n_obs_{n_observations}{sbt_txt}" 
output_data_filename = get_filename_with_ext(data_filename, folder=output_folder)

create_task_file(output_task_filename, n_tasks, n_observations, task_delimiter)

## Text and color QOL
ppt = {'bin':'binary', 'con':'continuous', 'euc':'Euclidean', 'hyp':'hyperbolic'}
colors = [plt.cm.plasma(i) for i in np.linspace(0, 1, n_observations)]

## We use "subjects"/"tasks" in order to be in line with the structure of the embedding code, which expects subjects/tasks.
## We save observations in a dictionary to be in line with the structure of the embedding code.
sim_data = {}
for si in range(1,n_subjects+1):
    for ti in range(n_tasks):
        ## Create the ground truth prior and observations
        key, gt_prior, observations = create_data(key)

        ## Save the ground truth prior
        gt_filename = get_filename_with_ext(f"gt_prior_{edge_type}_{geometry}_S{si}_T{ti}_N_{N}_n_obs_{n_observations}{sbt_txt}", folder=output_folder)
        with open(gt_filename, 'wb') as f:
            pickle.dump(gt_prior, f)

        ## Save the observations to the dictionary
        for oi, obs in enumerate(observations):
            dict_key = f"S{si}_T{ti}_obs{oi}"
            sim_data[dict_key] = obs

        if make_plot:
            ## Plot flat observations
            plt.figure()
            plt.imshow(observations, cmap=plt.cm.plasma, aspect='auto')
            plt.xlabel('Edge')
            plt.ylabel('Observation')
            title = f"Flat observation ({ppt[edge_type]} {ppt[geometry]})"
            if sigma_beta_T is not None:
                title = f"{title} for sigma/bound={invlogit(sigma_beta_T):.2f}"
            plt.title(title)
            if edge_type == 'con':
               plt.colorbar()
            plt.tight_layout()
            savefile = get_filename_with_ext(f"flat_obs_{edge_type}_{geometry}_S{si}_T{ti}_N{N}_n_obs{n_observations}{sbt_txt}", ext='png', folder=figure_output_folder)
            plt.savefig(savefile)
            plt.close()

            ## Make scatter plot of prior to show original network
            plt.figure()
            plt.scatter(gt_prior[latpos][:,0], gt_prior[latpos][:,1], c='k', s=0.5)
            title = f"Ground truth prior ({ppt[edge_type]} {ppt[geometry]}) \nS{si} T{ti}\n{N} nodes"
            if sigma_beta_T is not None:
                title = f"{title}, sigma/bound {invlogit(sigma_beta_T):.2f}"
            plt.title(title)
            plt.tight_layout()
            savefile = get_filename_with_ext(f"gt_prior_pos_S{si}_T{ti}_N_{N}", ext='png', folder=figure_output_folder)
            plt.savefig(savefile)
            plt.close()

            ## Make distance distribution plot
            n_bins = 100
            plt.figure()
            distances = get_det_params(gt_prior[latpos])['d']
            hist, bins = jnp.histogram(distances, bins=n_bins, density=True)
            plt.stairs(hist, bins, color='0.1', fill=True)
            plt.xlabel('Distance')
            plt.ylabel('Density')
            savefile = get_filename_with_ext(f"gt_prior_distances_S{si}_T{ti}_N_{N}", ext='png', folder=figure_output_folder)
            plt.savefig(savefile)
            plt.close()

            ## Make edge weight distribution plot
            plt.figure()
            for oi, obs in enumerate(observations):
                hist, bins = jnp.histogram(obs, bins=n_bins, density=True)
                plt.stairs(hist, bins, fill=True, color=colors[oi], label = f"obs{oi}", alpha=.4)
            plt.xlabel('Edge weight')
            plt.ylabel('Density')
            savefile = get_filename_with_ext(f"gt_prior_edgeweights_S{si}_T{ti}_N_{N}_nobs_{n_observations}", ext='png', folder=figure_output_folder)
            plt.savefig(savefile)
            plt.close()

## Save the simulated observations
with open(output_data_filename, 'wb') as f:
    pickle.dump(sim_data, f)
