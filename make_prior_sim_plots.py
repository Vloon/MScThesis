"""
Calling this function creates a number of plots, used to evaluate the prior simulation experiment.
"""

## Basics
import numpy as np
import matplotlib.pyplot as plt
import pickle
import jax
import jax.numpy as jnp
import os

## Self-made functions
from helper_functions import get_filename_with_ext, load_observations, get_cmd_params, set_GPU, get_safe_folder, get_attribute_from_trace, get_trace_correlation, invlogit
from bookstein_methods import add_bkst_to_smc_trace, get_bookstein_anchors
from plotting_functions import plot_metric, plot_sigma_convergence
from binary_euclidean_LSM import get_det_params as bin_euc_det_params
from binary_hyperbolic_LSM import get_det_params as bin_hyp_det_params
from continuous_euclidean_LSM import get_det_params as con_euc_det_params
from continuous_hyperbolic_LSM import get_det_params as con_hyp_det_params

### Create cmd argument list (arg_name, var_name, type, default[OPT], nargs[OPT]).
###  - arg_name is the name of the argument in the command line.
###  - var_name is the name of the variable in the returned dictionary (which we re-use as variable name here).
###  - type is the data-type of the variable.
###  - default is the default value it takes if nothing is passed to the command line. This argument is only optional if type is bool, where the default is always False.
###  - nargs is the number of arguments, where '?' (default) is 1 argument, '+' concatenates all arguments to 1 list. This argument is optional.
arguments = [('-eif', 'embedding_input_folder', str, 'Embeddings/prior_sim'), # base input folder of the embeddings
             ('-ebn', 'embedding_basename', str, 'embedding'), # base name of the embedding files
             ('-pbn', 'prior_basename', str, 'gt_prior'), # base name of the ground truth file
             ('-dif', 'data_input_folder', str, 'Data/prior_sim'), # base input folder of the data
             ('-conbdf', 'con_base_data_filename', str, 'processed_data_downsampled_evenly_spaced'), # the most basic version of the filename of the continuous saved data
             ('-binbdf', 'bin_base_data_filename', str, 'binary_data_downsampled_evenly_spaced_max_0.05unconnected'), # the filename of the binary saved data
             ('-np', 'n_particles', int, 2000), # number of particles used in the embedding
             ('-nm', 'n_mcmc_steps', int, 500), # number of mcmc steps used in the embedding
             ('-tf', 'task_filename', str, 'task_list'), # filename of the list of task names
             ('-of', 'output_folder', str, 'Figures/prior_sim'), # folder where to dump figures
             ('-D', 'D', int, 2), # latent space dimensionality
             ('-ns', 'n_subjects', int, 1), # number of subjects
             ('-nt', 'n_tasks', int, 1), # number of tasks
             ('-Ns', 'N_vals', int, [10, 20, 50, 100, 164], '+'), # list of numbers of nodes
             ('-nobs', 'n_observations', int, [1, 2, 3, 5, 10], '+'), # list of numbers of observations
             ('-sigs', 'sbt_vals', float, [-4.0, -2.0, -1.0, 0.0, 1.0], '+'), # list of sigma_beta_T values (round off to 1 digit pls)
             ('-et', 'edge_type', str, 'con'), # edge type ('con' or 'bin')
             ('-geo', 'geometry', str, 'hyp'), # LS geometry ('hyp' or 'euc')
             ('--plotidv', 'plot_individual', bool), # whether to plot 1 random individual particle's gt_dist vs sample_dist scatterplot
             ('--plotsig', 'plot_sigma', bool), # whether to plot sigma convergence
             ('-nbins', 'n_bins', int, 100), # number of bins in the sigma/bound histogram
             ('-alpha', 'alpha', float, 0.05), # alpha for plotting the full distribution
             ('-lfs', 'label_fontsize', float, 20),  # fontsize of labels (and legend)
             ('-tfs', 'tick_fontsize', float, 16),  # fontsize of the tick labels
             ('-wsz', 'wrapsize', float, 20),  # wrapped text width
             ('--usesave', 'use_save', bool), # whether to use a saved correlations pickle file
             ('-pklsf', 'pkl_savefile', str, 'prior_sim_corrs'), # pickle savefile name
             ('-gpu', 'gpu', str, ''),  # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]

## Get arguments from command line.
global_params = get_cmd_params(arguments)
set_GPU(global_params['gpu']) ## MUST BE RUN FIRST
label_fontsize = global_params['label_fontsize']
tick_fontsize = global_params['tick_fontsize']
wrapsize = global_params['wrapsize']
edge_type = global_params['edge_type']
geometry = global_params['geometry']
assert edge_type in ['bin', 'con'], f"edge_type must be 'bin' or 'con' but is {edge_type}"
assert geometry in ['euc', 'hyp'], f"geometry must be in 'euc' or 'hyp' but is {geometry}"
base_data_filename = global_params['bin_base_data_filename'] if edge_type == 'bin' else global_params['con_base_data_filename']
n_particles = global_params['n_particles']
n_mcmc_steps = global_params['n_mcmc_steps']
base_embedding_input_folder = f"{global_params['embedding_input_folder']}/{edge_type}_{geometry}"
embedding_basename = global_params['embedding_basename']
data_input_folder = f"{global_params['data_input_folder']}/{edge_type}_{geometry}"
task_filename = global_params['task_filename']
prior_basename = global_params['prior_basename']
output_folder = f"{get_safe_folder(global_params['output_folder'])}/{edge_type}_{geometry}"
D = global_params['D']
n_subjects = global_params['n_subjects']
n_tasks = global_params['n_tasks']
N_vals = global_params['N_vals']
N_vals.sort()
n_observations = global_params['n_observations']
n_observations.sort()
sbt_vals = global_params['sbt_vals']
sbt_vals.sort()
if edge_type == 'bin':
    sbt_vals = []
plot_individual = global_params['plot_individual']
plot_sigma = global_params['plot_sigma']
if plot_sigma:
    assert edge_type == 'con', f"To plot sigma, edge_type must be 'con' but is '{edge_type}'"
n_bins = global_params['n_bins']
alpha = global_params['alpha']
use_save = global_params['use_save']
pkl_savefile = get_filename_with_ext(global_params['pkl_savefile'], folder=data_input_folder)

## Define a number of variables based on geometry or edge type
det_params_dict = {
                   'bin_euc':bin_euc_det_params,
                   'bin_hyp':bin_hyp_det_params,
                   'con_euc':con_euc_det_params,
                   'con_hyp':con_hyp_det_params,
                   }
det_params_func = det_params_dict[f"{edge_type}_{geometry}"]
latpos = '_z' if geometry == 'hyp' else 'z'

## Create the full correlation matrix, contining all combinations of subjects, tasks, values for N, number of observations, number of noise terms (if continuous), and number of particles
if edge_type == 'con':
    full_correlations = np.zeros((n_subjects, n_tasks, len(N_vals), len(n_observations), len(sbt_vals), n_particles))
else:
    full_correlations = np.zeros((n_subjects, n_tasks, len(N_vals), len(n_observations), n_particles))

total = n_subjects * n_tasks * len(N_vals) * len(n_observations) 
if edge_type == 'con':
    total *= len(sbt_vals)
i = 0

## Load saved correlation data if it exists, used since creating the full correlation matrix takes long
no_save = False
if use_save:
    if os.path.exists(pkl_savefile):
        with open(pkl_savefile, 'rb') as f:
            full_correlations = pickle.load(f)
        print(f"Loaded data from {pkl_savefile}")
    else:
        no_save = True
        print(f"Tried to load data from {pkl_savefile} but could not find file")

## Create the full correlation matrix
if not use_save or no_save:
    for si, n_sub in enumerate(range(1,n_subjects+1)):
        for ti in range(n_tasks):
            for ni, N in enumerate(N_vals):
                M = N*(N-1)//2
                for oi, n_obs in enumerate(n_observations):
                    if edge_type == 'con':
                        for sbt_i, sbt in enumerate(sbt_vals):
                            i += 1
                            print(f"Iter {i}/{total}\tS{n_sub} T{ti} N={N} n_obs={n_obs} sbt={sbt}")

                            ## Get ground truth prior
                            gt_filename = get_filename_with_ext(f"{prior_basename}_{edge_type}_{geometry}_S{n_sub}_T{ti}_N_{N}_n_obs_{n_obs}_sbt_{sbt}", folder=data_input_folder)
                            with open(gt_filename, 'rb') as gt_f:
                                gt_prior = pickle.load(gt_f)
                            ## Get ground truth latent positions and distances
                            gt_latent_positions = gt_prior[latpos]
                            gt_distance = det_params_func(gt_latent_positions)['d']

                            ## Get learned embedding trace
                            embedding_input_folder = f"{base_embedding_input_folder}/N_{N}_n_obs_{n_obs}_sbt_{sbt}/{n_particles}p{n_mcmc_steps}s"
                            embedding_filename = get_filename_with_ext(f"{edge_type}_{geometry}_S{n_sub}_T{ti}_{embedding_basename}", folder=embedding_input_folder)
                            with open(embedding_filename, 'rb') as e_f:
                                embedding = pickle.load(e_f)
                            ## Get learned distances
                            distance_trace = get_attribute_from_trace(embedding.particles[latpos], det_params_func, 'd')

                            ## Save correlations
                            corrs = get_trace_correlation(distance_trace, gt_distance)
                            full_correlations[si, ti, ni, oi, sbt_i, :] = corrs

                            ## Plot a random particle's ground truth vs learned distances scatterplot
                            if plot_individual:
                                p_idx = np.random.randint(n_particles)
                                plt.figure()
                                plt.scatter(gt_distance, distance_trace[p_idx], c='k', s=0.5, alpha=0.8)
                                plt.xlabel('Ground truth distance', fontsize=label_fontsize)
                                plt.xticks(fontsize=tick_fontsize)
                                plt.ylabel('Learned distance', fontsize=label_fontsize)
                                plt.yticks(fontsize=tick_fontsize)
                                savefile = get_filename_with_ext(f"gt_vs_emb_dist_S{n_sub}_T{ti}_N{N}_n_obs{n_obs}_sbt{sbt}_p{p_idx}", ext='png', folder=output_folder)
                                plt.savefig(savefile, bbox_inches='tight')
                                plt.close()

                            if plot_sigma:
                                ## Load sigma data
                                sigma_filename = get_filename_with_ext(f"{edge_type}_{geometry}_sbt_N_{N}_nobs_{n_obs}_sbt_{sbt:.1f}_S{n_sub}_T{ti}_{base_data_filename}", folder=f"{data_input_folder}/sbt_traces")
                                with open(sigma_filename, 'rb') as sf:
                                    sigma_chain = pickle.load(sf)

                                ## Plot evolution of sigma
                                plt.figure(figsize=(20, 10))
                                xsize=2
                                ax = plt.gca()
                                ax = plot_sigma_convergence(sigma_chain, sbt, ax, n_bins, label_fontsize=xsize*label_fontsize, tick_fontsize=xsize*tick_fontsize)
                                plt.xlabel('SMC iteration', fontsize=xsize*label_fontsize)
                                plt.ylabel(r'$\sigma$/bound', fontsize=xsize*label_fontsize)
                                savefile = get_filename_with_ext(f"sbt_convergence_S{n_sub}_T{ti}_N{N}_n_obs{n_obs}_sbt{sbt}", ext='png', folder=output_folder)
                                plt.savefig(savefile, bbox_inches='tight')
                                plt.close()
                    else: # Getting here means the edge type is binary
                        i += 1
                        print(f"Iter {i}/{total}\tS{n_sub} T{ti} N={N} n_obs={n_obs}")

                        ## Get ground truth prior
                        gt_filename = get_filename_with_ext(f"{prior_basename}_{edge_type}_{geometry}_S{n_sub}_T{ti}_N_{N}_n_obs_{n_obs}", folder=data_input_folder)
                        with open(gt_filename, 'rb') as gt_f:
                            gt_prior = pickle.load(gt_f)
                        ## Get ground truth latent positions and distances
                        gt_latent_positions = gt_prior[latpos]
                        gt_distance = det_params_func(gt_latent_positions)['d']

                        ## Get learned embedding trace
                        embedding_input_folder = f"{base_embedding_input_folder}/N_{N}_n_obs_{n_obs}/{n_particles}p{n_mcmc_steps}s"
                        embedding_filename = get_filename_with_ext(f"{edge_type}_{geometry}_S{n_sub}_T{ti}_{embedding_basename}", folder=embedding_input_folder)
                        with open(embedding_filename, 'rb') as e_f:
                            embedding = pickle.load(e_f)

                        ## Get learned distances
                        distance_trace = get_attribute_from_trace(embedding.particles[latpos], det_params_func, 'd')

                        ## Save correlations
                        corrs = get_trace_correlation(distance_trace, gt_distance)
                        full_correlations[si, ti, ni, oi, :] = corrs

                        ## Plot a random particle's ground truth vs learned distances scatterplot
                        if plot_individual:
                            p_idx = np.random.randint(n_particles)
                            plt.figure()
                            plt.scatter(gt_distance, distance_trace[p_idx], c='k', s=0.5, alpha=0.8)
                            plt.xlabel('Ground truth distance', fontsize=label_fontsize)
                            plt.ylabel('Learned distance', fontsize=label_fontsize)
                            savefile = get_filename_with_ext(f"gt_vs_emb_dist_S{n_sub}_T{ti}_N{N}_n_obs{n_obs}_p{p_idx}", ext='png', folder=output_folder)
                            plt.savefig(savefile, bbox_inches='tight')
                            plt.close()

    ## Save data for re-use
    with open(pkl_savefile, 'wb') as f:
        pickle.dump(full_correlations, f)

## Axes of the corresponding dimensions: subtracts 2 for (subject, task) axes
N_ax = 0
obs_ax = 1
sbt_ax = 2
p_ax = 3

## General plotting parameters
avg_params = {'capsize':3, 'c':'k'}
dist_params = {'c':'k', 'alpha':alpha}

## Values to get decent autoscaling width.
boxplot_margin = 1/10 # Margin between sets of boxes as a percentage of the total width
N_dists = [N_vals[i+1] - N_vals[i] for i in range(len(N_vals) - 1)]
min_N_dist, max_N_dist = np.min(N_dists), N_vals[-1] - N_vals[0]
x_margin = max_N_dist * boxplot_margin
plot_N_vals = [N+i*x_margin for i, N in enumerate(N_vals)]
loc_increment = min_N_dist/len(sbt_vals) if edge_type == 'con' else np.nan
o_loc_increment, midpoint = min_N_dist/len(n_observations), min_N_dist/2
if edge_type == 'con':
    location_offset = [loc_increment*i-midpoint for i in range(len(sbt_vals))]
o_location_offset = [o_loc_increment*i-midpoint for i in range(len(n_observations))]
box_width = min_N_dist/(len(sbt_vals)+1) # Add one to leave space between boxes
o_box_width = min_N_dist/(len(n_observations)+1) # Add one to leave space between boxes

## Get colors lists
sbt_colors = [plt.cm.plasma(i) for i in np.linspace(0, 1, len(sbt_vals))]
obs_colors = [plt.cm.plasma(i) for i in np.linspace(0, 1, len(n_observations))]
N_colors = [plt.cm.plasma(i) for i in np.linspace(0, 1, len(N_vals))]

for si, n_sub in enumerate(range(1,n_subjects+1)):
    sub_correlations = jnp.mean(full_correlations[si], axis=0) # Take the average correlations over the tasks

    ## Correlation distribution for different Ns
    plt.figure()
    for ni, N in enumerate(N_vals):
        full_dist_N = np.ravel(sub_correlations[ni])
        plt.scatter(np.repeat(N, len(full_dist_N)), full_dist_N, **dist_params)
    plt.xlabel('Number of nodes', fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.ylabel('Correlation', fontsize=label_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    savefile = get_filename_with_ext(f"dist_corr_N_S{n_sub}",ext='png', folder=output_folder)
    plt.savefig(savefile, bbox_inches='tight')
    plt.close()

    ## Correlation distribution for different numbers of observations
    plt.figure()
    for oi, n_obs in enumerate(n_observations):
        full_dist_obs = np.ravel(sub_correlations[:, oi])
        plt.scatter(np.repeat(n_obs, len(full_dist_obs)), full_dist_obs, **dist_params)
    plt.xlabel('Number of observations', fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.ylabel('Correlation', fontsize=label_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    savefile = get_filename_with_ext(f"dist_corr_n_obs_S{n_sub}", ext='png', folder=output_folder)
    plt.savefig(savefile, bbox_inches='tight')
    plt.close()

    if edge_type == 'con':
        ## Correlation distribution for different noise terms
        plt.figure()
        for sbt_i, sbt in enumerate(sbt_vals):
            full_dist_sbt = np.ravel(sub_correlations[:, :, sbt_i])
            plt.scatter(np.repeat(invlogit(sbt), len(full_dist_sbt)), full_dist_sbt, **dist_params)
        plt.xlabel(r'$\sigma$/bound', fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.ylabel('Correlation', fontsize=label_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        savefile = get_filename_with_ext(f"dist_corr_sbt_S{n_sub}", ext='png', folder=output_folder)
        plt.savefig(savefile, bbox_inches='tight')
        plt.close()

    ## Plot correlation averaged over N
    ax = (obs_ax, sbt_ax, p_ax) if edge_type == 'con' else (obs_ax, p_ax-1)
    avg_N_corr = np.mean(sub_correlations, axis=ax)
    std_N_corr = np.std(sub_correlations, axis=ax)
    plt.figure()
    plt.errorbar(N_vals, avg_N_corr, yerr=std_N_corr, **avg_params)
    plt.xlabel('Number of nodes', fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.ylabel('Correlation', fontsize=label_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    savefile = get_filename_with_ext(f"avg_dist_corr_N_S{n_sub}",ext='png', folder=output_folder)
    plt.savefig(savefile, bbox_inches='tight')
    plt.close()

    ## Plot correlation averaged over number of observations
    ax = (N_ax, sbt_ax, p_ax) if edge_type == 'con' else (N_ax, p_ax-1) # p_ax-1 is sbt_ax but this is clearer to me
    avg_obs_corr = np.mean(sub_correlations, axis=ax)
    std_obs_corr = np.std(sub_correlations, axis=ax)
    plt.figure()
    plt.errorbar(n_observations, avg_obs_corr, yerr=std_obs_corr, **avg_params)
    plt.xlabel('Number of observations', fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.ylabel('Correlation', fontsize=label_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    savefile = get_filename_with_ext(f"avg_dist_corr_n_obs_S{n_sub}", ext='png', folder=output_folder)
    plt.savefig(savefile, bbox_inches='tight')
    plt.close()

    if edge_type == 'con':
        ## Plot correlation averaged over noise terms
        avg_sbt_corr = np.mean(sub_correlations, axis=(N_ax, obs_ax, p_ax))
        std_sbt_corr = np.std(sub_correlations, axis=(N_ax, obs_ax, p_ax))
        plt.figure()
        plt.errorbar([invlogit(sbt) for sbt in sbt_vals], avg_sbt_corr, yerr=std_sbt_corr, **avg_params)
        plt.xlabel(r'$\sigma$/bound', fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.ylabel('Correlation', fontsize=label_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        savefile = get_filename_with_ext(f"avg_dist_corr_sbt_S{n_sub}", ext='png', folder=output_folder)
        plt.savefig(savefile, bbox_inches='tight')
        plt.close()

        ## Plot average over only particles, a seperate line for each number of observations, with x=N, new plot per noise term
        for sbt_i, sbt in enumerate(sbt_vals):
            plt.figure()
            for oi, n_obs in enumerate(n_observations):
                avg_corr = np.mean(sub_correlations[:, oi, sbt_i, :], axis=-1)
                std_corr = np.std(sub_correlations[:, oi, sbt_i, :], axis=-1)
                plt.errorbar(N_vals, avg_corr, yerr=std_corr, capsize=avg_params['capsize'], c=obs_colors[oi], label=f"n_obs={n_obs}")
            if geometry == 'hyp':
                plt.legend(fontsize=label_fontsize)
            plt.xlabel('Number of nodes', fontsize=label_fontsize)
            plt.xticks(fontsize=tick_fontsize)
            if geometry == 'euc':
                plt.ylabel('Correlation', fontsize=label_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            savefile = get_filename_with_ext(f"avg_dist_corr_N_by_n_obs_S{n_sub}_sbt{sbt}", ext='png', folder=output_folder)
            plt.savefig(savefile, bbox_inches='tight')
            plt.close()

        ## Plot average over only particles, a seperate line for each N, with x=n_obs, new plot per noise term
        for sbt_i, sbt in enumerate(sbt_vals):
            plt.figure()
            for ni, N in enumerate(N_vals):
                avg_corr = np.mean(sub_correlations[ni, :, sbt_i, :], axis=-1)
                std_corr = np.std(sub_correlations[ni, :, sbt_i, :], axis=-1)
                plt.errorbar(n_observations, avg_corr, yerr=std_corr, capsize=avg_params['capsize'], c=N_colors[ni], label=f"N={N}")
            if geometry == 'hyp':
                plt.legend(fontsize=label_fontsize)
            plt.xlabel('Number of observations', fontsize=label_fontsize)
            plt.xticks(fontsize=tick_fontsize)
            if geometry == 'euc':
                plt.ylabel('Correlation', fontsize=label_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            savefile = get_filename_with_ext(f"avg_dist_corr_n_obs_by_N_S{n_sub}_sbt{sbt}", ext='png', folder=output_folder)
            plt.savefig(savefile, bbox_inches='tight')
            plt.close()

        ## Plot average over only particles, a seperate line for each noise term, with x=N, new plot per n_obs
        for oi, n_obs in enumerate(n_observations):
            plt.figure()
            for sbt_i, sbt in enumerate(sbt_vals):
                avg_corr = np.mean(sub_correlations[:, oi, sbt_i, :], axis=-1)
                std_corr = np.std(sub_correlations[:, oi, sbt_i, :], axis=-1)
                plt.errorbar(N_vals, avg_corr, yerr=std_corr, capsize=avg_params['capsize'], c=sbt_colors[sbt_i], label=r"$\sigma$/bound={:.2f}".format(invlogit(sbt)))
            if geometry == 'hyp':
                plt.legend(fontsize=label_fontsize, bbox_to_anchor=(1.0,1.0))
            plt.xlabel('Number of nodes', fontsize=label_fontsize)
            plt.xticks(fontsize=tick_fontsize)
            if geometry == 'euc':
                plt.ylabel('Correlation', fontsize=label_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            savefile = get_filename_with_ext(f"avg_dist_corr_N_by_sbt_S{n_sub}_n_obs{n_obs}", ext='png', folder=output_folder)
            plt.savefig(savefile, bbox_inches='tight')
            plt.close()

        ## The number of observations has clear effect: for each number of observations, we plot a boxplot per N, per noise term. The location of the boxplot is mostly determined by N, with a small offset for different noise terms.
        for oi, n_obs in enumerate(n_observations):
            plt.figure(figsize=(20,10))
            for sbt_i, sbt in enumerate(sbt_vals):
                for ni, N in enumerate(plot_N_vals):
                    corrs = sub_correlations[ni, oi, sbt_i, :]
                    location = N+location_offset[sbt_i]
                    bp = plt.boxplot(x=corrs,
                                     positions=[location],
                                     widths=box_width,
                                     notch=True,
                                     patch_artist=True,
                                     boxprops=dict(facecolor=sbt_colors[sbt_i]),
                                     medianprops=dict(color='black'),
                                     flierprops=dict(color='white', markeredgecolor='grey'))
                    if ni == 0:
                        bp['boxes'][0].set_label(fr"$\sigma$/bound={invlogit(sbt):.2f}")
            plt.xticks(ticks=plot_N_vals, labels=N_vals, fontsize=tick_fontsize)
            plt.xlim(plot_N_vals[0]-min_N_dist, plot_N_vals[-1]+min_N_dist)
            plt.xlabel('Number of nodes', fontsize=label_fontsize)
            plt.ylabel('Correlation', fontsize=label_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            plt.legend(fontsize=label_fontsize)
            savefile = get_filename_with_ext(f"box_dist_corr_S{n_sub}_n_obs{n_obs}", ext='png', folder=output_folder)
            plt.savefig(savefile, bbox_inches='tight')
            plt.close()

    else:
        ## Binary edges
        ## Plot average over only particles, a seperate line for each number of observations, with x=N
        plt.figure()
        for oi, n_obs in enumerate(n_observations):
            avg_corr = np.mean(sub_correlations[:, oi, :], axis=-1)
            std_corr = np.std(sub_correlations[:, oi, :], axis=-1)
            plt.errorbar(N_vals, avg_corr, yerr=std_corr, capsize=avg_params['capsize'], c=obs_colors[oi], label=f"n_obs={n_obs}")
        if geometry == 'hyp':
            plt.legend(fontsize=label_fontsize)
        plt.xlabel('Number of nodes', fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        if geometry == 'euc':
           plt.ylabel('Correlation', fontsize=label_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        savefile = get_filename_with_ext(f"avg_dist_corr_N_by_n_obs_S{n_sub}", ext='png', folder=output_folder)
        plt.savefig(savefile, bbox_inches='tight')
        plt.close()

        ## Plot average over only particles, a seperate line for each N, with x=number of observations
        plt.figure()
        for ni, N in enumerate(N_vals):
            avg_corr = np.mean(sub_correlations[ni, :, :], axis=-1)
            std_corr = np.std(sub_correlations[ni, :, :], axis=-1)
            plt.errorbar(n_observations, avg_corr, yerr=std_corr, capsize=avg_params['capsize'], c=N_colors[ni], label=f"N={N}")
        if geometry == 'hyp':
            plt.legend(fontsize=label_fontsize)
        plt.xlabel('Number of observations', fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        if geometry == 'euc':
            plt.ylabel('Correlation', fontsize=label_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        savefile = get_filename_with_ext(f"avg_dist_corr_n_obs_by_N_S{n_sub}", ext='png', folder=output_folder)
        plt.savefig(savefile, bbox_inches='tight')
        plt.close()

        ## The number of observations has clear effect: for each number of observations, we plot a boxplot per value of N.
        for oi, n_obs in enumerate(n_observations):
            plt.figure(figsize=(20, 10))
            for ni, N in enumerate(plot_N_vals):
                corrs = sub_correlations[ni, oi, :]
                location = N + o_location_offset[oi]
                bp = plt.boxplot(x=corrs,
                                 positions=[location],
                                 widths=o_box_width,
                                 notch=True,
                                 patch_artist=True,
                                 boxprops=dict(facecolor=obs_colors[oi]),
                                 medianprops=dict(color='black'),
                                 flierprops=dict(color='white', markeredgecolor='grey'))
                if ni == 0:
                    bp['boxes'][0].set_label(f"n_obs={n_obs}")
            plt.xticks(ticks=plot_N_vals, labels=N_vals, fontsize=tick_fontsize)
            plt.xlim(plot_N_vals[0] - min_N_dist, plot_N_vals[-1] + min_N_dist)
            plt.xlabel('Number of nodes', fontsize=label_fontsize)
            plt.ylabel('Correlation', fontsize=label_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            plt.legend()
            savefile = get_filename_with_ext(f"box_dist_corr_S{n_sub}", ext='png', folder=output_folder)
            plt.savefig(savefile, bbox_inches='tight')
            plt.close()