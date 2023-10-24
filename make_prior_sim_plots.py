import numpy as np
import matplotlib.pyplot as plt
import pickle
import jax
import jax.numpy as jnp
import os
import csv

from helper_functions import get_filename_with_ext, load_observations, get_cmd_params, set_GPU, get_safe_folder, get_attribute_from_trace, get_trace_correlation, invlogit
from continuous_hyperbolic_LSM import get_det_params as con_hyp_det_params
from bookstein_methods import add_bkst_to_smc_trace, get_bookstein_anchors
from plotting_functions import plot_metric

arguments = [('-eif', 'embedding_input_folder', str, 'Embeddings/prior_sim'), # base input folder of the embeddings
             ('-dif', 'data_input_folder', str, 'Data/prior_sim'), # base input folder of the data
             ('-np', 'n_particles', int, 1000), # number of particles used in the embedding
             ('-nm', 'n_mcmc_steps', int, 100), # number of mcmc steps used in the embedding
             ('-tf', 'task_filename', str, 'task_list'), # filename of the list of task names
             ('-of', 'output_folder', str, 'Figures/prior_sim'), # folder where to dump figures
             ('-D', 'D', int, 2), # latent space dimensionality
             ('-ns', 'n_subjects', int, 1), # number of subjects
             ('-nt', 'n_tasks', int, 1), # number of tasks
             ('-Ns', 'N_vals', int, [10, 20, 50, 100, 164], '+'), # list of numbers of nodes
             ('-nobs', 'n_observations', int, [1, 2, 3, 5, 10], '+'), # list of numbers of observations
             ('-sigs', 'sbt_vals', float, [-4.0, -2.0, 0.0, 1.0], '+'), # list of sigma_beta_T values (round off to 1 digit pls)
             ('-et', 'edge_type', str, 'con'), # edge type ('con' or 'bin')
             ('-geo', 'geometry', str, 'hyp'), # LS geometry ('hyp' or 'euc')
             ('--bkst', 'is_bookstein', bool), # whether the trace uses Bookstein anchors
             ('-bkstdist', 'bkst_dist', float, 0.3), # distance between the bookstein anchors.
             ('--log', 'do_log', bool), # whether to print correlations/R^2 value to a log file per subject/task/n_obs/N/sigma_beta_T
             ('--plotidv', 'plot_individual', bool), # whether to plot 1 random individual particle's gt_dist vs sample_dist scatterplot
             ('--plotsig', 'plot_sigma', bool), # whether to plot sigma convergence
             ('-nbins', 'n_bins', int, 100), # number of bins in the sigma/bound histogram
             ('-alpha', 'alpha', float, 0.05), # alpha for plotting the full distribution
             ('--usesave', 'use_save', bool), # whether to use a saved correlations pickle file
             ('-pklsf', 'pkl_savefile', str, 'prior_sim_corrs'), # pickle savefile name
             ('-gpu', 'gpu', str, ''),  # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]

global_params = get_cmd_params(arguments)
n_particles = global_params['n_particles']
n_mcmc_steps = global_params['n_mcmc_steps']
base_embedding_input_folder = global_params['embedding_input_folder']
data_input_folder = global_params['data_input_folder']
task_filename = global_params['task_filename']
output_folder = get_safe_folder(global_params['output_folder'])
D = global_params['D']
n_subjects = global_params['n_subjects']
n_tasks = global_params['n_tasks']
N_vals = global_params['N_vals']
N_vals.sort()
n_observations = global_params['n_observations']
n_observations.sort()
sbt_vals = global_params['sbt_vals']
sbt_vals.sort()
edge_type = global_params['edge_type']
geometry = global_params['geometry']
is_bookstein = global_params['is_bookstein']
bkst_dist = global_params['bkst_dist']
do_log = global_params['do_log']
plot_individual = global_params['plot_individual']
plot_sigma = global_params['plot_sigma']
if plot_sigma:
    assert edge_type == 'con', f"To plot sigma, edge_type must be 'con' but is '{edge_type}'"
n_bins = global_params['n_bins']
alpha = global_params['alpha']
use_save = global_params['use_save']
pkl_savefile = get_filename_with_ext(global_params['pkl_savefile'], folder=data_input_folder)
set_GPU(global_params['gpu'])

det_params_dict = {'con_hyp':con_hyp_det_params}
det_params_func = det_params_dict[f"{edge_type}_{geometry}"]

# n_subjects x n_tasks x n_Ns x n_obs x n_sigmas x n_particles
full_correlations = np.zeros((n_subjects, n_tasks, len(N_vals), len(n_observations), len(sbt_vals), n_particles))
idx_reference = np.empty((n_subjects, n_tasks, len(N_vals), len(n_observations), len(sbt_vals), 3))

latpos = '_z' if geometry == 'hyp' else 'z'

total = n_subjects * n_tasks * len(N_vals) * len(n_observations) * len(sbt_vals)
i = 0

x_tick_interval = 5 # every 5th iteration is ticked
x_offset = 0.5 # distance in between each histogram

## Load data if it exists
no_save = False
if use_save:
    if os.path.exists(pkl_savefile):
        with open(pkl_savefile, 'rb') as f:
            full_correlations = pickle.load(f)
        print(f"Loaded data from {pkl_savefile}")
    else:
        no_save = True
        print(f"Tried to load data from {pkl_savefile} but could not find file")

if not use_save or no_save:
    for si, n_sub in enumerate(range(1,n_subjects+1)):
        for ti in range(n_tasks):
            for ni, N in enumerate(N_vals):
                M = N*(N-1)//2
                for oi, n_obs in enumerate(n_observations):
                    for sbt_i, sbt in enumerate(sbt_vals):
                        i+=1
                        print(f"Iter {i}/{total}\tS{n_sub} T{ti} N={N} n_obs={n_obs} sbt={sbt}")

                        ## Get ground truth
                        gt_filename = get_filename_with_ext(f"gt_prior_S{n_sub}_T{ti}_N_{N}_n_obs_{n_obs}_sbt_{sbt}", folder=data_input_folder)
                        with open(gt_filename, 'rb') as gt_f:
                            gt_prior = pickle.load(gt_f)
                        ## Add bookstein
                        gt_latent_positions = gt_prior[latpos]
                        if is_bookstein:
                            bookstein_anchors = get_bookstein_anchors(D, bkst_dist)
                            gt_latent_positions = np.concatenate([bookstein_anchors,gt_latent_positions])
                        ## Get distance
                        gt_distance = det_params_func(gt_latent_positions)['d_norm']

                        ## Get embedding trace
                        embedding_input_folder = f"{base_embedding_input_folder}_N_{N}_n_obs_{n_obs}_sbt_{sbt}/{n_particles}p{n_mcmc_steps}s"
                        embedding_filename = get_filename_with_ext(f"{edge_type}_{geometry}_S{n_sub}_T{ti}_embedding", folder=embedding_input_folder)
                        with open(embedding_filename, 'rb') as e_f:
                            embedding = pickle.load(e_f)
                        ## Add bookstein
                        if is_bookstein:
                            embedding = add_bkst_to_smc_trace(embedding, bkst_dist)
                        ## Get distance
                        distance_trace = get_attribute_from_trace(embedding.particles[latpos], det_params_func, 'd_norm')  # n_particles x M

                        ## Save correlations
                        corrs = get_trace_correlation(distance_trace, gt_distance)
                        full_correlations[si, ti, ni, oi, sbt_i, :] = corrs
                        idx_reference[si, ti, ni, oi, sbt_i] = (N, n_obs, sbt)

                        ## Plot a random particle's gt_d vs d scatterplot
                        if plot_individual:
                            p_idx = np.random.randint(n_particles)
                            plt.figure()
                            plt.scatter(gt_distance, distance_trace[p_idx], c='k', s=0.5, alpha=0.8)
                            plt.xlabel('Ground truth distance (normalized)')
                            plt.ylabel('Embedding distance (normalized)')
                            plt.title(f"Ground truth vs embedding distance\nS{n_sub} T{ti} particle {p_idx}\n{N} nodes, {n_obs} observations, sigma/bound {invlogit(sbt):.3f}")
                            plt.tight_layout()
                            savefile = get_filename_with_ext(f"gt_vs_emb_dist_S{n_sub}_T{ti}_N{N}_n_obs{n_obs}_sbt{sbt}_p{p_idx}", ext='png', folder=output_folder)
                            plt.savefig(savefile)
                            plt.close()

                            if edge_type == 'con':
                                plt.figure()
                                bound_hist, bound_bins = jnp.histogram(invlogit(embedding.particles['sigma_beta_T']), bins=n_bins, density=True)
                                plt.stairs(bound_hist, bound_bins, fill=True, color='tab:gray')
                                ymin, ymax = plt.ylim()
                                plt.vlines(invlogit(sbt), ymin, ymax, 'red', 'dashed', label='True sigma/bound')
                                plt.xlabel('Sigma/bound')
                                plt.ylabel('Density')
                                plt.title(f"Histogram of embedding sigma/bound\nS{n_sub} T{ti}\n{N} nodes, {n_obs} observations")
                                plt.legend()
                                plt.tight_layout()
                                savefile = get_filename_with_ext(f"sbt_hist_S{n_sub}_T{ti}_N{N}_n_obs{n_obs}_sbt{sbt}", ext='png', folder=output_folder)
                                plt.savefig(savefile)
                                plt.close()

                                if plot_sigma:
                                    ## Load sigma data
                                    sigma_filename = get_filename_with_ext(f"sbt_prop_prior_sim_N_{N}_n_obs_{n_obs}_sbt_{sbt}_S{n_sub}_T{ti}", folder=data_input_folder)
                                    with open(sigma_filename, 'rb') as sf:
                                        sigma_proposals = pickle.load(sf)

                                    n_iter = len(sigma_proposals)

                                    ## Plot sigma convergence, should be super dope if it works.
                                    plt.figure(figsize=(20,10))
                                    for it in range(n_iter):
                                        sigma_hist, sigma_bins = jnp.histogram(invlogit(sigma_proposals[it]), bins=n_bins)
                                        # Normalize so the peak is at 1
                                        sigma_hist_nml = sigma_hist/jnp.max(sigma_hist)
                                        x_start = it*(1+x_offset)
                                        plt.stairs(sigma_hist_nml+x_start, sigma_bins, baseline=x_start, fill=True, color='tab:gray', orientation='horizontal')
                                    plt.xlabel('SMC iteration')
                                    plt.ylabel('Sigma/bound')
                                    xmin, xmax = 0, (n_iter+1)*(1+x_offset)
                                    plt.xticks(ticks=[it*(1+x_offset) for it in range(n_iter)][::x_tick_interval], labels=np.arange(n_iter)[::x_tick_interval])
                                    plt.hlines(invlogit(sbt), xmin, xmax, 'red', 'dashed', label='True sigma/bound')
                                    plt.xlim(xmin, xmax)
                                    plt.legend()
                                    plt.title(f"Convergence of sigma/bound over SMC iterations\nS{n_sub} T{ti}\n{N} nodes, {n_obs} observations, sigma/bound {invlogit(sbt):.3f}")
                                    savefile = get_filename_with_ext(f"sbt_convergence_S{n_sub}_T{ti}_N{N}_n_obs{n_obs}_sbt{sbt}", ext='png', folder=output_folder)
                                    plt.savefig(savefile)
                                    plt.close()

    # Save data for EZ re-use.
    with open(pkl_savefile, 'wb') as f:
        pickle.dump(full_correlations, f)

## Axes of the corresponding dimensions: subtracts 2 for (subject, task) axes
N_ax = 0
obs_ax = 1
sbt_ax = 2
p_ax = 3

## General plotting params
avg_params = {'capsize':3, 'c':'k'}
dist_params = {'c':'k', 'alpha':alpha}

## Noodle code to get decent autoscaling width.
boxplot_margin = 1/10 # Margin between sets of boxes as a percentage of the total width
N_dists = [N_vals[i+1] - N_vals[i] for i in range(len(N_vals) - 1)]
min_N_dist, max_N_dist = np.min(N_dists), N_vals[-1] - N_vals[0]
x_margin = max_N_dist * boxplot_margin
plot_N_vals = [N+i*x_margin for i, N in enumerate(N_vals)]
loc_increment, midpoint = min_N_dist/len(sbt_vals), min_N_dist/2
location_offset = [loc_increment*i-midpoint for i in range(len(sbt_vals))]
box_width = min_N_dist/(len(sbt_vals)+1) # Add one to leave space between boxes

sbt_colors = [plt.cm.plasma(i) for i in np.linspace(0, 1, len(sbt_vals))] # Remember this trick it's cool!
obs_colors = [plt.cm.plasma(i) for i in np.linspace(0, 1, len(n_observations))]
N_colors = [plt.cm.plasma(i) for i in np.linspace(0, 1, len(N_vals))]

for si, n_sub in enumerate(range(1,n_subjects+1)):
    # for ti in range(n_tasks): ## <--- We now average over tasks, essentially this is just 5 runs
        # FOR REFERENCE: full_correlations = np.zeros((n_subjects, n_tasks, len(N_vals), len(n_observations), len(sbt_vals), n_particles))
    sub_correlations = jnp.mean(full_correlations[si], axis=0) # Take the average over the tasks
    ## Full distribution N
    plt.figure()
    for ni, N in enumerate(N_vals):
        # full_dist_N = np.ravel(full_correlations[si,ti,ni,:,:,:])
        full_dist_N = np.ravel(sub_correlations[ni,:,:,:])
        plt.scatter(np.repeat(N, len(full_dist_N)), full_dist_N, **dist_params)
    plt.xlabel('Number of nodes')
    plt.ylabel('Ground truth vs embedding distance correlation')
    # plt.title(f"Distance correlation by number of nodes\nS{n_sub} T{ti}")
    plt.title(f"Distance correlation by number of nodes\nS{n_sub}")
    plt.tight_layout()
    # savefile = get_filename_with_ext(f"dist_corr_N_S{n_sub}_T{ti}",ext='png', folder=output_folder)
    savefile = get_filename_with_ext(f"dist_corr_N_S{n_sub}",ext='png', folder=output_folder)
    plt.savefig(savefile)
    plt.close()

    ## Full distribution observations
    plt.figure()
    for oi, n_obs in enumerate(n_observations):
        # full_dist_obs = np.ravel(full_correlations[si, ti, :, oi, :, :])
        full_dist_obs = np.ravel(sub_correlations[:, oi, :, :])
        plt.scatter(np.repeat(n_obs, len(full_dist_obs)), full_dist_obs, **dist_params)
    plt.xlabel('Number of observations')
    plt.ylabel('Ground truth vs embedding distance correlation')
    # plt.title(f"Distance correlation by number of observations\nS{n_sub} T{ti}")
    plt.title(f"Distance correlation by number of observations\nS{n_sub}")
    plt.tight_layout()
    # savefile = get_filename_with_ext(f"dist_corr_n_obs_S{n_sub}_T{ti}", ext='png', folder=output_folder)
    savefile = get_filename_with_ext(f"dist_corr_n_obs_S{n_sub}", ext='png', folder=output_folder)
    plt.savefig(savefile)
    plt.close()

    ## Full distribution sbt
    plt.figure()
    for sbt_i, sbt in enumerate(sbt_vals):
        # full_dist_sbt = np.ravel(full_correlations[si, ti, :, :, sbt_i, :])
        full_dist_sbt = np.ravel(sub_correlations[:, :, sbt_i, :])
        plt.scatter(np.repeat(invlogit(sbt), len(full_dist_sbt)), full_dist_sbt, **dist_params)
    plt.xlabel('sigma/bound')
    plt.ylabel('Ground truth vs embedding distance correlation')
    # plt.title(f"Distance correlation by sigma/bound\nS{n_sub} T{ti}")
    plt.title(f"Distance correlation by sigma/bound\nS{n_sub}")
    plt.tight_layout()
    # savefile = get_filename_with_ext(f"dist_corr_sbt_S{n_sub}_T{ti}", ext='png', folder=output_folder)
    savefile = get_filename_with_ext(f"dist_corr_sbt_S{n_sub}", ext='png', folder=output_folder)
    plt.savefig(savefile)
    plt.close()

    ## Plot average over not N
    # avg_N_corr = np.mean(full_correlations[si, ti], axis=(obs_ax, sbt_ax, p_ax))
    avg_N_corr = np.mean(sub_correlations, axis=(obs_ax, sbt_ax, p_ax))
    # std_N_corr = np.std(full_correlations[si, ti], axis=(obs_ax, sbt_ax, p_ax))
    std_N_corr = np.std(sub_correlations, axis=(obs_ax, sbt_ax, p_ax))
    plt.figure()
    plt.errorbar(N_vals, avg_N_corr, yerr=std_N_corr, **avg_params)
    plt.xlabel('Number of nodes')
    plt.ylabel('Ground truth vs embedding distance correlation')
    # plt.title(f"Distance correlation by number of nodes\nS{n_sub} T{ti}")
    plt.title(f"Distance correlation by number of nodes\nS{n_sub}")
    plt.tight_layout()
    # savefile = get_filename_with_ext(f"avg_dist_corr_N_S{n_sub}_T{ti}",ext='png', folder=output_folder)
    savefile = get_filename_with_ext(f"avg_dist_corr_N_S{n_sub}",ext='png', folder=output_folder)
    plt.savefig(savefile)
    plt.close()

    # Plot average over not obs
    # avg_obs_corr = np.mean(full_correlations[si, ti], axis=(N_ax, sbt_ax, p_ax))
    avg_obs_corr = np.mean(sub_correlations, axis=(N_ax, sbt_ax, p_ax))
    # std_obs_corr = np.std(full_correlations[si, ti], axis=(N_ax, sbt_ax, p_ax))
    std_obs_corr = np.std(sub_correlations, axis=(N_ax, sbt_ax, p_ax))
    plt.figure()
    plt.errorbar(n_observations, avg_obs_corr, yerr=std_obs_corr, **avg_params)
    plt.xlabel('Number of observations')
    plt.ylabel('Ground truth vs embedding distance correlation')
    # plt.title(f"Distance correlation by number of observations\nS{n_sub} T{ti}")
    plt.title(f"Distance correlation by number of observations\nS{n_sub}")
    plt.tight_layout()
    # savefile = get_filename_with_ext(f"avg_dist_corr_n_obs_S{n_sub}_T{ti}", ext='png', folder=output_folder)
    savefile = get_filename_with_ext(f"avg_dist_corr_n_obs_S{n_sub}", ext='png', folder=output_folder)
    plt.savefig(savefile)
    plt.close()

    # Plot average over not sbts
    # avg_sbt_corr = np.mean(full_correlations[si, ti], axis=(N_ax, obs_ax, p_ax))
    avg_sbt_corr = np.mean(sub_correlations, axis=(N_ax, obs_ax, p_ax))
    # std_sbt_corr = np.std(full_correlations[si, ti], axis=(N_ax, obs_ax, p_ax))
    std_sbt_corr = np.std(sub_correlations, axis=(N_ax, obs_ax, p_ax))
    plt.figure()
    plt.errorbar([invlogit(sbt) for sbt in sbt_vals], avg_sbt_corr, yerr=std_sbt_corr, **avg_params)
    plt.xlabel('sigma/bound')
    plt.ylabel('Ground truth vs embedding distance correlation')
    # plt.title(f"Distance correlation by sigma/bound\nS{n_sub} T{ti}")
    plt.title(f"Distance correlation by sigma/bound\nS{n_sub}")
    plt.tight_layout()
    # savefile = get_filename_with_ext(f"avg_dist_corr_sbt_S{n_sub}_T{ti}", ext='png', folder=output_folder)
    savefile = get_filename_with_ext(f"avg_dist_corr_sbt_S{n_sub}", ext='png', folder=output_folder)
    plt.savefig(savefile)
    plt.close()

    ## Plot average over only particles, a seperate line for each n_obs, x=N, new plot per sigma/bound
    for sbt_i, sbt in enumerate(sbt_vals):
        plt.figure()
        for oi, n_obs in enumerate(n_observations):
            # avg_corr = np.mean(full_correlations[si, ti, :, oi, sbt_i, :], axis=-1)
            avg_corr = np.mean(sub_correlations[:, oi, sbt_i, :], axis=-1)
            # std_corr = np.std(full_correlations[si, ti, :, oi, sbt_i, :], axis=-1)
            std_corr = np.std(sub_correlations[:, oi, sbt_i, :], axis=-1)
            plt.errorbar(N_vals, avg_corr, yerr=std_corr, capsize=avg_params['capsize'], c=obs_colors[oi], label=f"n_obs={n_obs}")
        plt.title(f"Average embedding distance vs ground truth distance correlations \nwith standard deviation\n Sigma/bound={invlogit(sbt):.2f}, S{n_sub}")
        plt.legend()
        plt.xlabel('Number of nodes')
        plt.ylabel('Ground truth vs embedding distance correlation')
        plt.tight_layout()
        # savefile = get_filename_with_ext(f"avg_dist_corr_N_by_n_obs_S{n_sub}_T{ti}_sbt{sbt}", ext='png', folder=output_folder)
        savefile = get_filename_with_ext(f"avg_dist_corr_N_by_n_obs_S{n_sub}_sbt{sbt}", ext='png', folder=output_folder)
        plt.savefig(savefile)
        plt.close()

    ## Plot average over only particles, a seperate line for each N, x=n_obs, new plot per sigma/bound
    for sbt_i, sbt in enumerate(sbt_vals):
        plt.figure()
        for ni, N in enumerate(N_vals):
            # avg_corr = np.mean(full_correlations[si, ti, ni, :, sbt_i, :], axis=-1)
            avg_corr = np.mean(sub_correlations[ni, :, sbt_i, :], axis=-1)
            # std_corr = np.std(full_correlations[si, ti, ni, :, sbt_i, :], axis=-1)
            std_corr = np.std(sub_correlations[ni, :, sbt_i, :], axis=-1)
            plt.errorbar(n_observations, avg_corr, yerr=std_corr, capsize=avg_params['capsize'], c=N_colors[ni], label=f"N={N}")
        plt.title(f"Average embedding distance vs ground truth distance correlations \nwith standard deviation\nSigma/bound={invlogit(sbt):.2f}, S{n_sub}")
        plt.legend()
        plt.xlabel('Number of observations')
        plt.ylabel('Ground truth vs embedding distance correlation')
        plt.tight_layout()
        # savefile = get_filename_with_ext(f"avg_dist_corr_n_obs_by_N_S{n_sub}_T{ti}_sbt{sbt}", ext='png', folder=output_folder)
        savefile = get_filename_with_ext(f"avg_dist_corr_n_obs_by_N_S{n_sub}_sbt{sbt}", ext='png', folder=output_folder)
        plt.savefig(savefile)
        plt.close()

    ## Plot average over only particles, a seperate line for each sigma/bound, x=N, new plot per n_obs
    for oi, n_obs in enumerate(n_observations):
        plt.figure()
        for sbt_i, sbt in enumerate(sbt_vals):
            # avg_corr = np.mean(full_correlations[si, ti, :, oi, sbt_i, :], axis=-1)
            avg_corr = np.mean(sub_correlations[:, oi, sbt_i, :], axis=-1)
            # std_corr = np.std(full_correlations[si, ti, :, oi, sbt_i, :], axis=-1)
            std_corr = np.std(sub_correlations[:, oi, sbt_i, :], axis=-1)
            plt.errorbar(N_vals, avg_corr, yerr=std_corr, capsize=avg_params['capsize'], c=sbt_colors[sbt_i], label=f"Sigma/bound={invlogit(sbt):.2f}")
        plt.title(f"Average embedding distance vs ground truth distance correlations \nwith standard deviation\nn_obs={n_obs}")
        plt.legend()
        plt.xlabel('Number of nodes')
        plt.ylabel('Ground truth vs embedding distance correlation')
        plt.tight_layout()
        # savefile = get_filename_with_ext(f"avg_dist_corr_N_by_sbt_S{n_sub}_T{ti}_n_obs{n_obs}", ext='png', folder=output_folder)
        savefile = get_filename_with_ext(f"avg_dist_corr_N_by_sbt_S{n_sub}_n_obs{n_obs}", ext='png', folder=output_folder)
        plt.savefig(savefile)
        plt.close()

    ## N_obs has clear effect: for each n_obs, we plot per N all boxplots for each sigma/bound
    for oi, n_obs in enumerate(n_observations):
        plt.figure(figsize=(20,10))
        for sbt_i, sbt in enumerate(sbt_vals):
            plt.plot()
            for ni, N in enumerate(plot_N_vals):
                # corrs = full_correlations[si, ti, ni, oi, sbt_i, :]
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
                    bp['boxes'][0].set_label(f"Sigma/bound={invlogit(sbt):.2f}")
        plt.xticks(ticks=plot_N_vals, labels=N_vals) ## TODO: Only ever nth number! 
        plt.xlim(plot_N_vals[0]-min_N_dist, plot_N_vals[-1]+min_N_dist)
        # plt.ylim(0, 1)
        plt.xlabel('Number of nodes')
        plt.ylabel('Ground truth vs embedding distance correlation')
        plt.legend()
        plt.title(f"Embedding distance vs ground truth distance correlations\nNumber of observations={n_obs}")
        plt.tight_layout()
        # savefile = get_filename_with_ext(f"box_dist_corr_S{n_sub}_T{ti}_n_obs{n_obs}", ext='png', folder=output_folder)
        savefile = get_filename_with_ext(f"box_dist_corr_S{n_sub}_n_obs{n_obs}", ext='png', folder=output_folder)
        plt.savefig(savefile)
        plt.close()