import numpy as np
import jax
import jax.numpy as jnp
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
import os

from helper_functions import get_filename_with_ext, load_observations, get_cmd_params, set_GPU, get_safe_folder, get_attribute_from_trace, get_trace_correlation, invlogit

from binary_euclidean_LSM import get_det_params as bin_euc_det_params
from binary_hyperbolic_LSM import get_det_params as bin_hyp_det_params
from continuous_euclidean_LSM import get_det_params as con_euc_det_params
from continuous_hyperbolic_LSM import get_det_params as con_hyp_det_params

from plotting_functions import plot_sigma_convergence

arguments = [('-df', 'data_folder', str, 'Data'), # data folder
             ('-conbdf', 'con_base_data_filename', str, 'processed_data'),  # the most basic version of the filename of the continuous saved data
             ('-binbdf', 'bin_base_data_filename', str, 'binary_data_max_0.05unconnected'), # the most basic version of the filename of the binary saved data
             ('-overwritedf', 'overwrite_data_filename', str, None),  # if used, it overwrites the default filename
             ('-if', 'base_input_folder', str, 'Embeddings'), # base input folder of the embeddings
             ('-np', 'n_particles', int, 1000), # number of particles used in the embedding
             ('-nm', 'n_mcmc_steps', int, 100), # number of mcmc steps used in the embedding
             ('-tf', 'task_filename', str, 'task_list'), # filename of the list of task names
             ('-of', 'output_folder', str, 'Figures/sanity_checks'), # folder where to dump figures
             ('-gtf', 'gt_folder', str, None), # folder of the ground truths to plot true distances
             ('-nps', 'number_plot_samples', int, 1), # number of particles to sample which will be plotted
             ('-N', 'N', int, 164), # number of nodes
             ('-s1', 'subject1', int, 1), # first subject to plot
             ('-sn', 'subjectn', int, 25), # last subject to plot
             ('-et', 'edge_type', str, 'con'), # edge type ('con' or 'bin')
             ('-geo', 'geometry', str, 'hyp'), # LS geometry ('hyp' or 'euc')
             ('-nbins', 'n_bins', int, 100), # number of bins for the distance correlation histogram
             ('-ssf', 'save_sigma_filename', str, 'sbt'), # filename WITHOUT the edge type and geometry
             ('-setsig', 'set_sigma', float, None), # value that sigma is set to (or None if learned)
             ('-palpha', 'plot_alpha', float, 1), # alpha of the scatter plots
             ('--partial', 'partial', bool), # whether to use partial correlations
             ('--bpf', 'bpf', bool), # whether to use band-pass filtered correlations
             ('--reg', 'add_regression', bool), # whether to add a regression line in the distance v correlation plot
             ('-gpu', 'gpu', str, ''),  # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]

global_params = get_cmd_params(arguments)
set_GPU(global_params['gpu'])
data_folder = global_params['data_folder']
edge_type = global_params['edge_type']
geometry = global_params['geometry']

base_data_filename = global_params['bin_base_data_filename'] if edge_type == 'bin' else global_params['con_base_data_filename']
overwrite_data_filename = global_params['overwrite_data_filename']
n_particles = global_params['n_particles']
N = global_params['N']
M = N*(N-1)//2
input_folder = f"{global_params['base_input_folder']}/{n_particles}p{global_params['n_mcmc_steps']}s"
task_filename = get_filename_with_ext(global_params['task_filename'], ext='txt', folder=data_folder)
output_folder = get_safe_folder(f"{global_params['output_folder']}/{edge_type}_{geometry}")
gt_folder = global_params['gt_folder']
ground_truth = gt_folder is not None
number_plot_samples = global_params['number_plot_samples']
subject1 = global_params['subject1']
subjectn = global_params['subjectn']

save_sigma_filename = global_params['save_sigma_filename']
set_sigma = global_params['set_sigma']
set_sigma_txt = f"_sigma_set_{set_sigma}" if set_sigma is not None else ''
n_bins = global_params['n_bins']
plot_alpha = global_params['plot_alpha']
partial = global_params['partial']
add_regression = global_params['add_regression']
bpf = global_params['bpf']

det_params_dict = {'bin_euc':bin_euc_det_params,
                   'bin_hyp':bin_hyp_det_params,
                   'con_euc':con_euc_det_params,
                   'con_hyp':con_hyp_det_params}
det_params_func = det_params_dict[f"{edge_type}_{geometry}"]

latpos = '_z' if geometry == 'hyp' else 'z'

if not overwrite_data_filename:
    data_filename = get_filename_with_ext(base_data_filename, partial, bpf, folder=data_folder)
else:
    data_filename = overwrite_data_filename
obs, tasks, encs = load_observations(data_filename, task_filename, subject1, subjectn, M)

for si, n_sub in enumerate(range(subject1, subjectn + 1)):
    for ti, task in enumerate(tasks):
        if edge_type == 'con' and set_sigma is None:
            ## Load and plot sigma convergence
            sigma_filename = get_filename_with_ext(f"con_{geometry}_{save_sigma_filename}_S{n_sub}_{task}_{base_data_filename}", partial=partial, folder=data_folder)
            with open(sigma_filename, 'rb') as f:
                sigma_chain = pickle.load(f)

            plt.figure(figsize=(20,10))
            ax = plt.gca()
            ax = plot_sigma_convergence(sigma_chain, ax=ax, legend=False)
            plt.xlabel('SMC iteration')
            plt.ylabel('Sigma/bound')
            plt.title(f"Sigma convergence for S{n_sub} {task}")
            plt.tight_layout()
            figure_filename = get_filename_with_ext(f"{edge_type}_{geometry}_sigma_convergence_S{n_sub}_{task}", ext='png', partial=partial, folder=output_folder)
            plt.savefig(figure_filename)
            plt.close()

        # Load embedding
        embedding_filename = get_filename_with_ext(f"{edge_type}_{geometry}_S{n_sub}_{task}_embedding_{base_data_filename}{set_sigma_txt}", partial=partial, folder=input_folder)
        with open(embedding_filename, 'rb') as f:
            embedding = pickle.load(f)

        # Get embedding distances
        distance_trace = get_attribute_from_trace(embedding.particles[latpos], det_params_func, 'd') # n_particles x M
        if edge_type == 'con' and set_sigma is None:
            sigma_div_bound_trace = invlogit(embedding.particles['sigma_beta_T']) # n_particles x 1
            bound_trace = get_attribute_from_trace(embedding.particles[latpos], det_params_func, 'bound') # n_particles x M
            sigma_trace = sigma_div_bound_trace * bound_trace # n_particles x M

        if ground_truth:
            gt_filename = get_filename_with_ext(f"gt_prior_S{n_sub}_T{ti}_N_{N}_n_obs_{n_obs}_sbt_{sbt}", folder=gt_folder) # <-- srry this is uggley
            with open(gt_filename, 'rb') as f:
                gt_embedding = pickle.load(f)
            gt_distances = det_params_func(gt_embedding[latpos])['d']
            if edge_type == 'con':
                gt_bound = det_params_func(gt_embedding[latpos])['bound']
                gt_sigma_div_bound = invlogit(gt_embedding['sigma_beta_T'])
                gt_sigma = gt_sigma_div_bound*gt_bound

            distance_correlation = np.array(get_trace_correlation(distance_trace, gt_distances))

            ### PLOT DISTANCE VS GROUND TRUTH DISTANCE
            output_file = get_filename_with_ext(f"dist_corr_hist_S{n_sub}_{task}", partial, bpf, ext='png', folder=output_folder)
            title = f"Correlation between embedding distances and ground truth distance\nS{n_sub} {task}"
            plt.figure()
            d_corr_hist, d_corr_bins = jnp.histogram(distance_correlation, bins=n_bins, density=True)
            plt.stairs(d_corr_hist, d_corr_bins, fill=True, color='tab:gray')
            plt.xlabel('correlation')
            plt.ylabel('count')
            plt.title(title)
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()

        if edge_type == 'con' and set_sigma is None:
            ### PLOT SIGMA DISTRIBUTIONS
            output_file = get_filename_with_ext(f"sigma_over_bound_S{n_sub}_{task}_{base_data_filename}", partial, bpf, ext='png', folder=output_folder)
            title = f"Sigma over bound distribution\nS{n_sub} {task}"
            plt.figure()
            sigma_div_bound_hist, sigma_div_bound_bins = jnp.histogram(sigma_div_bound_trace, bins=n_bins, density=True)
            plt.stairs(sigma_div_bound_hist, sigma_div_bound_bins, fill=True, color='tab:gray', label='Embedding')
            if ground_truth:
                ymin, ymax = plt.ylim()
                plt.vlines(gt_sigma_div_bound, ymin, ymax, colors='r', label='Ground truth')
                plt.legend()
            plt.xlabel('sigma / bound')
            plt.ylabel('count')
            plt.title(title)
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()

            alpha = 0.5 if ground_truth else 1
            xmargin = 0.01
            ymargin = 5

            ### PLOT BOUND DISTRIBUTIONS
            output_file = get_filename_with_ext(f"bound_S{n_sub}_{task}_{base_data_filename}", partial, bpf, ext='png', folder=output_folder)
            title = f"Bound distribution\nS{n_sub} {task}"
            plt.figure()
            plt.autoscale(True)
            bound_hist, bound_bins = jnp.histogram(bound_trace, bins=n_bins, density=True)
            plt.stairs(bound_hist, bound_bins, fill=True, color='tab:gray', alpha=alpha, label='Embedding')
            top = jnp.max(bound_hist)
            left = jnp.min(bound_bins)
            right = jnp.max(bound_bins)
            if ground_truth:
                gt_bound_hist, gt_bound_bins = jnp.histogram(gt_bound, bins=n_bins, density=True)
                plt.stairs(gt_bound_hist, gt_bound_bins, fill=True, color='tab:red', alpha=alpha, label='Ground truth')
                top = max(jnp.max(gt_bound_hist), top)
                left = min(jnp.min(gt_bound_bins), left)
                right = max(jnp.max(gt_bound_bins), right)
            plt.ylim(0, top+ymargin)
            plt.xlim(left-xmargin, right+xmargin)
            plt.xlabel('bound')
            plt.ylabel('count')
            plt.title(title)
            if ground_truth:
                plt.legend()
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()

            output_file = get_filename_with_ext(f"sigma_S{n_sub}_{task}_{base_data_filename}", partial, bpf, ext='png',folder=output_folder)
            title = f"Sigma distribution\nS{n_sub} {task}"
            plt.figure()
            sigma_hist, sigma_bins = jnp.histogram(sigma_trace, bins=n_bins, density=True)
            plt.stairs(sigma_hist, sigma_bins, fill=True, color='tab:gray', alpha=alpha, label='Embedding')
            top = jnp.max(sigma_hist)
            left = jnp.min(sigma_bins)
            right = jnp.max(sigma_bins)
            if ground_truth:
                gt_sigma_hist, gt_sigma_bins = jnp.histogram(gt_sigma, bins=n_bins, density=True)
                plt.stairs(gt_sigma_hist, gt_sigma_bins, fill=True, color='tab:red', alpha=alpha, label='Ground truth')
                top =  max(jnp.max(gt_sigma_hist), top)
                left = min(jnp.min(gt_sigma_bins), left)
                right = max(jnp.max(gt_sigma_bins), right)
                plt.legend()
            plt.ylim(0, top+ymargin)
            plt.xlim(left - xmargin, right + xmargin)
            plt.xlabel('sigma')
            plt.ylabel('count')
            plt.title(title)
            if ground_truth:
                plt.legend()
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()

        particle_plot_sample_idc = sorted(np.random.choice(np.arange(n_particles), number_plot_samples, replace=False))
        for pi, ppsi in enumerate(particle_plot_sample_idc):
            distances = distance_trace[ppsi, :]

            if ground_truth:
                output_file = get_filename_with_ext(f"dist_vs_gt_dist_S{n_sub}_{task}_sample{pi}", partial, bpf, ext='png', folder=output_folder)
                title = f"Embedding distance vs Ground truth distance\nS{n_sub} {task}\nParticle {ppsi}"
                plt.figure()
                plt.scatter(distances, gt_distances, s=.5, c='k', alpha=plot_alpha)
                plt.xlabel('Embedding distances')
                plt.ylabel('Ground truth distances')
                if add_regression:
                    reg = LinearRegression().fit(distances.reshape((M, 1)), gt_distances)
                    xmin, xmax = plt.xlim()
                    x = np.linspace(xmin, xmax, M)
                    y = reg.intercept_ + reg.coef_ * x
                    plt.plot(x, y, color='r')
                plt.title(title)
                plt.savefig(output_file, bbox_inches='tight')
                plt.close()

            for ei, enc in enumerate(encs):
                correlations = obs[si, ti, ei, :] # M observed correlations

                output_file = get_filename_with_ext(f"dist_vs_corr_S{n_sub}_{task}_{enc}_sample{pi}_{base_data_filename}",partial,bpf,ext='png',folder=output_folder)

                partial_txt = 'partial ' if partial else ''
                title = f"Distance vs {partial_txt}correlation\nS{n_sub} {task} {enc}\nParticle {ppsi}"
                plt.figure()
                plt.scatter(distances, correlations, s=0.5, c='k', alpha=plot_alpha)
                plt.xlabel('Distance')
                plt.ylabel('Correlation')
                if add_regression:
                    reg = LinearRegression().fit(distances.reshape((M,1)), correlations)
                    xmin, xmax = plt.xlim()
                    x = np.linspace(xmin, xmax, M)
                    y = reg.intercept_ + reg.coef_*x
                    plt.plot(x, y, color='r')
                plt.title(title)
                plt.tight_layout()
                plt.savefig(output_file)
                plt.close()