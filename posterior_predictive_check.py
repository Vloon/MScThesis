"""
Calling this file loads the learned embeddings, and generates new data from those embeddings.
If not calling this file with '--onlypred', it also makes a number of plots to provide insight into the embedding's ability to capture the data-generation process.
"""

## Basics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, NullLocator, AutoLocator
import pickle
import jax
import jax.numpy as jnp
import os
import time

## Typing
from jax._src.typing import ArrayLike
from jax._src.prng import PRNGKeyArray
from blackjax.smc.tempered import TemperedSMCState
from typing import Callable, Tuple

from helper_functions import set_GPU, get_cmd_params, get_filename_with_ext, get_safe_folder, load_observations, get_attribute_from_trace, triu2mat

## The function to get the log-likelihood from the distance is the same for the same edge-type, regardless of the geometry. So bin_euc_loglikelihood_from_distance = bin_hyp_loglikelihood_from_distance
from binary_euclidean_LSM import get_det_params as bin_euc_det_params, sample_observation as bin_euc_sample_observation, log_likelihood_from_dist as bin_loglikelihood
from binary_hyperbolic_LSM import get_det_params as bin_hyp_det_params, sample_observation as bin_hyp_sample_observation
from continuous_euclidean_LSM import get_det_params as con_euc_det_params, sample_observation as con_euc_sample_observation, log_likelihood_from_dist as con_loglikelihood
from continuous_hyperbolic_LSM import get_det_params as con_hyp_det_params, sample_observation as con_hyp_sample_observation

### Create cmd argument list (arg_name, var_name, type, default[OPT], nargs[OPT]).
###  - arg_name is the name of the argument in the command line.
###  - var_name is the name of the variable in the returned dictionary (which we re-use as variable name here).
###  - type is the data-type of the variable.
###  - default is the default value it takes if nothing is passed to the command line. This argument is only optional if type is bool, where the default is always False.
###  - nargs is the number of arguments, where '?' (default) is 1 argument, '+' concatenates all arguments to 1 list. This argument is optional.
arguments = [('-overwritedf', 'overwrite_data_filename', str, None),  # if used, it overwrites the default filename
             ('-datfol', 'data_folder', str, 'Data'),  # folder where the data is stored
             ('-conbdf', 'con_base_data_filename', str, 'processed_data_downsampled_evenly_spaced'), # the most basic version of the filename of the continuous saved data
             ('-binbdf', 'bin_base_data_filename', str, 'binary_data_downsampled_evenly_spaced_max_0.05unconnected'), # the most basic version of the filename of the binary saved data
             ('-ef', 'embedding_folder', str, 'Embeddings'), # base input folder of the embeddings
             ('-ff', 'figure_folder', str, 'Figures/sanity_checks/posterior_predictive_checks'), # figure output folder
             ('-tf', 'task_filename', str, 'task_list'), # filename of the list of task names
             ('-lab', 'label_location', str, 'Figures/lobelabels.npz'),  # file location of the labels
             ('-np', 'n_particles', int, 2000), # number of particles used in the embedding
             ('-nm', 'n_mcmc_steps', int, 500), # number of mcmc steps used in the embedding
             ('-N', 'N', int, 164), # number of nodes
             ('-s1', 'subject1', int, 1), # first subject to plot
             ('-sn', 'subjectn', int, 25), # last subject to plot
             ('-et', 'edge_type', str, 'con'),  # edge type ('con' or 'bin')
             ('-geo', 'geometry', str, 'hyp'),  # LS geometry ('hyp' or 'euc')
             ('-setsig', 'set_sigma', float, None), # value that sigma is set to (or None if learned)
             ('-nshuffle', 'n_shuffle', int, 100), # number of shuffles per particles to check for chance-level embedding
             ('-ebins', 'n_embedding_bins', int, 50), # number of bins in the embedding histogram
             ('-sbins', 'n_shuffle_bins', int, 1000), # number of bins in the shuddled histogram
             ('-palpha', 'plot_alpha', float, 0.7), # alpha for the histograms
             ('-cmap', 'cmap', str, 'bwr'), # colormap for the imshows of the real vs predicted observations
             ('-lfs', 'label_fontsize', float, 20), # fontsize of labels (and legend)
             ('-tfs', 'tick_fontsize', float, 16), # fontsize of the tick labels
             ('-wsz', 'wrapsize', float, 20), # wrapped text width
             ('--partial', 'partial', bool), # whether to use partial correlations
             ('--bpf', 'bpf', bool), # whether to use band-pass filtered rs-fMRI data
             ('--perm', 'permutation_test', bool), # whether to perform a permutation test
             ('--saveerrs', 'save_errors', bool), # whether to save the predictions
             ('--onlypred', 'only_predict', bool), # whether to only create the prediction error file without doing any plotting
             ('-errname', 'error_savename', str, 'prediction_errors'), # name with which to save the prediction error file
             ('-errfld', 'error_folder', str, 'Statistics'), # folder where to save the prediction error file
             ('-seed', 'seed', int, 1234), # starting random key
             ('-gpu', 'gpu', str, ''),  # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]

## Get arguments from command line.
global_params = get_cmd_params(arguments)
set_GPU(global_params['gpu'])
label_fontsize = global_params['label_fontsize']
tick_fontsize = global_params['tick_fontsize']
wrapsize = global_params['wrapsize']
overwrite_data_filename = global_params['overwrite_data_filename']
n_particles = global_params['n_particles']
embedding_folder = f"{global_params['embedding_folder']}/{n_particles}p{global_params['n_mcmc_steps']}s"
data_folder = global_params['data_folder']
edge_type = global_params['edge_type']
geometry = global_params['geometry']
set_sigma = global_params['set_sigma']
sigma_txt = f"_sigma_set_{set_sigma}" if set_sigma is not None else ''
base_data_filename = global_params['bin_base_data_filename'] if edge_type == 'bin' else global_params['con_base_data_filename']
figure_folder = get_safe_folder(f"{global_params['figure_folder']}/{edge_type}_{geometry}")
task_filename = global_params['task_filename']
label_location = global_params['label_location']
N = global_params['N']
M = N*(N-1)//2
subject1 = global_params['subject1']
subjectn = global_params['subjectn']
n_shuffle = global_params['n_shuffle']
n_embedding_bins = global_params['n_embedding_bins']
n_shuffle_bins = global_params['n_shuffle_bins']
plot_alpha = global_params['plot_alpha']
cmap = global_params['cmap']
partial = global_params['partial']
bpf = global_params['bpf']
permutation_test = global_params['permutation_test']
save_errors = global_params['save_errors']
error_folder = get_safe_folder(global_params['error_folder'])
error_savename = get_filename_with_ext(f"{edge_type}_{geometry}_{global_params['error_savename']}", partial, bpf, folder=error_folder)
avg_error_savename = get_filename_with_ext(f"{edge_type}_{geometry}_avg_{global_params['error_savename']}", partial, bpf, folder=error_folder)
only_predict = global_params['only_predict']
## Use JAX functions only after setting the GPU, otherwise it will use all GPUs by default.
key = jax.random.PRNGKey(global_params['seed'])    

## Load labels
label_data = np.load(label_location)
plt_labels = label_data[label_data.files[0]]
if len(plt_labels) != N:
    plt_labels = None

## Define a number of variables based on geometry or edge type
det_params_dict = {'bin_euc':bin_euc_det_params,
                   'bin_hyp':bin_hyp_det_params,
                   'con_euc':con_euc_det_params,
                   'con_hyp':con_hyp_det_params}
det_params_func = det_params_dict[f"{edge_type}_{geometry}"]
sample_obs_dict = {'bin_euc':bin_euc_sample_observation,
                   'bin_hyp':bin_hyp_sample_observation,
                   'con_euc':con_euc_sample_observation,
                   'con_hyp':con_hyp_sample_observation}
sample_obs_func = sample_obs_dict[f"{edge_type}_{geometry}"]

latpos = '_z' if geometry == 'hyp' else 'z'

def get_con_log_likelihood(i:int, log_likelihoods:ArrayLike, distances:ArrayLike, sigma_T_trace:ArrayLike, obs:ArrayLike) -> ArrayLike:
    """
    Returns the log-likelihood of the model given continuous data based on the distance.
    PARAMS:
    i : particle index
    log_likelihoods : (n_particles,) to be returned array of log-likelihoods
    distances : (n_particles, M) to be used distance array
    sigma_T_trace : (n_particles,) trace of the transformed std
    obs : (n_obs, M) observation array
    """
    next_ll = con_loglikelihood(distances[i], sigma_T_trace[i], obs=obs)
    log_likelihoods = log_likelihoods.at[i].set(next_ll)
    return log_likelihoods

def get_bin_log_likelihood(i:int, log_likelihoods:ArrayLike, distances:ArrayLike, obs:ArrayLike, **kwargs) -> ArrayLike:
    """
    Returns the log-likelihood of the model given binary data based on the distance.
    PARAMS:
    i : particle index
    log_likelihoods : (n_particles,) to be returned array of log-likelihoods
    distances : (n_particles, M) to be used distance array
    obs : (n_obs, M) observation array
    **kwargs will catch the sigma_beta_T (always None here) which is passed to generalize the code.
    """
    next_ll = bin_loglikelihood(distances[i], obs=obs)
    log_likelihoods = log_likelihoods.at[i].set(next_ll)
    return log_likelihoods

def get_obs_samples(i:int, state:Tuple[PRNGKeyArray,ArrayLike], parameters:list, embedding:TemperedSMCState, set_sigma:float=set_sigma, sample_obs_func:Callable=sample_obs_func):
    """
    Returns a sampled observation, based on the given posterior .
    PARAMS:
    i : particle index
    state: tuple containing key, predictions
        key : JAX random key
        predictions : (n_particles, M) prediced observations (to be filled in)
    parameters : list of parameters that need to be taken from the embedding (e.g. ['_z', 'sigma_beta_T])
    embeddding : posterior, containing latent positions and possibly sigma_beta_T
    set_sigma : value to which sigma was set (or None)
    sample_obs_func : function to sample the observations
    """
    key, predictions = state
    posterior = {param:embedding.particles[param][i] for param in parameters}
    if set_sigma is not None: 
        posterior['sigma_beta_T'] = set_sigma
    key, pred = sample_obs_func(key, posterior)
    predictions = predictions.at[i].set(pred[0]) # Take first element because it's a (1,M) array which doesn't fit in an M sized space
    return key, predictions

def get_predictions(key:PRNGKeyArray, n_sub, task, edge_type:str=edge_type, geometry:str=geometry, base_data_filename:str=base_data_filename,
                    sigma_txt:str=sigma_txt, partial:bool=partial, bpf:bool=bpf, embedding_folder:str=embedding_folder,
                    set_sigma:bool=set_sigma, n_particles:int=n_particles, M:int=M):
    """
    Loads the embedding and makes the predictions for the given subject, task
    PARAMS:
    key : jax random key
    n_sub : subject number
    task : task name
    edge_type : edge type
    geometry : latent space geometry
    base_data_filename : the most basic version of the filename of the embedding
    sigma_txt : text to be added to the embedding filename for set sigma runs
    partial : whether to use partial correlations
    bpf : whether to use band-pass filtered data for the two resting states
    embedding_folder : folder containing the embeddings
    set_sigma : whether sigma is set
    n_particles : number of particles
    M : number of edges
    """
    ## Load the embedding
    embedding_filename = get_filename_with_ext(f"{edge_type}_{geometry}_S{n_sub}_{task}_embedding_{base_data_filename}{sigma_txt}", partial, bpf, folder=embedding_folder)
    with open(embedding_filename, 'rb') as f:
        embedding = pickle.load(f)
    ## Create a prediction from that embedding
    parameters = [latpos, 'sigma_beta_T'] if edge_type == 'con' and set_sigma is None else [latpos]
    get_pred_func = lambda i, state: get_obs_samples(i, state, parameters, embedding)
    key, predictions = jax.lax.fori_loop(0, n_particles, get_pred_func, (key, jnp.zeros((n_particles, M))))
    return key, embedding, predictions

ll_f = get_con_log_likelihood if edge_type == 'con' else get_bin_log_likelihood

## Get data that was trained on
if not overwrite_data_filename:
    data_filename = get_filename_with_ext(base_data_filename, partial, bpf, folder=data_folder)
else:
    data_filename = overwrite_data_filename
task_filename = get_filename_with_ext(task_filename, ext='txt', folder=data_folder)
obs, tasks, encs = load_observations(data_filename, task_filename, subject1, subjectn, M)

n_subjects, n_tasks, n_obs = subjectn+1-subject1, len(tasks), len(encs)
prediction_errors = np.zeros((n_subjects, n_tasks, n_particles, n_obs))

if only_predict:
    ## Make predictions without making figures, use when needing many predictions
    for si, n_sub in enumerate(range(subject1, subjectn + 1)):
        for ti, task in enumerate(tasks):
            key, embedding, predictions = get_predictions(key, n_sub, task)
            get_sse = lambda i, state: (state[0].at[i].set(np.sum((obs[si, ti] - state[1][i]) ** 2)), state[1])
            for oi in range(n_obs):
                sse, _ = jax.lax.fori_loop(0, n_particles, get_sse, (np.zeros(n_particles), predictions))
                prediction_errors[si, ti, :, oi] = sse
else:
    ## Make figures as well as predictions
    for si, n_sub in enumerate(range(subject1, subjectn+1)):
        for ti, task in enumerate(tasks):
            print(f"Running S{n_sub}, {task}")

            ## Get embedding, predictions from that embedding, and embedding distances
            key, embedding, predictions = get_predictions(key, n_sub, task)
            distance_trace = get_attribute_from_trace(embedding.particles[latpos], det_params_func, 'd')

            ## Average over the particles
            d_avg = np.mean(distance_trace, axis=0)
            d_avg = triu2mat(d_avg)

            ## Get prediction errors
            ## State contains the sse values which are being filled in, as well as the predictions, which we iterate over.
            get_sse = lambda i, state: (state[0].at[i].set(np.sum((obs[si, ti] - state[1][i]) ** 2)), state[1])
            for oi in range(n_obs):
                sse, _ = jax.lax.fori_loop(0, n_particles, get_sse, (np.zeros(n_particles), predictions))
                prediction_errors[si, ti, :, oi] = sse

            ## Sort distance matrix by lobe
            lobes = [l[1].split(';')[0] for l in plt_labels]
            idc = np.argsort(lobes)
            d_sorted = np.zeros((N, N))
            for row in range(N):
                d_sorted[row] = d_avg[idc[row]][idc]

            ordered = np.sort(lobes)
            uordered = np.unique(ordered)
            first_idc = [list(ordered).index(ul) for ul in uordered]

            ## Create figure of the sorted distances
            plt.figure()
            plt.imshow(d_sorted, cmap=cmap)
            plt.yticks(ticks=first_idc, labels=uordered, fontsize=tick_fontsize)
            plt.xticks(ticks=first_idc, labels=uordered, rotation=75, fontsize=tick_fontsize, ha='right')
            cbar = plt.colorbar()
            ax = plt.gca()
            ax.xaxis.tick_top()
            savename = get_filename_with_ext(f"dist_matrix_by_lobe_S{n_sub}_{task}_{base_data_filename}", ext='png', partial=partial, bpf=bpf, folder=figure_folder)
            plt.savefig(savename, bbox_inches='tight')
            plt.close()

            ## Perform permutation test
            if permutation_test:
                sigma_T_trace = embedding.particles['sigma_beta_T'] if edge_type == 'con' else None
                embedding_log_likelihoods = jnp.zeros(n_particles)
                get_ll = lambda i, ll: ll_f(i,ll, distances=distance_trace, obs=obs[si, ti], sigma_T_trace=sigma_T_trace) # Curry the observations, distance and sigma
                embedding_log_likelihoods = jax.lax.fori_loop(0, n_particles, get_ll, embedding_log_likelihoods)
                emb_hist, emb_bins = jnp.histogram(embedding_log_likelihoods, bins=n_embedding_bins, density=True)

                ## Shuffle each particle's distance matrix n_shuffle times (too large to do all in one fori-loop).
                shuffled_log_likelihoods = jnp.zeros((n_particles * n_shuffle))
                for i in range(n_particles):
                    key, subkey = jax.random.split(key)
                    shuffled_distances = jax.random.permutation(subkey, np.tile(distance_trace[i], (n_shuffle, 1)), axis=1, independent=True)
                    get_ll = lambda i, ll: ll_f(i,ll, distances=shuffled_distances, obs=obs[si, ti], sigma_T_trace=sigma_T_trace)
                    start, end = i*n_shuffle, (i+1)*n_shuffle
                    shuffled_log_likelihoods = shuffled_log_likelihoods.at[start:end].set(jax.lax.fori_loop(0, n_shuffle, get_ll, jnp.zeros((n_shuffle))))
                shuf_hist, shuf_bins = jnp.histogram(shuffled_log_likelihoods, bins=n_shuffle_bins, density=True)

                ## Plot log-likelihood distributions for true embedding and shuffled version
                plt.figure()
                plt.stairs(emb_hist, emb_bins, fill=True, label='embedding', color='red', alpha=plot_alpha)
                plt.stairs(shuf_hist, shuf_bins, fill=True, label='shuffled', color='tab:gray', alpha=plot_alpha)
                plt.xlabel('Log-likelihood', fontsize=label_fontsize)
                plt.xticks(fontsize=tick_fontsize)
                ax = plt.gca()
                ax.set_xticks(ax.get_xticks()[::2])
                plt.ylabel('Density', fontsize=label_fontsize)
                plt.yticks(fontsize=tick_fontsize)
                plt.legend(fontsize=label_fontsize)
                plt.tight_layout()
                savename = get_filename_with_ext(f"emb_vs_shuf_loglikelihood_S{n_sub}_{task}_{base_data_filename}", ext='png', partial=partial, bpf=bpf, folder=figure_folder)
                plt.savefig(savename)
                plt.close()

            ## Get average correlation and sort by those edges
            avg_obs = jnp.mean(obs[si, ti], axis=0)
            pred_sort = jnp.argsort(avg_obs)

            ## Get average and standard deviation of the predictions (with sorted versions)
            avg_prediction = jnp.mean(predictions, axis=0)
            plt_avg_pred = avg_prediction[pred_sort]
            std_prediction = jnp.std(predictions, axis=0)
            plt_std_pred = std_prediction[pred_sort]

            ## Plot generated data
            x_plt = np.arange(M)
            plt.figure()
            plt.xlabel('edge index', fontsize=label_fontsize)
            plt.xticks(fontsize=tick_fontsize, ha='right', rotation=45)
            ylab = 'correlation' if edge_type == 'con' else 'edge value'
            plt.ylabel(ylab, fontsize=label_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            s = 2 # Actual marker size
            bigs = 15 # Marker size in the legend
            farout = 1000 # Coordinate of the legend point
            plt.scatter(x_plt, plt_avg_pred, s=s, color='royalblue', alpha=plot_alpha)

            ## Plot observations
            for oi, obs_i in enumerate(obs[si, ti]):
                sc = plt.scatter(x_plt, obs_i[pred_sort], s=s, color='pink', alpha=plot_alpha)
            plt.scatter(x_plt, avg_obs[pred_sort], s=s, color='red')

            xmin, xmax = plt.xlim()
            ## Create legend
            plt.scatter(-farout, .5, s=bigs, color='royalblue', alpha=plot_alpha, label='average prediction')
            plt.scatter(-farout, .5, s=bigs, color='pink', alpha=plot_alpha, label='observation')
            plt.scatter(-farout, .5, s=bigs, color='red', label='average observation')
            plt.legend(fontsize=label_fontsize, bbox_to_anchor=(1.0, 1.0))
            savename = get_filename_with_ext(f"{edge_type}_{geometry}_ppc_edges_S{n_sub}_{task}_{base_data_filename}", ext='png', partial=partial, bpf=bpf, folder=figure_folder)
            plt.savefig(savename, bbox_inches='tight')
            plt.close()

            ## Imshow figures of average prediction and average observation
            fig, axs = plt.subplots(2)
            fig.suptitle(f"S{n_sub} {task}")
            axs[0].imshow(triu2mat(avg_obs), cmap=cmap)
            axs[0].set_title("Average observation")
            axs[0].axis('off')

            axs[1].imshow(triu2mat(avg_prediction), cmap=cmap)
            axs[1].set_title("Average prediction")
            axs[1].axis('off')

            savename = get_filename_with_ext(f"{edge_type}_{geometry}_imshow_ppc_S{n_sub}_{task}_{base_data_filename}", ext='png', partial=partial, bpf=bpf, folder=figure_folder)
            plt.savefig(savename, bbox_inches='tight')
            plt.close()

            ## Imshow figure of differences between prediction and observation
            pred_diff = triu2mat(avg_prediction) - triu2mat(avg_obs)
            pred_diff_sorted = np.zeros((N, N))
            for row in range(N):
                pred_diff_sorted[row] = pred_diff[idc[row]][idc]

            plt.figure()
            plt.imshow(pred_diff_sorted, cmap=cmap, vmin=-1, vmax=1)
            plt.yticks(ticks=first_idc, labels=uordered, fontsize=tick_fontsize)
            plt.xticks(ticks=first_idc, labels=uordered, rotation=75, fontsize=tick_fontsize)
            cbar = plt.colorbar()
            ax = plt.gca()
            ax.xaxis.tick_top()
            savename = get_filename_with_ext(f"{edge_type}_{geometry}_imshow_diff_ppc_S{n_sub}_{task}_{base_data_filename}", ext='png', partial=partial, bpf=bpf, folder=figure_folder)
            plt.savefig(savename, bbox_inches='tight')
            plt.close()

            ## Actual vs predicted scatterplot
            plt.figure()
            plt.scatter(avg_obs, avg_prediction, s=1, c='k')
            plt.xlabel('Average observation', fontsize=label_fontsize)
            plt.xticks(fontsize=tick_fontsize)
            plt.ylabel('Average prediction', fontsize=label_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            savename = get_filename_with_ext(f"{edge_type}_{geometry}_scatter_ppc_S{n_sub}_{task}_{base_data_filename}", ext='png', partial=partial, bpf=bpf, folder=figure_folder)
            plt.savefig(savename, bbox_inches='tight')
            plt.close()

if save_errors:
    with open(error_savename, 'wb') as f:
        pickle.dump(prediction_errors, f)
    
    avg_prediction_errors = np.mean(prediction_errors, axis=2) # Take average over particles
    with open(avg_error_savename, 'wb') as f:
        pickle.dump(avg_prediction_errors, f)