import numpy as np
import matplotlib.pyplot as plt
import pickle
import jax
import jax.numpy as jnp
import os
import time
from jax._src.typing import ArrayLike
from jax._src.prng import PRNGKeyArray
from blackjax.smc.tempered import TemperedSMCState
from typing import Callable, Tuple

from helper_functions import set_GPU, get_cmd_params, get_filename_with_ext, get_safe_folder, load_observations, get_attribute_from_trace, triu2mat

### Within edge-type log-likelihood from distance is the same, regardless of the geometry. So bin_euc_loglikelihood_from_distance = bin_hyp_loglikelihood_from_distance
from binary_euclidean_LSM import get_det_params as bin_euc_det_params, sample_observation as bin_euc_sample_observation, log_likelihood_from_dist as bin_loglikelihood
from binary_hyperbolic_LSM import get_det_params as bin_hyp_det_params, sample_observation as bin_hyp_sample_observation
from continuous_euclidean_LSM import get_det_params as con_euc_det_params, sample_observation as con_euc_sample_observation, log_likelihood_from_dist as con_loglikelihood
from continuous_hyperbolic_LSM import get_det_params as con_hyp_det_params, sample_observation as con_hyp_sample_observation


arguments = [('-overwritedf', 'overwrite_data_filename', str, None),  # if used, it overwrites the default filename
             ('-datfol', 'data_folder', str, 'Data'),  # folder where the data is stored
             ('-conbdf', 'con_base_data_filename', str, 'processed_data'), # the most basic version of the filename of the continuous saved data
             ('-binbdf', 'bin_base_data_filename', str, 'binary_data_max_0.05unconnected'), # the most basic version of the filename of the binary saved data
             ('-ef', 'embedding_folder', str, 'Embeddings'), # base input folder of the embeddings
             ('-ff', 'figure_folder', str, 'Figures/sanity_checks/posterior_predictive_checks'), # figure output folder
             ('-tf', 'task_filename', str, 'task_list'), # filename of the list of task names
             ('-lab', 'label_location', str, 'Figures/lobelabels.npz'),  # file location of the labels
             ('-np', 'n_particles', int, 1000), # number of particles used in the embedding
             ('-nm', 'n_mcmc_steps', int, 100), # number of mcmc steps used in the embedding
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
             ('--partial', 'partial', bool), # whether to use partial correlations
             ('--bpf', 'bpf', bool), # whether to use band-pass filtered rs-fMRI data
             ('--shuf', 'do_shuffle', bool), # whether to do the whole shuffling thang
             ('-seed', 'seed', int, 1234), # starting random key
             ('-gpu', 'gpu', str, ''),  # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]

global_params = get_cmd_params(arguments)
set_GPU(global_params['gpu'])
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
do_shuffle = global_params['do_shuffle']

### JAX stuff
key = jax.random.PRNGKey(global_params['seed'])    

### Get correct variables based on edge type and geometry
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

# Load labels
label_data = np.load(label_location)
plt_labels = label_data[label_data.files[0]]
if len(plt_labels) != N:
    plt_labels = None

def get_con_log_likelihood(i:int, log_likelihoods:ArrayLike, distances:ArrayLike, sigma_T_trace:ArrayLike, obs:ArrayLike) -> ArrayLike:
    """
    Returns the log-likelihood of the model given continuous data based on the distance.
    PARAMS:
    i : particle / shuffle index
    log_likelihoods : (n_particles,) to be returned array of log-likelihoods
    distances : (n_particles,M) to be used distance array
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
    i : particle / shuffle index
    log_likelihoods : (n_particles,) to be returned array of log-likelihoods
    distances : (n_particles,M) to be used distance array
    obs : (n_obs, M) observation array
    **kwargs will catch the sigma_beta_T (always None here) which is passed to generalize the code. It ugly but it work. 
    """
    next_ll = bin_loglikelihood(distances[i], obs=obs)
    log_likelihoods = log_likelihoods.at[i].set(next_ll)
    return log_likelihoods

def get_obs_samples(i:int, state:Tuple[PRNGKeyArray,ArrayLike], parameters:list, embedding:TemperedSMCState, set_sigma:float=set_sigma, sample_obs_func:Callable=sample_obs_func):
    """
    Returns a sampled observation, based on the given posterior (posterior predictive check). 
    PARAMS:
    i : particle index
    state: tuple containing key, predictions
        key : jax random key
        predictions (n_particles, M) : prediced observations (to be filled in)
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
    predictions = predictions.at[i].set(pred[0]) # Take first element because it's technically a 1,M array which doesn't fit in a M sized space
    return key, predictions

ll_f = get_con_log_likelihood if edge_type == 'con' else get_bin_log_likelihood

### Get data that was trained on
if not overwrite_data_filename:
    data_filename = get_filename_with_ext(base_data_filename, partial, bpf, folder=data_folder)
else:
    data_filename = overwrite_data_filename
task_filename = get_filename_with_ext(task_filename, ext='txt', folder=data_folder)
obs, tasks, encs = load_observations(data_filename, task_filename, subject1, subjectn, M)

for si, n_sub in enumerate(range(subject1, subjectn+1)):
    for ti, task in enumerate(tasks):
        print(f"Running S{n_sub}, {task}")
        ### Load embedding and get distances
        
        embedding_filename = get_filename_with_ext(f"{edge_type}_{geometry}_S{n_sub}_{task}_embedding_{base_data_filename}{sigma_txt}", partial, bpf, folder=embedding_folder)
        with open(embedding_filename, 'rb') as f:
            embedding = pickle.load(f)

        distance_trace = get_attribute_from_trace(embedding.particles[latpos], det_params_func, 'd')  # -> (n_particles, M)

        d_avg = np.mean(distance_trace, axis=0) # -> (M,)
        d_avg = triu2mat(d_avg)

        ### Sort distance matrix by lobe
        lobes = [l[1].split(';')[0] for l in plt_labels]
        idc = np.argsort(lobes)
        d_sorted = np.zeros((N, N))
        for row in range(N):
            d_sorted[row] = d_avg[idc[row]][idc]

        ordered = np.sort(lobes)
        uordered = np.unique(ordered)
        first_idc = [list(ordered).index(ul) for ul in uordered]

        plt.figure()
        plt.imshow(d_sorted, cmap=cmap)
        plt.yticks(ticks=first_idc, labels=uordered)
        plt.xticks(ticks=first_idc, labels=uordered, rotation=75)
        ax = plt.gca()
        ax.xaxis.tick_top()
        plt.title(f"Distance matrix sorted by lobe\nS{n_sub} {task}")
        savename = get_filename_with_ext(f"dist_matrix_by_lobe_S{n_sub}_{task}_{base_data_filename}", ext='png', partial=partial, bpf=bpf, folder=figure_folder)
        plt.savefig(savename, bbox_inches='tight')
        plt.close()
        
        ### Make shuffled distance distribution checks
        if do_shuffle:
            sigma_T_trace = embedding.particles['sigma_beta_T'] if edge_type == 'con' else None
            embedding_log_likelihoods = jnp.zeros(n_particles)
            get_ll = lambda i, ll: ll_f(i,ll, distances=distance_trace, obs=obs[si, ti], sigma_T_trace=sigma_T_trace) ## Curry the observations, distance and sigma
            embedding_log_likelihoods = jax.lax.fori_loop(0, n_particles, get_ll, embedding_log_likelihoods)
            emb_hist, emb_bins = jnp.histogram(embedding_log_likelihoods, bins=n_embedding_bins, density=True)

            ### Shuffle each particle's distance matrix n_shuffle times
            ### THIS IS TOO BIG TO BE IN ONE BIG FORI_LOOP! :(
            shuffled_log_likelihoods = jnp.zeros((n_particles * n_shuffle))
            for i in range(n_particles):
                key, subkey = jax.random.split(key)
                shuffled_distances = jax.random.permutation(subkey, np.tile(distance_trace[i], (n_shuffle, 1)), axis=1, independent=True) # -> (n_shuffle, M)
                get_ll = lambda i, ll: ll_f(i,ll, distances=shuffled_distances, obs=obs[si, ti], sigma_T_trace=sigma_T_trace)
                start, end = i*n_shuffle, (i+1)*n_shuffle
                shuffled_log_likelihoods = shuffled_log_likelihoods.at[start:end].set(jax.lax.fori_loop(0, n_shuffle, get_ll, jnp.zeros((n_shuffle))))
            shuf_hist, shuf_bins = jnp.histogram(shuffled_log_likelihoods, bins=n_shuffle_bins, density=True)

            plt.figure()
            plt.stairs(emb_hist, emb_bins, fill=True, label='embedding', color='red', alpha=plot_alpha)
            plt.stairs(shuf_hist, shuf_bins, fill=True, label='shuffled', color='tab:gray', alpha=plot_alpha)
            plt.xlabel('log-likelihood')
            plt.ylabel('density')
            plt.title(f"S{n_sub} {task}")
            plt.legend()
            plt.tight_layout()
            savename = get_filename_with_ext(f"emb_vs_shuf_loglikelihood_S{n_sub}_{task}_{base_data_filename}", ext='png', partial=partial, bpf=bpf, folder=figure_folder)
            plt.savefig(savename)
            plt.close()

        ### Make prior predictive check
        # Create data from embedding
        parameters = [latpos, 'sigma_beta_T'] if edge_type == 'con' and set_sigma is None else [latpos]
        get_pred_func = lambda i, state: get_obs_samples(i, state, parameters, embedding)
        start = time.time()
        key, predictions = jax.lax.fori_loop(0, n_particles, get_pred_func, (key, jnp.zeros((n_particles, M)))) # love a good 4-stack
        end = time.time()

        # Get average correlation and sort by those edges
        avg_corr = jnp.mean(obs[si, ti], axis=0)  ## -> (M,)
        pred_sort = jnp.argsort(avg_corr)

        # Get averages over the predictions
        avg_prediction = jnp.mean(predictions, axis=0)
        plt_avg_pred = avg_prediction[pred_sort]
        std_prediction = jnp.std(predictions, axis=0)
        plt_std_pred = std_prediction[pred_sort]

        # Plot fake data
        x_plt = np.arange(M)
        plt.figure()
        plt.xlabel('edge index')
        plt.ylabel('correlation')
        plt.scatter(x_plt, plt_avg_pred, s=1, color='royalblue', alpha=plot_alpha, label='average predicted')
        plt.fill_between(x_plt, plt_avg_pred - plt_std_pred, plt_avg_pred + plt_std_pred, color='lightskyblue', alpha=plot_alpha, label='std predicted')

        # Plot observations (later because z-order)
        for oi, obs_i in enumerate(obs[si, ti]):
            sc = plt.scatter(x_plt, obs_i[pred_sort], s=1, color='pink', alpha=plot_alpha)
        sc.set_label('observed') # Set only last once
        plt.scatter(x_plt, avg_corr[pred_sort], s=1, color='red', label='average observed')
        plt.legend()
        plt.title(f"S{n_sub} {task} edges ordered by average edge weight")
        plt.tight_layout()
        savename = get_filename_with_ext(f"{edge_type}_{geometry}_posterior_predictive_check_S{n_sub}_{task}_{base_data_filename}", ext='png', partial=partial, bpf=bpf, folder=figure_folder)
        plt.savefig(savename)
        plt.close()

        ## Same but with imshows
        fig, axs = plt.subplots(2)
        fig.suptitle(f"S{n_sub} {task}")
        axs[0].imshow(triu2mat(avg_corr), cmap=cmap)
        axs[0].set_title("Average observed edge weight")
        axs[0].axis('off')

        axs[1].imshow(triu2mat(avg_prediction), cmap=cmap)
        axs[1].set_title("Average predicted edge weight")
        axs[1].axis('off')

        savename = get_filename_with_ext(f"{edge_type}_{geometry}_imshow_posterior_predictive_check_S{n_sub}_{task}_{base_data_filename}", ext='png', partial=partial, bpf=bpf, folder=figure_folder)
        plt.savefig(savename, bbox_inches='tight')
        plt.close()
