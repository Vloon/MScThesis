# Home brew functions
from helper_functions import set_GPU, get_cmd_params, get_filename_with_ext, get_safe_folder, load_observations, get_attribute_from_trace, \
    logit, invlogit, euclidean_distance, is_valid, get_plt_labels, key2str, print_versions
from bookstein_methods import get_bookstein_anchors, bookstein_position, smc_bookstein_position, add_bkst_to_smc_trace, smc_bkst_inference_loop
from plotting_functions import plot_posterior, plot_network
from functools import partial as deco_partial

# Basics
import pickle
import time
import os
import csv
import numpy as np

import matplotlib.pyplot as plt

# Sampling
import jax
import jax.numpy as jnp
from jax._src.prng import random_wrap
from jax._src.prng import threefry_prng_impl
import jax.scipy.stats as jstats
import blackjax as bjx
import blackjax.smc.resampling as resampling

# Typing
from jax._src.prng import PRNGKeyArray
from jax._src.typing import ArrayLike
from blackjax.types import PyTree
from blackjax.mcmc.rmh import RMHState
from blackjax.smc.tempered import TemperedSMCState
from blackjax.smc.base import SMCInfo
from typing import Callable, Tuple

# Keep this here in case you somehow import the file and need these constants??
eps = 1e-5 # If eps < 1e-5, rounding to zero can start to happen... DON'T TEMPT IT BOY!
obs_eps = 1e-12
mu = [0., 0.]
sigma = 1.
mu_sigma = 0.
sigma_sigma = 1.
overwrite_sigma_over_bound = None
N = 164
D = 2
s1 = 1
sn = 100
n_particles = 2_000
n_mcmc_steps = 500
rmh_sigma = 1e-2
bookstein_dist = 0.3
overwrite_data_filename = None
data_folder = 'Data'
base_data_filename = 'processed_data_downsampled_evenly_spaced'
task_filename = 'task_list'
base_output_folder = 'Embeddings'
make_plot = False
figure_folder = 'Figures'
label_location = 'Figures/lobelabels.npz'
r_margin = 0.1
plot_threshold = 0.4
do_print = False
save_stats = False
save_sigma_filename = 'con_euc_sbt'
stats_filename = 'statistics'
stats_folder = 'Statistics'
dl = ';'
seed_file = 'seed.txt'
seed = None
key_data = None
gpu = ''

if __name__ == "__main__":
    # Create cmd argument list (arg_name, var_name, type, default [xcpt BOOL], nargs[OPT])
    arguments = [('-e', 'eps', float, eps),  # d->d_max offset
                 ('-obseps', 'obs_eps', float, obs_eps),  # observation clipping offset
                 ('-m', 'mu', float, mu, '+'),  # mean of distribution to sample _z
                 ('-s', 'sigma', float, sigma),  # std of distribution to sample _z
                 ('-ms', 'mu_sigma', float, mu_sigma),  # mean of distribution to sample sigma_T
                 ('-ss', 'sigma_sigma', float, sigma_sigma),  # std of distribution to sample sigma_T
                 ('-overwritesig', 'overwrite_sigma_over_bound', float, overwrite_sigma_over_bound), # overwrite sigma_T with this value if not None
                 ('-N', 'N', int, N),  # number of nodes
                 ('-D', 'D', int, D),  # latent space dimensions
                 ('-s1', 'subject1', int, s1),  # first subject to be used
                 ('-sn', 'subjectn', int, sn),  # last subject to be used
                 ('-np', 'n_particles', int, n_particles),  # number of smc particles
                 ('-nm', 'n_mcmc_steps', int, n_mcmc_steps),  # number of mcmc steps within smc
                 ('-r', 'rmh_sigma', float, rmh_sigma),  # sigma of the RMH sampler within SMC
                 ('-bdist', 'bookstein_dist', float, bookstein_dist),  # distance between the bookstein anchors
                 ('-overwritedf', 'overwrite_data_filename', str, overwrite_data_filename),  # if used, it overwrites the default filename
                 ('-datfol', 'data_folder', str, data_folder),  # folder where the data is stored
                 ('-bdf', 'base_data_filename', str, base_data_filename),  # filename of the saved data
                 ('-tf', 'task_filename', str, task_filename),  # filename of the list of task names
                 ('-of', 'base_output_folder', str, base_output_folder),  # folder where to dump the LSM embeddings
                 ('--plot', 'make_plot', bool),  # whether to create a plot
                 ('-ff', 'figure_folder', str, figure_folder),  # base folder where to dump the figures
                 ('-lab', 'label_location', str, label_location),  # file location of the labels
                 ('-rmarg', 'r_margin', float, r_margin), # offset for the radius of the disk drawn around the positions
                 ('-plotth', 'plot_threshold', float, plot_threshold),  # threshold for plotting edges
                 ('--print', 'do_print', bool),  # whether to print cute info
                 ('--stats', 'save_stats', bool),  # whether to save the statistics in a csv
                 ('--savesigma', 'save_sigma_chain', bool), # whether to save the sigma proposals in a pkl file for checking convergence
                 ('-ssf', 'save_sigma_filename', str, save_sigma_filename),  # base filename for saving sigma proposals
                 ('--partial', 'partial', bool),  # whether to use partial correlations
                 ('--bpf', 'bpf', bool),  # whether to use band-pass filtered rs-fMRI data
                 ('--nolabels', 'no_labels', bool),  # whether to not use labels
                 ('-stf', 'stats_filename', str, stats_filename),  # statistics filename
                 ('-stfl', 'stats_folder', str, stats_folder),  # statistics folder
                 ('-dl', 'dl', str, dl),  # save stats delimeter
                 ('-seedfile', 'seed_file', str, seed_file),  # save file for the seed
                 ('-seed', 'seed', int, seed),  # starting random key
                 ('-keydata', 'key_data', int, key_data, '+'), # If given, replaces the seed-based key initialization with directly wrapping an integer array to PRNG key
                 ('-gpu', 'gpu', str, gpu), # number of gpu to use (in string form). If no GPU is specified, CPU is used.
                 ]

    # Get arguments from CMD
    global_params = get_cmd_params(arguments)
    set_GPU(global_params['gpu']) ### MUST BE RUN FIRST
    eps = global_params['eps']
    obs_eps = global_params['obs_eps']
    mu = global_params['mu']
    sigma = global_params['sigma']
    mu_sigma = global_params['mu_sigma']
    overwrite_sigma_over_bound = global_params['overwrite_sigma_over_bound']
    sigma_sigma = global_params['sigma_sigma']
    N = global_params['N']
    M = N * (N - 1) // 2
    D = global_params['D']
    subject1 = global_params['subject1']
    subjectn = global_params['subjectn']
    n_particles = global_params['n_particles']
    n_mcmc_steps = global_params['n_mcmc_steps']
    rmh_sigma = global_params['rmh_sigma']
    bookstein_dist = global_params['bookstein_dist']
    data_folder = global_params['data_folder']
    overwrite_data_filename = global_params['overwrite_data_filename']
    base_data_filename = global_params['base_data_filename']
    task_filename = global_params['task_filename']
    output_folder = get_safe_folder(f"{global_params['base_output_folder']}/{n_particles}p{n_mcmc_steps}s")
    make_plot = global_params['make_plot']
    figure_folder = get_safe_folder(f"{global_params['figure_folder']}/{n_particles}p{n_mcmc_steps}s")
    label_location = global_params['label_location']
    r_margin = global_params['r_margin']
    plot_threshold = global_params['plot_threshold']
    do_print = global_params['do_print']
    save_stats = global_params['save_stats']
    save_sigma_chain = global_params['save_sigma_chain']
    save_sigma_filename = global_params['save_sigma_filename']
    partial = global_params['partial']
    bpf = global_params['bpf']
    no_labels = global_params['no_labels']
    stats_folder = get_safe_folder(stats_folder)
    stats_filename = get_filename_with_ext(global_params['stats_filename'], ext='csv', folder=stats_folder)
    dl = global_params['dl']
    seed_file = global_params['seed_file']
    seed = global_params['seed']
    key_data = global_params['key_data']

    # Initialize JAX stuff
    if do_print:
        print_versions()

    # USE JAX ONLY AFTER SETTING THE GPU, OTHERWISE IT WILL USE ALL GPUS
    if key_data is None:
        if seed is None:
            with open(seed_file, 'r') as f:
                seed = int(f.read())
        key = jax.random.PRNGKey(seed)
    else:
        assert len(key_data) == 2, f"key_data should have length 2 but has length {len(key_data)}"
        key = random_wrap(jnp.array(key_data, dtype=jnp.uint32), impl=threefry_prng_impl)

    mu = jnp.array(mu)

overwrite_sigma_val = logit(overwrite_sigma_over_bound) if overwrite_sigma_over_bound is not None else None

## Define functions to calculate the continuous hyperbolic parameters
def get_det_params(z:ArrayLike, eps:float=eps, **kwargs) -> ArrayLike:
    """
    Calculates all deterministicly dependent parameters, up to and including the bound of sigma_beta, and returns those in a dictionary.
    PARAMS:
    z : latent positions
    eps : offset for clipping mu-related variables
    **kwargs allows us to pass non-used parameters, which is handy when we want to allow other files to use the con hyp model but also others and allow this function to just catch the parameters it needs
    """
    N, D = z.shape
    triu_indices = jnp.triu_indices(N, k=1)

    d = euclidean_distance(z)[triu_indices]
    mu_beta = jnp.clip(jnp.exp(-d ** 2), eps, 1 - eps) # Clip means to be in (0,1) excl.
    bound = jnp.sqrt(mu_beta * (1 - mu_beta))

    params = {'z': z,
              'd': d,
              'mu_beta': mu_beta,
              'bound': bound}
    return params

def get_ab(mu_beta:ArrayLike, sigma_beta:ArrayLike, eps:float=eps) -> Tuple[ArrayLike, ArrayLike]:
    """
    Calculates the a and b parameters for the beta distribution
    PARAMS:
    mu_beta : mean of the beta distribution
    sigma_beta : standard deviations of the beta distribution
    eps : offset for calculating kappa, to ensure not dividing by zero
    """
    kappa = jnp.maximum(mu_beta*(1-mu_beta)/jnp.maximum(sigma_beta**2,eps) - 1, eps)
    a = mu_beta*kappa
    b = (1-mu_beta)*kappa
    return a, b

## Sampling functions
def sample_prior(key:PRNGKeyArray, shape:tuple, mu:ArrayLike=mu, sigma:float=sigma, mu_sigma:float=mu_sigma, sigma_sigma:float=sigma_sigma, overwrite_sigma_val:float=overwrite_sigma_val, **kwargs) -> Tuple[PRNGKeyArray, dict]:
    """
    Samples z positions from a normal distribution, as well as the shared transformed stds of the beta distributions.
    NOTE: the resulting z positions are not in Bookstein form. If you want to sample in proper Bookstein form, you can't yet scuzi.
    Returns the prior parameters in a dictionary.
    PARAMS:
    key : random key for JAX functions
    shape : shape of the prior positions
    mu : mean of the 2D Gaussian to sample latent positions z
    sigma : standard deviation of the 2D Gaussian to sample z
    mu_sigma : mean of the 1D Gaussian to sample sigma_beta_T
    sigma_sigma : standard deviation of the 1D Gaussian to sample sigma_beta_T
    overwrite_sigma_val : possibility to overwrite sigma
    """
    key, z_key, sigma_key, = jax.random.split(key, 3)
    z = sigma * jax.random.normal(z_key, shape=shape) + mu
    sigma_beta_T = jax.lax.select(overwrite_sigma_val is not None, overwrite_sigma_val, sigma_sigma*jax.random.normal(sigma_key, (1,)) + mu_sigma)
    
    prior = {'z': z,
             'sigma_beta_T': sigma_beta_T}
    return key, prior

def sample_observation(key:PRNGKeyArray, prior:dict, n_samples:int=1, mu:ArrayLike=mu, eps:float=eps, obs_eps:float=obs_eps, **kwargs) -> Tuple[PRNGKeyArray, jnp.array]:
    """
    Generates an observation based on the prior
    PARAMS:
    key : random key for JAX functions
    prior : dictionary containing sampled variables from the prior ('z', 'sigma_beta_T')
    n_samples : number of observations to sample
    mu : mean of the positions' normal distribution
    eps : offset for clipping mu-related variables
    obs_eps : offset for clipping A so that 0 < A < 1 instead of 0 <= A <= 1
    """
    # Get prior position and sigma
    z, sigma_beta_T = prior['z'], prior['sigma_beta_T']
    N = z.shape[0]
    M = N * (N - 1) // 2

    # Calculate mu and bound
    params = get_det_params(z, eps)
    mu_beta, bound = params['mu_beta'], params['bound']

    # Transform sigma_beta back
    sigma_beta = invlogit(sigma_beta_T) * bound

    # Calculate a, b parameters
    a, b = get_ab(mu_beta, sigma_beta, eps)

    # Sample observations A, and clip between eps and 1-eps
    key, subkey = jax.random.split(key)
    A = jnp.clip(jax.random.beta(subkey, a, b, shape=(n_samples, M)), obs_eps, 1. - obs_eps)
    return key, A

## Probability distributions
def log_prior(z:ArrayLike, sigma_beta_T:float, mu:ArrayLike=mu, sigma:float=sigma, mu_sigma:float=mu_sigma, sigma_sigma:float=sigma_sigma, overwrite_sigma:bool=False) -> float:
    """
    Returns the log-prior for a full _z state and sigma state, no Bookstein faffing.
    PARAMS:
    z : latent positions
    sigma_beta_T : transformed standard deviations of the beta distribution
    mu : mean of the positions' normal distribution
    sigma : standard deviation of the 2D Gaussian that is projected to the hyperbolic plane
    mu_sigma : mean of the 1D Gaussian that samples the transformed standard deviation of the beta distribution
    sigma_sigma : standard deviation of the 1D Gaussian that samples the transformed standard deviation of the beta distribution
    overwrite_sigma : whether we overwrite sigma with a set value (removing it from the RVs)
    """
    logprob_z = jstats.norm.logpdf(z, loc=mu, scale=sigma).sum()
    logprob_sigma_T = jax.lax.select(overwrite_sigma, jstats.norm.logpdf(sigma_beta_T, loc=mu_sigma, scale=sigma_sigma).sum(), 0.)  # Replace with 0 if we overwrite since it's not an RV
    return logprob_z + logprob_sigma_T

def bookstein_log_prior(z:ArrayLike, zb_x:float, sigma_beta_T:float, mu:ArrayLike=mu, sigma:float=sigma, mu_sigma:float=mu_sigma, sigma_sigma:float=sigma_sigma, overwrite_sigma:bool=False) -> float:
    """
    Returns the log-prior, taking into account Bookstein anchors.
    PARAMS:
    z : latent positions
    zb_x : x-coordinate of the 2nd Bookstein anchor. Its y-coordinate is always 0.
    sigma_beta_T : transformed standard deviations of the beta distribution
    mu : mean of the positions' normal distribution
    sigma : standard deviation of the 2D Gaussian that is projected to the hyperbolic plane
    mu_sigma : mean of the Gaussian of transformed sigma
    sigma_sigma : standard deviation of the Gaussian of transformed sigma
    overwrite_sigma : whether we overwrite sigma with a set value (removing it from the RVs)
    """
    zb_x_logprior = jstats.truncnorm.logpdf(zb_x, a=0, b=jnp.inf, loc=-bookstein_dist, scale=sigma).sum()  # Logprior for the node restricted in y = 0.
    zb_y_logprior = jnp.log(2) + jstats.norm.logpdf(z[0, :], loc=mu, scale=sigma).sum() - jnp.inf * (z[0, 1] < 0)  # Logprior for the node restricted in y>0 # Kan dit niet ook met een truncnorm?
    rest_logprior = log_prior(z[1:, :], sigma_beta_T, mu, sigma, mu_sigma, sigma_sigma, overwrite_sigma)  # Use not all _z values
    return rest_logprior + zb_x_logprior + zb_y_logprior

def log_likelihood(z:ArrayLike, sigma_beta_T:float, obs:ArrayLike, eps:float=eps, obs_eps:float=obs_eps) -> float:
    """
    Returns the log-likelihood
    PARAMS:
    z : latent positions
    sigma_beta_T : transformed standard deviations of the beta distribution
    obs : observed correlations (samples x edges)
    eps : offset for clipping mu-related variables
    obs_eps : offset for clipping the observations, to insure 0 < correlations < 1
    """
    params = get_det_params(z, eps)
    mu_beta, bound = params['mu_beta'], params['bound']
    sigma_beta = invlogit(sigma_beta_T) * bound  # Transform sigma_beta's back to [0, bound] to get a,b
    a, b = get_ab(mu_beta, sigma_beta)
    obs_clip = jnp.clip(obs, obs_eps, 1 - obs_eps)
    logprob_A = jstats.beta.logpdf(obs_clip, a, b)
    return logprob_A.sum()

def log_likelihood_from_dist(d:ArrayLike, sigma_beta_T:float, obs:ArrayLike, eps:float=eps, obs_eps:float=obs_eps) -> float:
    """
    Returns the log-likelihood of the state from the given distances rather than positions
    PARAMS
    dist : (M) upper triangle of the distance matrix
    sigma_beta_T : transformed standard deviations of the beta distribution
    obs : (n_obs x M) observed correlations
    eps : offset for clipping mu-related variables
    obs_eps : offset for clipping the observations, to insure 0 < correlations < 1
    """
    mu_beta = jnp.clip(jnp.exp(-d ** 2), eps, 1 - eps)
    bound = jnp.sqrt(mu_beta * (1 - mu_beta))
    sigma_beta = invlogit(sigma_beta_T) * bound
    a, b = get_ab(mu_beta, sigma_beta)
    obs_clip = jnp.clip(obs, obs_eps, 1 - obs_eps)
    logprob_A = jstats.beta.logpdf(obs_clip, a, b)
    return logprob_A.sum()

def bookstein_log_likelihood(z:ArrayLike, zb_x:float, sigma_beta_T:float, obs:ArrayLike, bookstein_dist:float=bookstein_dist) -> float:
    """
    Returns the log-likelihood given that _z is missing its Bookstein anchors
    PARAMS:
    z : latent positions (without bookstein anchors)
    zb_x : x-coordinate of the 2nd Bookstein anchor. Its y-coordinate is always 0.
    sigma_beta_T : transformed standard deviations of the beta distribution
    obs : observed correlations (samples x edges)
    bookstein_dist : offset of the first Bookstein anchor
    """
    n_dims = z.shape[1]
    bookstein_anchors = get_bookstein_anchors(zb_x, n_dims, bookstein_dist)

    # Concatenate bookstein anchors to z
    zc = jnp.concatenate([bookstein_anchors, z])
    return log_likelihood(zc, sigma_beta_T, obs)

def log_density(z:ArrayLike, sigma_beta_T:float, obs:ArrayLike, mu:float=mu, sigma:float=sigma, mu_sigma:float=mu_sigma, sigma_sigma:float=sigma_sigma) -> float:
    """
    Returns the log-probability density of the observed edge weights under the Continuous Hyperbolic LSM.
    PARAMS:
    z : latent positions
    sigma_beta_T : transformed standard deviations of the beta distributions
    obs : observed correlations (samples x edges)
    mu : mean of the latent positions' normal distribution
    sigma : standard deviation of the latent positions' normal distribution
    mu_sigma : mean of the Gaussian of transformed sigma
    sigma_sigma : standard deviation of the Gaussian of transformed sigma
    """
    prior_prob = log_prior(z, sigma_beta_T, mu, sigma, mu_sigma, sigma_sigma)
    likelihood_prob = log_likelihood(z, sigma_beta_T, obs)
    return prior_prob + likelihood_prob

## SMC + Bookstein methods
def initialize_bkst_particles(key:PRNGKeyArray, num_particles:int, shape:tuple, mu:float=mu, sigma:float=sigma, mu_sigma:float=mu_sigma, sigma_sigma:float=sigma_sigma, bookstein_dist:float=bookstein_dist, overwrite_sigma:bool=False) -> Tuple[PRNGKeyArray, dict]:
    """
    Initializes the SMC particles, but with Bookstein coordinates.
    PARAMS:
    key : random key for JAX functions
    num_particles : number of SMC particles
    shape : number of nodes by number of dimensions
    mu : mean of the latent positions' normal distribution
    sigma : std of the positions' 2D Gaussian
    mu_sigma : mean of the transformed std distribution
    sigma_sigma : std of the transformed std distribution
    bookstein_dist : offset of the first Bookstein anchor
    overwrite_sigma : whether to overwrite sigma
    """
    N, D = shape
    key, z_key, z_bx_key, sigma_beta_T_key = jax.random.split(key, 4)
    initial_position = { 'z': smc_bookstein_position(sigma * jax.random.normal(z_key, shape=(num_particles, N - D, D)) + mu), # N-D to skip first D bkst nodes. First node is rigid.
                         'zb_x': sigma * jax.random.truncated_normal(z_bx_key, lower=-bookstein_dist, upper=jnp.inf, shape=(num_particles, 1)), # Second node is restricted to just an x-position.
                       }

    if not overwrite_sigma:
        initial_position['sigma_beta_T'] = sigma_sigma * jax.random.normal(sigma_beta_T_key, shape=(num_particles, 1)) + mu_sigma
    return key, initial_position

def get_LSM_embedding(key:PRNGKeyArray, obs:ArrayLike, N:int=N, D:int=D, rmh_sigma:float=rmh_sigma, n_mcmc_steps:int=n_mcmc_steps, n_particles:int=n_particles, overwrite_sigma_val:float=overwrite_sigma_val) -> Tuple[PRNGKeyArray, int, float, TemperedSMCState]:
    """
    Creates a latent space embedding based on the given observations.
    Returns key,
    PARAMS:
    key: random key for JAX functions
    obs : (n_obs x M) upper triangles of the correlation matrices.
    N : number of nodes
    D : dimension of the latent space
    rmh_sigma : std of the within-SMC RMH sampler
    n_mcmc_steps : number of MCMC steps taken within each SMC iteration
    n_particles : number of SMC particles
    overwrite_sigma_val : value with which to overwrite sigma
    """
    # Define smc+bkst sampler
    overwrite_sigma = overwrite_sigma_val is not None
    # Define which paremeter sets are used for the prior/likelihood.
    # Other parameters are taken from global parameters
    if overwrite_sigma:
        _bookstein_log_prior = lambda state: bookstein_log_prior(**state, sigma_beta_T=overwrite_sigma_val, overwrite_sigma=True)
        _bookstein_log_likelihood = lambda state: bookstein_log_likelihood(**state, obs=obs, sigma_beta_T=overwrite_sigma_val)
    else:
        _bookstein_log_prior = lambda state: bookstein_log_prior(**state)
        _bookstein_log_likelihood = lambda state: bookstein_log_likelihood(**state, obs=obs)

    n_vars = (N - D) * D + 1 + 1 - (overwrite_sigma_val is not None)  # 1 for z_bx, 1 for sigma_beta_T (which we remove if we overwrite)
    rmh_parameters = {'sigma': rmh_sigma * jnp.eye(n_vars)}
    smc = bjx.adaptive_tempered_smc(
        logprior_fn=_bookstein_log_prior,
        loglikelihood_fn=_bookstein_log_likelihood,
        mcmc_algorithm=bjx.rmh,
        mcmc_parameters=rmh_parameters,
        resampling_fn=resampling.systematic,
        target_ess=0.5,
        mcmc_iter=n_mcmc_steps,
    )

    # Initialize the particles
    key, init_position = initialize_bkst_particles(key, n_particles, (N, D), overwrite_sigma=overwrite_sigma)
    initial_smc_state = smc.init(init_position)

    # Run SMC inference
    results = smc_bkst_inference_loop(key, smc.step, initial_smc_state)

    # Add Bookstein coordinates to SMC states
    states_rwm_smc = add_bkst_to_smc_trace(results[-1], bookstein_dist, 'z', D)

    if overwrite_sigma_val is None:
        key, n_iter, lml, sigma_trace, _ = results
        return key, n_iter, lml, sigma_trace, states_rwm_smc
    else:
        key, n_iter, lml, _ = results
        return key, n_iter, lml, states_rwm_smc

if __name__ == "__main__":
    """
    Data is in a dictionary, where the keys are defined by "S{n_sub}_{task}_{enc}", e.g. "S1_EMOTION_RL".
    The values per key are the upper triangle of the correlation matrix (length M).
    For each subject/task, we take both encodings as seperate observations to create 1 embedding. 
    """

    # Load labels
    plt_labels = get_plt_labels(label_location, make_plot, no_labels, N)

    # Load data
    if not overwrite_data_filename:
        data_filename = get_filename_with_ext(base_data_filename, partial, bpf, folder=data_folder)
    else:
        data_filename = overwrite_data_filename
    task_filename = get_filename_with_ext(task_filename, ext='txt', folder=data_folder)
    obs, tasks, encs = load_observations(data_filename, task_filename, subject1, subjectn, M)

    for si, n_sub in enumerate(range(subject1, subjectn + 1)):
        for ti, task in enumerate(tasks):
            # Create LS embedding
            start_time = time.time()
            results = get_LSM_embedding(key, obs[si, ti, :,:])  # Other parameters to get_LSM_embeddings are taken from globals.
            end_time = time.time()
            if overwrite_sigma_val is not None:
                key, n_iter, lml, smc_embedding = results
            else:
                key, n_iter, lml, sigma_trace, smc_embedding = results

            if do_print:
                print(f'Embedded S{n_sub}_{task} in {n_iter} iterations')

            # Save the statistics to the csv file (subject, task, n_particles, n_mcmc_steps, lml, runtime)
            if save_stats:
                stats_row = [f'S{n_sub}', task, n_particles, n_mcmc_steps, lml, end_time - start_time]
                with open(stats_filename, 'a', newline='') as f:
                    writer = csv.writer(f, delimiter=dl)
                    writer.writerow(stats_row)

            # Save sigma values
            if save_sigma_chain and overwrite_sigma_val is None:
                filename = get_filename_with_ext(f"{save_sigma_filename}_S{n_sub}_{task}_{base_data_filename}", partial=partial, folder=f"{data_folder}/sbt_traces")
                with open(filename, 'wb') as f:
                    pickle.dump(sigma_trace[:n_iter], f)

            partial_txt = '_partial' if partial else ''
            set_sigma_txt = f"_sigma_set_{overwrite_sigma_val:.1f}" if overwrite_sigma_val is not None else ''
            base_save_filename = f"con_euc_S{n_sub}_{task}_embedding_{base_data_filename}{partial_txt}{set_sigma_txt}"

            if make_plot:
                z_positions = smc_embedding.particles['z']
                radii = jnp.sqrt(jnp.sum(z_positions ** 2, axis=2))
                max_r = jnp.max(radii)

                ## TODO: what is the proper way to show the edges, there are 2 observations
                # Plot posterior
                plt.figure()
                ax = plt.gca()
                plot_posterior(z_positions,
                               edges=obs[si, ti, 0],
                               pos_labels=plt_labels,
                               ax=ax,
                               title=f"Proposal S{n_sub} {task}",
                               hyperbolic=False,
                               continuous=True,
                               bkst=True,
                               disk_radius=max_r,
                               margin=r_margin,
                               threshold=plot_threshold)
                poincare_disk = plt.Circle((0, 0), max_r * (1 + r_margin), color='k', fill=False, clip_on=False)
                ax.add_patch(poincare_disk)
                # Save figure
                savefile = get_filename_with_ext(base_save_filename, ext='png', partial=partial, folder=figure_folder)
                plt.savefig(savefile, bbox_inches='tight')
                plt.close()

            ## Save data
            embedding_filename = get_filename_with_ext(base_save_filename, partial=partial, bpf=bpf, folder=output_folder)
            info_filename = get_filename_with_ext(f"con_euc", ext='txt', folder=output_folder)
            with open(embedding_filename, 'wb') as f:
                pickle.dump(smc_embedding, f)
            with open(info_filename, 'a') as f:
                info_string = f"S{n_sub} Task {task} took {end_time - start_time:.4f}sec ({n_iter} iterations) with lml={lml:.4f}\n"
                f.write(info_string)

        # Add an empty line between each subject in the info file
        with open(info_filename, 'a') as f:
            f.write('\n')

    ## Save the new seed
    with open(seed_file, 'w') as f:
        f.write(key2str(key))
    if do_print:
        print('Done')
