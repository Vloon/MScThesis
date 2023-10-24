# Home brew functions
from helper_functions import set_GPU, get_cmd_params, get_filename_with_ext, get_safe_folder, load_observations, get_attribute_from_trace, \
    invlogit, lorentz_to_poincare, hyp_pnt, lorentz_distance, parallel_transport, exponential_map, is_valid
from bookstein_methods import get_bookstein_anchors, bookstein_position, smc_bookstein_position, add_bkst_to_smc_trace
from plotting_functions import plot_posterior, plot_network

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
eps = 1e-5 # If eps < 1e-6, rounding to zero can start to happen... DON'T TEMPT IT BOY!
obs_eps = 1e-12
mu = [0., 0.]
sigma = 1.
mu_sigma = 0.
sigma_sigma = 1.
overwrite_sigma = None
N = 164
D = 2
s1 = 1
sn = 25
n_particles = 1_000
n_mcmc_steps = 100
rmh_sigma = 1e-2
bookstein_dist = 0.3
data_folder = 'Data'
base_data_filename = 'processed_data'
task_filename = 'task_list'
base_output_folder = 'Embeddings'
make_plot = False
figure_folder = 'Figures'
label_location = 'Figures/lobelabels.npz'
plot_threshold = .4
do_print = False
save_stats = False
save_sigma_filename = 'con_hyp_sbt'
stats_filename = 'statistics'
stats_folder = 'Statistics'
dl = ';'
seed_file = 'seed.txt'
seed = None
gpu = ''

if __name__ == "__main__":
    # Create cmd argument list (arg_name, var_name, type, default, nargs[OPT])
    arguments = [('-e', 'eps', float, eps),  # d->d_max offset
                 ('-obseps', 'obs_eps', float, obs_eps), # observation clipping offset
                 ('-m', 'mu', float, mu, '+'),  # mean of distribution to sample _z
                 ('-s', 'sigma', float, sigma),  # std of distribution to sample _z
                 ('-ms', 'mu_sigma', float, mu_sigma),  # mean of distribution to sample sigma_T
                 ('-ss', 'sigma_sigma', float, sigma_sigma),  # std of distribution to sample sigma_T
                 ('-overwritesig', 'overwrite_sigma', float, overwrite_sigma), # overwrite sigma_T with this value if not None
                 ('-N', 'N', int, N),  # number of nodes
                 ('-D', 'D', int, D),  # latent space dimensions
                 ('-s1', 'subject1', int, s1),  # first subject to be used
                 ('-sn', 'subjectn', int, sn),  # last subject to be used
                 ('-np', 'n_particles', int, n_particles),  # number of smc particles
                 ('-nm', 'n_mcmc_steps', int, n_mcmc_steps),  # number of mcmc steps within smc
                 ('-r', 'rmh_sigma', float, rmh_sigma), # sigma of the RMH sampler within SMC
                 ('-bdist', 'bookstein_dist', float, bookstein_dist), # distance between the bookstein anchors
                 ('-overwritedf', 'overwrite_data_filename', str, None), # if used, it overwrites the default filename
                 ('-datfol', 'data_folder', str, data_folder), # folder where the data is stored
                 ('-bdf', 'base_data_filename', str, base_data_filename),  # filename of the saved data
                 ('-tf', 'task_filename', str, task_filename), # filename of the list of task names
                 ('-of', 'base_output_folder', str, base_output_folder), # folder where to dump the LSM embeddings
                 ('--plot', 'make_plot', bool), # whether to create a plot
                 ('-ff', 'figure_folder', str, figure_folder), # base folder where to dump the figures
                 ('-lab', 'label_location', str, label_location), # file location of the labels
                 ('-plotth', 'plot_threshold', float, plot_threshold), # threshold for plotting edges
                 ('--print', 'do_print', bool), # whether to print cute info
                 ('--stats', 'save_stats', bool), # whether to save the statistics in a csv
                 ('--savesigma', 'save_sigma_chain', bool), # whether to save the sigma proposals in a pkl file for checking convergence
                 ('-ssf', 'save_sigma_filename', str, save_sigma_filename), # base filename for saving sigma proposals
                 ('--partial', 'partial', bool), # whether to use partial correlations
                 ('--bpf', 'bpf', bool), # whether to use band-pass filtered rs-fMRI data
                 ('-stf', 'stats_filename', str, stats_filename),  # statistics filename
                 ('-stfl', 'stats_folder', str, stats_folder),  # statistics folder
                 ('-dl', 'dl', str, dl),  # save stats delimeter
                 ('-seedfile', 'seed_file', str, seed_file), # save file for the seed
                 ('-seed', 'seed', int, seed), # starting random key
                 ('-gpu', 'gpu', str, gpu), # number of gpu to use (in string form). If no GPU is specified, CPU is used.
                 ]

    # Get arguments from CMD
    global_params = get_cmd_params(arguments)
    eps = global_params['eps']
    obs_eps = global_params['obs_eps']
    mu = global_params['mu']
    sigma = global_params['sigma']
    mu_sigma = global_params['mu_sigma']
    overwrite_sigma = global_params['overwrite_sigma']
    sigma_sigma = global_params['sigma_sigma']
    N = global_params['N']
    M = N*(N-1)//2
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
    plot_threshold = global_params['plot_threshold']
    do_print = global_params['do_print']
    save_stats = global_params['save_stats']
    save_sigma_chain = global_params['save_sigma_chain']
    save_sigma_filename = global_params['save_sigma_filename']
    partial = global_params['partial']
    bpf = global_params['bpf']
    stats_folder = get_safe_folder(stats_folder)
    stats_filename = get_filename_with_ext(global_params['stats_filename'], ext='csv', folder=stats_folder)
    dl = global_params['dl']
    seed_file = global_params['seed_file']
    seed = global_params['seed']
    set_GPU(global_params['gpu'])

    # Initialize JAX stuff
    if do_print:
        print('Using blackjax version', bjx.__version__)
        print(f'Running on {jax.devices()}')

    # USE JAX ONLY AFTER SETTING THE GPU, OTHERWISE IT WILL USE ALL GPUS
    if seed is None:
        with open(seed_file, 'r') as f:
            seed = int(f.read())
    key = jax.random.PRNGKey(seed)

    mu = jnp.array(mu)

def get_default_params() -> dict:
    """
    Returns a dictionary with default parameters for the get_det_params functions.
    Done this convoluted way so all default parameters are always kept in this class, and commandline values aren't overwritten.
    """
    return {'mu':mu, 'eps':eps}

## Define functions to calculate the continuous hyperbolic parameters
def get_det_params(_z:ArrayLike, mu:ArrayLike=mu, eps:float=eps) -> ArrayLike:
    """
    Calculates all deterministicly dependent parameters, up to and including the bound of sigma_beta, and returns those in a dictionary.
    Assumes that _z follows a Wrapped Hyperbolic normal distribution (Nagano et al. 2019)
    PARAMS:
    _z : pre-hyperbolic projection latent positions
    mu : mean of the wrapped hyperbolic normal prior
    eps : offset for calculating d_norm, to insure max(d_norm) < 1
    """
    mu = jnp.array(mu)
    N, D = _z.shape
    triu_indices = jnp.triu_indices(N, k=1)

    mu_0 = jnp.zeros((N, D+1))
    mu_0 = mu_0.at[:,0].set(1)
    
    # Mu can be a value (e.g. 0) to indicate (0,0), or array-like (e.g. [0,0])
    if hasattr(mu, "__len__"):
        assert len(mu) == D, 'Dimension of mu must correspond to the dimension of each point in _z'
        mu_tilde = jnp.reshape(jnp.tile(mu, N), (N,D))
    else:
        mu_tilde = mu*jnp.ones_like(_z)
        
    mu = hyp_pnt(mu_tilde) # UGLY?! How else to get a proper mean in H? Just kinda.. guess a correct H coordinate?
    v = jnp.concatenate([jnp.zeros((N,1)), _z], axis=1)
    u = parallel_transport(v, mu_0, mu)
    z = exponential_map(mu, u)

    d = lorentz_distance(z)[triu_indices]
    d_norm = jnp.clip(d/(jnp.max(d)), eps, 1.-eps) # Clip to make sure the boundaries are excluding 0, 1.
    mu_beta = 1.-d_norm
    bound = jnp.sqrt(mu_beta*(1.-mu_beta))

    params = {'_z':_z,
              'z':z,
              'd':d,
              'd_norm':d_norm,
              'mu_beta':mu_beta,
              'bound':bound}
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
def sample_prior(key:PRNGKeyArray, shape:tuple, sigma:float=sigma, mu_sigma:float=mu_sigma, sigma_sigma:float=sigma_sigma) -> Tuple[PRNGKeyArray, dict]:
    """
    Samples _z positions from the Wrapped Hyperbolic normal distribution (Nagano et al. 2019), as well as transformed stds of the beta distributions
    Returns the prior parameters in a dictionary.
    PARAMS:
    key : random key for JAX functions
    shape : shape of the prior positions
    sigma : standard deviation of the 2D Gaussian to sample _z
    mu_sigma : mean of the 1D Gaussian to sample sigma_beta_T
    sigma_sigma : standard deviation of the 1D Gaussian to sample sigma_beta_T
    """
    key, subkey = jax.random.split(key)
    _z = sigma*jax.random.normal(subkey, shape=shape) # Is always centered at 0

    key, subkey = jax.random.split(key)
    sigma_beta_T = sigma_sigma*jax.random.normal(subkey, (1,)) + mu_sigma

    prior = {'_z':_z,
            'sigma_beta_T':sigma_beta_T}
    return key, prior

def sample_observation(key:PRNGKeyArray, prior:dict, n_samples:int=1, eps:float=eps, obs_eps:float=obs_eps) -> Tuple[PRNGKeyArray, jnp.array]:
    """
    Generates an observation based on the prior
    PARAMS:
    key : random key for JAX functions
    prior : dictionary containing sampled variables from the prior ('_z', 'sigma_beta_T')
    n_samples : number of observations to sample
    eps : offset for the normalization of d
    obs_eps : offset for clipping A so that 0 < A < 1 instead of 0 <= A <= 1
    """
    # Get prior position and sigma
    _z, sigma_beta_T = prior['_z'], prior['sigma_beta_T']
    N = _z.shape[0]
    M = N*(N-1)//2

    # Calculate mu and bound
    params = get_det_params(_z, eps=eps)
    mu_beta, bound = params['mu_beta'], params['bound']
    
    # Transform sigma_beta back
    sigma_beta = invlogit(sigma_beta_T)*bound

    # Calculate a, b parameters
    a, b = get_ab(mu_beta, sigma_beta, eps)

    # Sample observations A, and clip between eps and 1-eps
    key, subkey = jax.random.split(key)
    A = jnp.clip(jax.random.beta(subkey, a, b, shape=(n_samples, M)), obs_eps, 1.-obs_eps)
    return key, A

## Probability distributions
def log_prior(_z:ArrayLike, sigma_beta_T:float, sigma:float=sigma, mu_sigma:float=mu_sigma, sigma_sigma:float=sigma_sigma) -> float:
    """
    Returns the log-prior
    PARAMS:
    _z : pre-hyperbolic transformed positions
    sigma_beta_T : transformed standard deviations of the beta distribution
    sigma : standard deviation of the 2D Gaussian that is projected to the hyperbolic plane
    mu_sigma : mean of the 1D Gaussian that samples the transformed standard deviation of the beta distribution
    sigma_sigma : standard deviation of the 1D Gaussian that samples the transformed standard deviation of the beta distribution
    """
    logprob__z = jstats.norm.logpdf(_z, loc=0., scale=sigma).sum()
    logprob_sigma_T = jstats.norm.logpdf(sigma_beta_T, loc=mu_sigma, scale=sigma_sigma).sum()
    return logprob__z + logprob_sigma_T

def bookstein_log_prior(_z:ArrayLike, sigma_beta_T:float, sigma:float=sigma, mu_sigma:float=mu_sigma, sigma_sigma:float=sigma_sigma) -> float:
    """
    Returns the log-prior for a network with bookstein coordinates
    PARAMS:
    _z : pre-hyperbolic transformed positions
    sigma_beta_T : transformed standard deviations of the beta distribution
    sigma : standard deviation of the 2D Gaussian that is projected to the hyperbolic plane
    mu_sigma : mean of the Gaussian of transformed sigma
    sigma_sigma : standard deviation of the Gaussian of transformed sigma
    """
    rest_logprior = log_prior(_z[1:,:], sigma_beta_T, sigma, mu_sigma, sigma_sigma) # Use all sigma_T's, not all _z values
    pivot_point_logprior = jnp.log(2)+jstats.norm.logpdf(_z[0,:], loc=0., scale=sigma).sum() + np.NINF*(_z[0,1] < 0.) # log[2*N(_z[2]|mu,sigma)] if _z[2] >= 0 else -INF.
    return rest_logprior + pivot_point_logprior

def log_likelihood(_z:ArrayLike, sigma_beta_T:float, obs:ArrayLike, obs_eps:float=obs_eps) -> float:
    """
    Returns the log-likelihood
    PARAMS:
    _z : x,y coordinates of the positions
    sigma_beta_T : transformed standard deviations of the beta distribution
    obs : observed correlations (samples x edges)
    obs_eps : offset for clipping the observations, to insure 0 < correlations < 1
    """
    params = get_det_params(_z, eps=eps)
    mu_beta, bound = params['mu_beta'], params['bound']
    sigma_beta = invlogit(sigma_beta_T)*bound # Transform sigma_beta back to [0, bound] to get a,b
    a,b = get_ab(mu_beta, sigma_beta)
    obs_clip = jnp.clip(obs, obs_eps, 1-obs_eps)
    logprob_A = jstats.beta.logpdf(obs_clip, a, b)
    return logprob_A.sum()

def bookstein_log_likelihood(_z:ArrayLike, sigma_beta_T:float, obs:ArrayLike, bookstein_dist:float=bookstein_dist) -> float:
    """
    Returns the log-likelihood given that _z is missing its Bookstein anchors
    PARAMS:
    _z : x,y coordinates of the positions, without bookstein anchors
    sigma_beta_T : transformed standard deviations of the beta distribution
    obs : observed correlations (samples x edges)
    bookstein_dist : distance between the Bookstein anchors (in the x-direction)
    """
    n_dims = _z.shape[1]
    bookstein_anchors = get_bookstein_anchors(n_dims, bookstein_dist)

    # Concatenate bookstein anchors to _z
    _zc = jnp.concatenate([bookstein_anchors, _z])
    return log_likelihood(_zc, sigma_beta_T, obs)

def log_density(_z:ArrayLike, sigma_beta_T:float, obs:ArrayLike,sigma:float=sigma) -> float:
    """
    Returns the log-probability density of the observed edge weights under the Continuous Hyperbolic LSM.
    PARAMS:
    _z : positions on Euclidean plane (pre hyperbolic projection)
    sigma_beta_T : transformed standard deviations of the beta distributions
    obs : observed correlations (samples x edges)
    sigma : standard deviation of the 2D Gaussian that is projected to the hyperbolic plane
    """
    prior_prob = log_prior(_z, sigma_beta_T, sigma)
    likelihood_prob = log_likelihood(_z, sigma_beta_T, obs)
    return prior_prob + likelihood_prob

def bookstein_log_density(_z:ArrayLike, sigma_beta_T:float, obs:ArrayLike, sigma:float=sigma, is_full_state:bool=False) -> float:
    """
    Returns the log-density using _z on Bookstein coordinates.
    PARAMS:
    _z : positions on Euclidean plane (pre hyperbolic projection)
    sigma_beta_T : transformed standard deviations of the beta distributions
    obs : observed correlations (samples x edges)
    sigma : standard deviation of the 2D Gaussian that is projected to the hyperbolic plane
    is_full_state : whether the _z is the full state (True) or the cut-off state used by SMC (False)
    """
    if is_full_state:
        return bookstein_log_prior(_z[2:,:], sigma_beta_T, sigma) + log_likelihood(_z, sigma_beta_T, obs)
    else:
        return bookstein_log_prior(_z, sigma_beta_T, sigma) + bookstein_log_likelihood(_z, sigma_beta_T, obs)

## SMC + Bookstein methods
def initialize_bkst_particles(key:PRNGKeyArray, num_particles:int, shape:tuple, sigma:float=sigma, mu_sigma:float=mu_sigma, sigma_sigma:float=sigma_sigma) -> Tuple[PRNGKeyArray, dict]:
    """
    Initializes the SMC particles, but with Bookstein coordinates.
    PARAMS:
    key : random key for JAX functions
    num_particles : number of SMC particles
    shape : number of nodes by number of dimensions
    sigma : std of the positions' 2D Gaussian
    mu_sigma : mean of the transformed std distribution
    sigma_sigma : std of the transformed std distribution
    """
    N, D = shape
    key, _z_key, sigma_beta_T_key = jax.random.split(key, 3)
    initial_position = {'_z': smc_bookstein_position(sigma*jax.random.normal(_z_key, shape=(num_particles, N-D, D))), # N-D to skip first D bkst nodes (they are implicit)
                        'sigma_beta_T': sigma_sigma*jax.random.normal(sigma_beta_T_key, shape=(num_particles, 1))+mu_sigma}
    return key, initial_position

def smc_bkst_inference_loop(key:PRNGKeyArray, smc_kernel:Callable, initial_state:ArrayLike) -> Tuple[PRNGKeyArray, float, TemperedSMCState]:
    """
    Run the temepered SMC algorithm with Bookstein anchoring.

    Run the adaptive algorithm until the tempering parameter lambda reaches the value lambda=1.
    PARAMS:
    key: random key for JAX functions
    smc_kernel: kernel for the SMC particles
    initial_state: beginning position of the algorithm
    """
    def cond(carry):
        state = carry[6]
        return state.lmbda < 1

    @jax.jit
    def step(carry):
        i, lml, lambdas, weights, ancestors, s_values, state, key = carry
        key, subkey = jax.random.split(key)
        state, info = smc_kernel(subkey, state)
        s_values = s_values.at[i].set(state.particles['sigma_beta_T'])
        ancestors = ancestors.at[i].set(info.ancestors)
        weights = weights.at[i].set(info.weights)
        lambdas = lambdas.at[i].set(state.lmbda)
        lml += info.log_likelihood_increment
        return i+1, lml, lambdas, weights, ancestors, s_values, state, key

    max_iters = 200
    n_iter, lml, lambdas, weights, ancestors, s_values, final_state, key = jax.lax.while_loop(
        cond, step, (0, 0., jnp.zeros((max_iters,)), jnp.zeros((max_iters, n_particles)), jnp.zeros((max_iters, n_particles)), jnp.zeros((max_iters, n_particles, 1)), initial_state, key)
    )
    if n_iter > max_iters and do_print:
        print(f"Embedding took {n_iter} iterations but max_iter is only {max_iters}! Not all proposals are saved.")
    return key, n_iter, lml, lambdas, weights, ancestors, s_values, final_state

def get_LSM_embedding(key:PRNGKeyArray, obs:ArrayLike, N:int=N, D:int=D, rmh_sigma:float=rmh_sigma, n_mcmc_steps:int=n_mcmc_steps, n_particles:int=n_particles) -> Tuple[PRNGKeyArray, int, float, TemperedSMCState]:
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
    """
    # Define smc+bkst sampler
    _bookstein_log_prior = lambda state: bookstein_log_prior(**state) # Parameters are taken from global parameters
    _bookstein_log_likelihood = lambda state: bookstein_log_likelihood(**state, obs=obs) # Parameters are taken from global parameters

    rmh_parameters = {'sigma': rmh_sigma * jnp.eye((N - D) * D + 1)}
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
    key, init_position = initialize_bkst_particles(key, n_particles, (N, D))
    initial_smc_state = smc.init(init_position)

    # Run SMC inference
    key, n_iter, lml, lambdas, weights, ancestors, s_values, states_rwm_smc = smc_bkst_inference_loop(key, smc.step, initial_smc_state)

    # Add Bookstein coordinates to SMC states
    states_rwm_smc = add_bkst_to_smc_trace(states_rwm_smc, bookstein_dist, D)

    return key, n_iter, lml, lambdas, weights, ancestors, s_values, states_rwm_smc

if __name__ == "__main__":
    """
    Data is in a dictionary, where the keys are defined by "S{n_sub}_{task}_{enc}", e.g. "S1_EMOTION_RL".
    The values per key are the upper triangle of the correlation matrix (length M).
    For each subject/task, we take both encodings as seperate observations to create 1 embedding. 
    """

    # Load labels
    if make_plot and label_location is not None:
        label_data = np.load(label_location)
        plt_labels = label_data[label_data.files[0]]
        if len(plt_labels) is not N:
            plt_labels = None
    else:
        plt_labels = None

    # Load data
    if not overwrite_data_filename:
        data_filename = get_filename_with_ext(base_data_filename, partial, bpf, folder=data_folder)
    else:
        data_filename = overwrite_data_filename
    task_filename = get_filename_with_ext(task_filename, ext='txt', folder=data_folder)
    obs, tasks, encs = load_observations(data_filename, task_filename, subject1, subjectn, M)

    for si, n_sub in enumerate(range(subject1, subjectn+1)):
        for ti, task in enumerate(tasks):
            # Create LS embedding
            start_time = time.time()
            key, n_iter, lml, lambdas, weights, ancestors, s_values, smc_embedding = get_LSM_embedding(key, obs[si, ti, :, :])  # Other parameters to get_LSM_embeddings are taken from globals.
            end_time = time.time()

            if do_print:
                print(f'Embedded S{n_sub}_{task} in {n_iter} iterations')

            # Save the statistics to the csv file (subject, task, n_particles, n_mcmc_steps, lml, runtime)
            if save_stats:
                stats_row = [f'S{n_sub}', task, n_particles, n_mcmc_steps, lml, end_time - start_time]
                with open(stats_filename, 'a', newline='') as f:
                    writer = csv.writer(f, delimiter=dl)
                    writer.writerow(stats_row)

            # Save sigma values
            if save_sigma_chain:
                filename = get_filename_with_ext(f"{save_sigma_filename}_S{n_sub}_{task}", folder=data_folder)
                with open(filename, 'wb') as f:
                    pickle.dump(s_values[:n_iter], f)

            partial_txt = '_partial' if partial else ''
            base_save_filename = f"con_hyp_S{n_sub}_{task}_embedding{partial_txt}"

            if make_plot:
                last__z_positions = smc_embedding.particles['_z']
                z_positions = lorentz_to_poincare(get_attribute_from_trace(last__z_positions, get_det_params, 'z', shape=(n_particles, N, D+1)))

                ## TODO: ADD LABELS!
                ## TODO: what is the proper way to show the edges, there are 2 observations
                # Plot posterior
                plt.figure()
                ax = plt.gca()
                plot_posterior(z_positions,
                               edges=obs[si, ti, 0],
                               pos_labels=plt_labels,
                               ax=ax,
                               title=f"Proposal S{n_sub} {task}",
                               hyperbolic=True,
                               bkst=True,
                               threshold=plot_threshold)
                poincare_disk = plt.Circle((0, 0), 1, color='k', fill=False, clip_on=False)
                ax.add_patch(poincare_disk)
                # Save figure
                savefile = get_filename_with_ext(base_save_filename, ext='png', folder=figure_folder)
                plt.savefig(savefile)
                plt.close()

            # Save data
            embedding_filename = get_filename_with_ext(base_save_filename, folder=output_folder)
            info_filename = get_filename_with_ext(f"con_hyp_S{n_sub}", ext='txt', folder=output_folder)
            with open(embedding_filename, 'wb') as f:
                pickle.dump(smc_embedding, f)
            with open(info_filename, 'a') as f:
                info_string = f"Task {task} took {end_time-start_time:.4f}sec ({n_iter} iterations) with lml={jnp.sum(lml):.4f}\n"
                f.write(info_string)

        # Add an empty line between each subject in the info file
        with open(info_filename, 'a') as f:
            f.write('\n')

    ## Save the new seed
    with open(seed_file, 'w') as f:
        f.write(str(key[1]))
    if do_print:
        print('Done')