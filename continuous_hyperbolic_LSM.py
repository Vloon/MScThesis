# Home brew functions
from helper_functions import *
from bookstein_methods import *

# Basics
import pickle
import time
import os
import csv

from plotting import plot_posterior, plot_network
import matplotlib.pyplot as plt

# Sampling
import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import blackjax as bjx
import blackjax.smc.resampling as resampling
print('Using blackjax version',bjx.__version__)
# Typing
from jax._src.prng import PRNGKeyArray
from jax._src.typing import ArrayLike
from blackjax.types import PyTree
from blackjax.mcmc.rmh import RMHState
from blackjax.smc.tempered import TemperedSMCState
from blackjax.smc.base import SMCInfo
from typing import Callable, Tuple
from pprint import pprint

# Keep this here in case you somehow import the file and need these constants??
mu = [0., 0.]
eps = 1e-6
sigma = 1.
mu_sigma = 0.
sigma_sigma = 1.
N = 164
D = 2
s1 = 1
sn = 25
n_particles = 1_000
n_mcmc_steps = 100
rmh_sigma = 1e-2
data_filename = 'processed_data.pkl'
task_filename = 'task_list.txt'
base_output_folder = 'Embeddings'
make_plot = False
figure_folder = 'Figures'
label_location = 'Figures/prettylabels.npz'
do_print = False
save_stats = False
stats_filename = 'statistics.csv'
stats_folder = 'Statistics'
dl = ';'
key = 0
gpu = ''

if __name__ == "__main__":
    # Create cmd argument list (arg_name, var_name, type, default, nargs[OPT])
    arguments = [('-e', 'eps', float, eps),  # d->d_max offset
                 ('-m', 'mu', float, mu, '+'),  # mean of distribution to sample z
                 ('-s', 'sigma', float, sigma),  # std of distribution to sample z
                 ('-ms', 'mu_sigma', float, mu_sigma),  # mean of distribution to sample sigma_T
                 ('-ss', 'sigma_sigma', float, sigma_sigma),  # std of distribution to sample sigma_T
                 ('-N', 'N', int, N),  # number of nodes
                 ('-D', 'D', int, D),  # latent space dimensions
                 ('-s1', 'subject1', int, s1),  # first subject to be used
                 ('-sn', 'subjectn', int, sn),  # last subject to be used
                 ('-np', 'n_particles', int, n_particles),  # number of smc particles
                 ('-nm', 'n_mcmc_steps', int, n_mcmc_steps),  # number of mcmc steps within smc
                 ('-r', 'rmh_sigma', float, rmh_sigma), # sigma of the RMH sampler within SMC
                 ('-df', 'data_filename', str, data_filename),  # filename of the saved data
                 ('-tf', 'task_filename', str, task_filename), # filename of the list of task names
                 ('-of', 'base_output_folder', str, base_output_folder), # folder where to dump the LSM embeddings
                 ('--plot', 'make_plot', bool), # whether to create a plot
                 ('-ff', 'figure_folder', str, figure_folder), # base folder where to dump the figures
                 ('-lab', 'label_location', str, label_location), # file location of the labels
                 ('--print', 'do_print', bool), # whether to print cute info
                 ('--stats', 'save_stats', bool), # whether to save the statistics in a csv
                 ('-stf', 'stats_filename', str, stats_filename),  # statistics filename
                 ('-stfl', 'stats_folder', str, stats_folder),  # statistics folder
                 ('-dl', 'dl', str, dl),  # save stats delimeter
                 ('-key', 'key', int, key), # starting random key
                 ('-gpu', 'gpu', str, gpu), # number of gpu to use (in string form). If no GPU is specified, CPU is used.
                 ]

    # Get arguments from CMD
    global_params = get_cmd_params(arguments)
    eps = global_params['eps']
    mu = global_params['mu']
    sigma = global_params['sigma']
    mu_sigma = global_params['mu_sigma']
    sigma_sigma = global_params['sigma_sigma']
    N = global_params['N']
    M = N*(N-1)//2
    D = global_params['D']
    subject1 = global_params['subject1']
    subjectn = global_params['subjectn']
    n_particles = global_params['n_particles']
    n_mcmc_steps = global_params['n_mcmc_steps']
    rmh_sigma = global_params['rmh_sigma']
    data_filename = global_params['data_filename']
    task_filename = global_params['task_filename']
    output_folder = global_params['base_output_folder']+f'/{n_particles}p{n_mcmc_steps}s'
    make_plot = global_params['make_plot']
    figure_folder = global_params['figure_folder']
    label_location = global_params['label_location']
    do_print = global_params['do_print']
    save_stats = global_params['save_stats']
    stats_filename = f"{global_params['stats_folder']}/{global_params['stats_filename']}"
    dl = global_params['dl']
    key = global_params['key']
    gpu = global_params['gpu']
    if gpu is None: # <-- if visible cuda in os.environ is set to None, ALL GPUs will be used. Empty string means CPU is used.
        gpu = ''

    # Set before using any JAX functions to avoid issues with seeing GPUs!
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Initialize JAX stuff
    if do_print:
        print(f'Running on {jax.devices()}')
    key = jax.random.PRNGKey(key)

    # Create output folder if it does not yet exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ONLY NOW TURN MU INTO JNP ARRAY, OTHERWISE JAX WILL F*** WITH THE GPUs THE WRONG WAY!!
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
    PARAMS:
    _z : 2D Gaussian, to be projected onto the hyperbolic plane
    mu : mean of the wrapped hyperbolic normal prior
    eps : offset for calculating d_norm, to insure max(d_norm) < 1
    """
    N, D = _z.shape
    triu_indices = jnp.triu_indices(N, k=1)

    mu_0 = jnp.zeros((N, D+1))
    mu_0 = mu_0.at[:,0].set(1)
    
    # Mu can be a value (e.g. 0) to indicate (0,0), or a list (e.g. [0,0])
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
    d_norm = jnp.clip(d/(jnp.max(d)), eps, 1-eps) # Clip to make sure the boundaries are excluding 0, 1.
    mu_beta = 1-d_norm
    bound = jnp.sqrt(mu_beta*(1-mu_beta))

    params = {'_z':_z,
              'z':z,
              'd':d,
              'd_norm':d_norm,
              'mu_beta':mu_beta,
              'bound':bound}
    return params

def get_ab(mu_beta:ArrayLike, sigma_beta:float, eps:float=eps) -> tuple:
    """
    Calculates the a and b parameters for the beta distribution
    PARAMS:
    mu_beta : mean of the beta distribution
    sigma_beta : standard deviations of the beta distribution
    """
    _kappa = mu_beta*(1-mu_beta)/(jnp.maximum(sigma_beta**2,eps)) - 1
    kappa = jnp.clip(_kappa, eps, 1-eps)
    a = mu_beta*kappa
    b = (1-mu_beta)*kappa
    return a, b

## Sampling functions
def sample_prior(key:PRNGKeyArray, shape:tuple, sigma:float=sigma, mu_sigma:float=mu_sigma, sigma_sigma:float=sigma_sigma) -> dict:
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

def sample_observation(key:PRNGKeyArray, prior:dict, n_samples:int=1, eps:float=eps) -> dict:
    """
    Generates an observation based on the prior
    PARAMS:
    key : random key for JAX functions
    prior : dictionary containing sampled variables from the prior ('_z', 'sigma_beta_T')
    n_samples : number of observations to sample
    eps : offset for the normalization of d, and clipping of A
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

    # Sample A, and clip between eps and 1-eps
    key, subkey = jax.random.split(key)
    A = jnp.clip(jax.random.beta(subkey, a, b, shape=(n_samples, M,)), eps, 1-eps)
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
    return rest_logprior+pivot_point_logprior

def log_likelihood(_z:ArrayLike, sigma_beta_T:float, obs:ArrayLike, eps:float=eps, idx=-1) -> float:
    """
    Returns the log-likelihood
    PARAMS:
    _z : x,y coordinates of the positions
    sigma_beta_T : transformed standard deviations of the beta distribution
    obs : observed correlations (samples x edges)
    eps : offset for calculating d_norm, to insure max(d_norm) < 1
    """
    params = get_det_params(_z, eps=eps)
    mu_beta, bound = params['mu_beta'], params['bound']
    sigma_beta = invlogit(sigma_beta_T)*bound # Transform sigma_beta back to [0, bound] to get a,b
    a,b = get_ab(mu_beta, sigma_beta)
    logprob_A = jstats.beta.logpdf(obs, a, b)
    return logprob_A.sum()

def bookstein_log_likelihood(_z:ArrayLike, sigma_beta_T:float, obs:ArrayLike, eps:float=eps, idx=-1) -> float:
    """
    Returns the log-likelihood
    PARAMS:
    _z : x,y coordinates of the positions
    sigma_beta_T : transformed standard deviations of the beta distribution
    obs : observed correlations (samples x edges)
    eps : offset for calculating d_norm, to insure max(d_norm) < 1
    """
    n_dims = _z.shape[1]
    bookstein_target = get_bookstein_target(n_dims)

    # Concatenate bookstein targets to _z
    _zc = jnp.concatenate([bookstein_target, _z])
    return log_likelihood(_zc, sigma_beta_T, obs, idx=idx)

def log_density(_z:ArrayLike, sigma_beta_T:float, obs:ArrayLike, mu:ArrayLike=mu, sigma:float=sigma, eps:float=eps) -> float:
    """
    Returns the log-probability density of the observed edge weights under the Continuous Hyperbolic LSM.
    PARAMS:
    _z : positions on Euclidean plane (pre hyperbolic projection)
    sigma_beta_T : transformed standard deviations of the beta distributions
    obs : observed correlations (samples x edges)
    mu : mean of the 2D Gaussian that is projected to the hyperbolic plane
    sigma : standard deviation of the 2D Gaussian that is projected to the hyperbolic plane
    eps : offset for calculating d_norm, to insure max(d_norm) < 1
    """
    prior_prob = log_prior(_z, sigma_beta_T, mu, sigma)
    likelihood_prob = log_likelihood(_z, sigma_beta_T, obs) # (samples,)
    return prior_prob + likelihood_prob

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
    _z_key, sigma_beta_T_key, key = jax.random.split(key, 3)
    initial_position = {'_z': smc_bookstein_position(sigma*jax.random.normal(_z_key, shape=(num_particles, N-D, D))), # N-D to skip first D bkst nodes (they are implicit)
                        'sigma_beta_T': sigma_sigma*jax.random.normal(sigma_beta_T_key, shape=(num_particles, 1))+mu_sigma}
    return key, initial_position

def smc_bkst_inference_loop(key:PRNGKeyArray, smc_kernel:Callable, initial_state:ArrayLike, likelihood_func) -> Tuple[PRNGKeyArray, float, TemperedSMCState]:
    """
    Run the temepered SMC algorithm with Bookstein anchoring.

    Run the adaptive algorithm until the tempering parameter lambda reaches the value lambda=1.
    PARAMS:
    key: random key for JAX functions
    smc_kernel: kernel for the SMC particles
    initial_state: beginning position of the algorithm
    """
    def cond(carry):
        state = carry[7]
        return state.lmbda < 1

    @jax.jit
    def step(carry):
        i, lml, lambdas, weights, ancestors, _z_proposals, s_proposals, state, key = carry
        key, subkey = jax.random.split(key)
        state, info = smc_kernel(subkey, state)
        _z_proposals = _z_proposals.at[i].set(info.proposals['_z'])
        s_proposals = s_proposals.at[i].set(info.proposals['sigma_beta_T'])
        ancestors = ancestors.at[i].set(info.ancestors)
        weights = weights.at[i].set(info.weights)
        lambdas = lambdas.at[i].set(state.lmbda)
        lml += info.log_likelihood_increment
        return i+1, lml, lambdas, weights, ancestors, _z_proposals, s_proposals, state, key

    max_iters = 200
    n_iter, lml, lambdas, weights, ancestors, _z_proposals, s_proposals, final_state, key = jax.lax.while_loop(
        cond, step, (0, 0., jnp.zeros((max_iters,)), jnp.zeros((max_iters, n_particles)), jnp.zeros((max_iters, n_particles)), jnp.zeros((max_iters, n_particles, N-D, D)), jnp.zeros((max_iters, n_particles, 1)), initial_state, key)
    )

    return key, n_iter, lml, lambdas, weights, ancestors, _z_proposals, s_proposals, final_state

def get_LSM_embedding(key:PRNGKeyArray, obs:ArrayLike, N:int=N, D:int=D, rmh_sigma:float=rmh_sigma, n_mcmc_steps:int=n_mcmc_steps, n_particles:int=n_particles) -> Tuple[PRNGKeyArray, float, TemperedSMCState]:
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
    key, n_iter, lml, lambdas, weights, ancestors, _z_proposals, s_proposals, states_rwm_smc = smc_bkst_inference_loop(key, smc.step, initial_smc_state, _bookstein_log_likelihood)

    # Add Bookstein coordinates to SMC states
    states_rwm_smc = add_bkst_to_smc_trace(states_rwm_smc, D)

    return key, n_iter, lml, lambdas, weights, ancestors, _z_proposals, s_proposals, states_rwm_smc

if __name__ == "__main__":
    
    ###
    ### Data is in a dictionary. The keys are defined by "S{n_sub}_{task}_{enc}", e.g. "S1_EMOTION_RL".
    ### The values per key are the upper triangle of the correlation matrix (length M list).
    ### We go through each subject/task, and take both encodings as seperate observations to create 1 embedding.
    ###

    # Load plt labels here to avoid opening in a loop
    if make_plot:
        label_data = np.load(label_location)
        plt_labels = label_data[label_data.files[0]]

    obs, tasks = load_observations(data_filename, task_filename, subject1, subjectn, M) 

    for si, n_sub in enumerate(range(subject1, subjectn+1)):
        for ti, task in enumerate(tasks):
            # Create LS embedding
            start_time = time.time()
            key, n_iter, lml, lambdas, weights, ancestors, _z_proposals, s_proposals, smc_embedding = get_LSM_embedding(key, obs[si, ti, :, :])  # Other parameters to get_LSM_embeddings are taken from globals.
            end_time = time.time()

            if do_print:
                print(f'Embedded S{n_sub}_{task} in {n_iter} iterations')

            # Save the statistics to the csv file
            if save_stats:
                stats_row = [f'S{n_sub}', task, n_particles, n_mcmc_steps, lml, end_time - start_time]
                with open(stats_filename, 'a', newline='') as f:
                    writer = csv.writer(f, delimiter=dl)
                    writer.writerow(stats_row)

            if make_plot:
                last__z_positions = smc_embedding.particles['_z']
                z_positions = lorentz_to_poincare(get_attribute_from_trace(last__z_positions, get_det_params, 'z', shape=(n_particles, N, D+1)))

                ## ADD LABELS!
                # Plot posterior
                plt.figure()
                ax = plt.gca()
                plot_posterior(z_positions,
                               edges=obs[si, ti, 0],
                               pos_labels=plt_labels,
                               ax=ax,
                               hyperbolic=True,
                               bkst=True)
                poincare_disk = plt.Circle((0, 0), 1, color='k', fill=False, clip_on=False)
                ax.add_patch(poincare_disk)
                plt.title(f'Proposal S{n_sub} {task}')
                # Save figure
                fig_output_folder = f'{figure_folder}/{n_particles}p{n_mcmc_steps}s'
                if not os.path.exists(fig_output_folder): # Create folder if it does not yet exist
                    os.makedirs(fig_output_folder)
                savefile = f'./{fig_output_folder}/con_hyp_S{n_sub}_{task}.png'
                plt.savefig(savefile)
                plt.close()

            # Save data
            embedding_filename = f'./{output_folder}/con_hyp_S{n_sub}_{task}_embedding.pkl'
            info_filename = f'./{output_folder}/con_hyp_S{n_sub}.txt'
            with open(embedding_filename, 'wb') as f:
                pickle.dump(smc_embedding, f)
            with open(info_filename, 'a') as f:
                info_string = 'Task {} took {:.4f}sec ({} iterations) with lml={:.4f}\n'.format(task, end_time - start_time, n_iter, jnp.sum(lml))
                f.write(info_string)

        # Add an empty line between each subject in the info file
        with open(info_filename, 'a') as f:
            f.write('\n')
    if do_print:
        print('Done')