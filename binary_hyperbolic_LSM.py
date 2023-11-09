# Home brew functions
from helper_functions import set_GPU, get_cmd_params, get_filename_with_ext, get_safe_folder, load_observations, get_attribute_from_trace, lorentz_distance, \
    lorentz_to_poincare, hyp_pnt, lorentz_distance, parallel_transport, exponential_map, is_valid
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
eps = 1e-6
obs_eps = 1e-12
mu = [0., 0.]
sigma = 1.
N = 164
D = 2
s1 = 1
sn = 25
n_particles = 1_000
n_mcmc_steps = 100
rmh_sigma = 1e-2
bookstein_dist = 0.3
data_folder = 'Data'
base_data_filename = 'binary_data'
data_threshold = 0.50
task_filename = 'task_list'
base_output_folder = 'Embeddings'
make_plot = False
figure_folder = 'Figures'
label_location = 'Figures/lobelabels.npz'
r_margin = 0.1
do_print = False
save_stats = False
stats_filename = 'statistics'
stats_folder = 'Statistics'
dl = ';'
seed_file = 'seed.txt'
seed = None
gpu = ''

if __name__ == "__main__":
    # Create cmd argument list (arg_name, var_name, type, default, nargs[OPT])
    arguments = [('-e', 'eps', float, eps),  # d->d_max offset
                 ('-obseps', 'obs_eps', float, obs_eps),  # observation clipping offset
                 ('-m', 'mu', float, mu, '+'),  # mean of distribution to sample z
                 ('-s', 'sigma', float, sigma),  # std of distribution to sample z
                 ('-N', 'N', int, N),  # number of nodes
                 ('-D', 'D', int, D),  # latent space dimensions
                 ('-s1', 'subject1', int, s1),  # first subject to be used
                 ('-sn', 'subjectn', int, sn),  # last subject to be used
                 ('-np', 'n_particles', int, n_particles),  # number of smc particles
                 ('-nm', 'n_mcmc_steps', int, n_mcmc_steps),  # number of mcmc steps within smc
                 ('-r', 'rmh_sigma', float, rmh_sigma),  # sigma of the RMH sampler within SMC
                 ('-bdist', 'bookstein_dist', float, bookstein_dist),  # distance between the bookstein anchors
                 ('-overwritedf', 'overwrite_data_filename', str, None),  # if used, it overwrites the default filename
                 ('-datfol', 'data_folder', str, data_folder),  # folder where the data is stored
                 ('-df', 'base_data_filename', str, base_data_filename),  # filename of the saved data
                 ('-datath', 'data_threshold', float, data_threshold),  # threshold to be used for the HCP data
                 ('-tf', 'task_filename', str, task_filename),  # filename of the list of task names
                 ('-of', 'base_output_folder', str, base_output_folder),  # folder where to dump the LSM embeddings
                 ('--plot', 'make_plot', bool),  # whether to create a plot
                 ('-ff', 'figure_folder', str, figure_folder),  # base folder where to dump the figures
                 ('-lab', 'label_location', str, label_location),  # file location of the labels
                 ('-rmarg', 'r_margin', float, r_margin), # offset for the radius of the disk drawn around the positions
                 ('--print', 'do_print', bool),  # whether to print cute info
                 ('--stats', 'save_stats', bool),  # whether to save the statistics in a csv
                 ('--partial', 'partial', bool),  # whether to use partial correlations
                 ('--nolabels', 'no_labels', bool), # whether to not use labels
                 ('--bpf', 'bpf', bool),  # whether to use band-pass filtered rs-fMRI data
                 ('-stf', 'stats_filename', str, stats_filename),  # statistics filename
                 ('-stfl', 'stats_folder', str, stats_folder),  # statistics folder
                 ('-dl', 'dl', str, dl),  # save stats delimeter
                 ('-seedfile', 'seed_file', str, seed_file),  # save file for the seed
                 ('-seed', 'seed', int, seed),  # starting random key
                 ('-gpu', 'gpu', str, gpu), # number of gpu to use (in string form). If no GPU is specified, CPU is used.
                 ]

    # Get arguments from CMD
    global_params = get_cmd_params(arguments)
    eps = global_params['eps']
    obs_eps = global_params['obs_eps']
    mu = global_params['mu']
    sigma = global_params['sigma']
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
    data_threshold = global_params['data_threshold']
    output_folder = get_safe_folder(f"{global_params['base_output_folder']}/{n_particles}p{n_mcmc_steps}s")
    make_plot = global_params['make_plot']
    figure_folder = get_safe_folder(f"{global_params['figure_folder']}/{n_particles}p{n_mcmc_steps}s")
    label_location = global_params['label_location']
    r_margin = global_params['r_margin']
    do_print = global_params['do_print']
    save_stats = global_params['save_stats']
    partial = global_params['partial']
    no_labels = global_params['no_labels']
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
    if seed is None:
        with open(seed_file, 'r') as f:
            seed = int(f.read())
    key = jax.random.PRNGKey(seed)

    # ONLY NOW TURN MU INTO JNP ARRAY, OTHERWISE JAX WILL F*** WITH THE GPUs THE WRONG WAY!!
    mu = jnp.array(mu)

## Define functions to calculate the continuous hyperbolic parameters
def get_det_params(_z:ArrayLike, mu:ArrayLike=mu, eps:float=eps, **kwargs) -> dict:
    """
    Calculates all parameters based on z, and returns those in a dictionary.
    PARAMS:
    _z : pre-hyperbolic projection latent positions
    mu : mean of the wrapped hyperbolic normal prior, pre-hypebolic projection
    eps : offset for calculating d_norm, to insure max(d_norm) < 1
    **kwargs allows us to pass non-used parameters, which is handy when we want to allow other files to use the con hyp model but also others and allow this function to just catch the parameters it needs
    """
    N, D = _z.shape
    triu_indices = jnp.triu_indices(N, k=1)

    mu_0 = jnp.zeros((N, D + 1))
    mu_0 = mu_0.at[:, 0].set(1)

    # Mu can be a value (e.g. 0) to indicate (0,0), or array-like (e.g. [0,0])
    if hasattr(mu, '__len__'):
        mu = jnp.array(mu)
        assert len(mu) == D, f"Dimension of mu (={len(mu)}) must correspond to the dimension of each point in _z (={D})"
        mu_tilde = jnp.reshape(jnp.tile(mu, N), (N, D))
    else:
        mu_tilde = mu * jnp.ones_like(_z)

    mu = hyp_pnt(mu_tilde)  # UGLY?! How else to get a proper mean in H? Just kinda.. guess a correct H coordinate?
    v = jnp.concatenate([jnp.zeros((N, 1)), _z], axis=1)
    u = parallel_transport(v, mu_0, mu)
    z = exponential_map(mu, u)

    d = lorentz_distance(z)[triu_indices]

    p = jnp.clip(jnp.exp(-d**2), eps, 1-eps)

    params = {'_z':_z,
              'z': z,
              'd':d,
              'p':p}
    return params

## Sampling functions
def sample_prior(key:PRNGKeyArray, shape:tuple, sigma:float = sigma, eps:float = eps, **kwargs) -> Tuple[PRNGKeyArray, dict]:
    """
    Samples a prior by taking a 2D normal distribution and solving for z on the hyperbolic plane.
    Returns the prior parameters in a dictionary.
    PARAMS:
    key : Random key for JAX functions
    shape : shape of the prior positions
    sigma : standard deviation of the 2D Gaussian to sample p
    eps : offset for calculating d_norm, to insure max(d_norm) < 1
    """
    mu = jnp.array(mu)
    N = shape[0]
    M = N * (N - 1) // 2

    # Sample positions and temp
    key, _z_key, = jax.random.split(key)
    prior = {'_z' : sigma * jax.random.normal(_z_key, shape=shape)} # Is always centered at 0

    return key, prior

def sample_observation(key:PRNGKeyArray, prior:dict, n_samples:int=1, eps:float=eps, **kwargs) -> Tuple[PRNGKeyArray, jnp.array]:
    """ 
    Generates an observation based on the prior
    PARAMS:
    key : Random key for JAX functions
    prior : dictionary containing sampled variables from the prior ('_z')
    n_samples : number of observations to sample
    eps : offset for the normalization of d, and clipping of A
    """
    # Get prior position and sigma
    _z = prior['_z']
    N = _z.shape[0]
    M = N*(N-1)//2

    # Calculate p
    params = get_det_params(_z, eps=eps)
    p = params['p']

    # Sample A
    key, subkey = jax.random.split(key)
    Y = jax.random.bernoulli(subkey, p, shape=(n_samples, M))

    return key, Y

## Probability distributions
def log_prior(_z:ArrayLike, sigma:float=sigma) -> float:
    """
    Returns the log-prior for a full _z state and sigma state, no Bookstein faffing.
    PARAMS:
    _z : pre-hyperbolic transformed positions
    sigma : standard deviation of the 2D Gaussian that is projected to the hyperbolic plane
    """
    logprob__z = jstats.norm.logpdf(_z, loc=0, scale=sigma).sum()
    return logprob__z 

def bookstein_log_prior(_z:ArrayLike, _zb_x:float, sigma:float=sigma) -> float:
    """
    Returns the log-prior, taking into account Bookstein anchors.
    PARAMS:
    _z : pre-hyperbolic transformed positions
    _zb_x : x-coordinate of the 2nd Bookstein anchor. Its y-coordinate is always 0.
    sigma : standard deviation of the 2D Gaussian that is projected to the hyperbolic plane
    """
    _zb_x_logprior = jstats.truncnorm.logpdf(_zb_x, a=0, b=jnp.inf, loc=-bookstein_dist, scale=sigma).sum() # Logprior for the node restricted in y = 0.
    _zb_y_logprior = jnp.log(2)+jstats.norm.logpdf(_z[0,:], loc=0, scale=sigma).sum() - jnp.inf*(_z[0,1] < 0) # Logprior for the node restricted in y>0 # Kan dit niet ook met een truncnorm?
    rest_logprior = log_prior(_z[1:,:], sigma) 
    return rest_logprior + _zb_x_logprior + _zb_y_logprior

def log_likelihood(_z:ArrayLike, obs:ArrayLike, eps:float=eps) -> float:
    """
    Returns the log-likelihood
    PARAMS:
    _z : pre-hyperbolic transformed positions
    obs : observed correlations (samples x edges)
    eps : offset for the normalization of d, and clipping of A
    """
    params = get_det_params(_z, eps=eps)
    p = params['p']

    logprob_Y = jstats.bernoulli.logpmf(obs, p).sum() # S x M obs --sum--> scalar logprob_A
    return logprob_Y

def log_likelihood_from_dist(d:ArrayLike, obs:ArrayLike, eps:float=eps) -> float:
    """
    Returns the log-likelihood based on the distance rather than positions
    PARAMS:
    d : (M) upper triangle of the distance matrix
    obs : (n_obs x M) observed correlations
    eps : offset for clipping
    """
    p = jnp.clip(jnp.exp(-d**2), eps, 1-eps)
    logprob_Y = jstats.bernoulli.logpmf(obs, p).sum()
    return logprob_Y

def bookstein_log_likelihood(_z:ArrayLike, _zb_x:float, obs:ArrayLike, bookstein_dist:float=bookstein_dist, eps:float=eps) -> float:
    """
    Returns the log-likelihood
    PARAMS:
    _z : x,y coordinates of the positions, without bookstein anchors
    _zb_x : x-coordinate of the 2nd Bookstein anchor. Its y-coordinate is always 0.
    obs : observed correlations (samples x edges)
    bookstein_dist : x-offset of the first Bookstein anchor
    eps : offset for the normalization of d, and clipping of A
    """
    n_dims = _z.shape[1]
    bookstein_anchors = get_bookstein_anchors(_zb_x, n_dims, bookstein_dist)

    # Concatenate bookstein anchors to _z
    _zc = jnp.concatenate([bookstein_anchors, _z])
    return log_likelihood(_zc, obs)

def log_density(_z:ArrayLike, obs:ArrayLike, mu:float=mu, sigma:float=sigma, eps:float=eps) -> float:
    """
    Returns the log-probability density of the observed edge weights under the Continuous Hyperbolic LSM.
    PARAMS:
    _z : positions on Euclidean plane (pre hyperbolic projection)
    obs : observed correlations (samples x edges)
    mu : mean of the 2D Gaussian to sample _z
    sigma : standard deviation of the 2D Gaussian to sample _z
    eps : offset for the normalization of d, and clipping of A
    """
    prior_prob = log_prior(_z, sigma)
    likelihood_prob = log_likelihood(_z, obs, eps=eps)
    return prior_prob + likelihood_prob

## SMC + Bookstein methods
def initialize_bkst_particles(key:PRNGKeyArray, num_particles:int, shape:tuple, sigma:float=sigma, bookstein_dist:float=bookstein_dist) -> Tuple[PRNGKeyArray, dict]:
    """
    Initializes the SMC particles, but with Bookstein coordinates.
    PARAMS:
    key : random key for JAX functions
    num_particles : number of SMC particles
    shape : number of nodes by number of dimensions
    sigma : std of the positions' 2D Gaussian
    bookstein_dist : offset of the first Bookstein anchor
    """
    N, D = shape
    key, _z_key, _z_bx_key = jax.random.split(key, 3)
    initial_position = {'_z': smc_bookstein_position(sigma*jax.random.normal(_z_key, shape=(num_particles, N-D, D))), # N-D to skip first D bkst nodes. First node is rigid.
                        '_zb_x': sigma*jax.random.truncated_normal(_z_bx_key, lower=-bookstein_dist, upper=jnp.inf, shape=(num_particles,1))} # Second node is restricted to just an x-position.
    return key, initial_position

def smc_bkst_inference_loop(key:PRNGKeyArray, smc_kernel:Callable, initial_state:ArrayLike) -> Tuple[PRNGKeyArray, int, float, TemperedSMCState]:
    """
    Run the temepered SMC algorithm with Bookstein anchoring.

    Run the adaptive algorithm until the tempering parameter lambda reaches the value lambda=1.
    PARAMS:
    key: random key for JAX functions
    smc_kernel: kernel for the SMC particles
    initial_state: beginning position of the algorithm
    """
    def cond(carry):
        state = carry[2]
        return state.lmbda < 1

    @jax.jit
    def step(carry):
        i, lml, state, key = carry
        key, subkey = jax.random.split(key)
        state, info = smc_kernel(subkey, state)
        lml += info.log_likelihood_increment
        return i+1, lml, state, key

    n_iter, lml, final_state, key = jax.lax.while_loop(
        cond, step, (0, 0., initial_state, key)
    )

    return key, n_iter, lml, final_state

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

    rmh_parameters = {'sigma': rmh_sigma * jnp.eye((N - D) * D + 1 )} # + 1 for _z_bx
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
    key, n_iter, lml, states_rwm_smc = smc_bkst_inference_loop(key, smc.step, initial_smc_state)

    # Add Bookstein coordinates to SMC states
    states_rwm_smc = add_bkst_to_smc_trace(states_rwm_smc, bookstein_dist, D=D)

    return key, n_iter, lml, states_rwm_smc


if __name__ == "__main__":
    """
    Data is in a dictionary. The keys are defined by "S{n_sub}_{task}_{enc}", e.g. "S1_EMOTION_RL".
    The values per key are the upper triangle of the correlation matrix (length M list).
    We go through each subject/task, and take both encodings as seperate observations to create 1 embedding.
    """

    # Load labels
    if make_plot and not no_labels:
        label_data = np.load(label_location)
        plt_labels = label_data[label_data.files[0]]
        if len(plt_labels) is not N:
            plt_labels = None
    else:
        plt_labels = None

    # Load data
    if not overwrite_data_filename:
        data_filename = get_filename_with_ext(f"{base_data_filename}_th{data_threshold:.2f}", partial, bpf,
                                              folder=data_folder)
    else:
        data_filename = overwrite_data_filename
    task_filename = get_filename_with_ext(task_filename, ext='txt', folder=data_folder)
    obs, tasks, encs = load_observations(data_filename, task_filename, subject1, subjectn, M)

    for si, n_sub in enumerate(range(subject1, subjectn + 1)):
        for ti, task in enumerate(tasks):
            # Create LS embedding
            start_time = time.time()
            key, n_iter, lml, smc_embedding = get_LSM_embedding(key, obs[si, ti, :,:])  # Other parameters to get_LSM_embeddings are taken from globals.
            end_time = time.time()

            if do_print:
                print(f'Embedded S{n_sub}_{task} in {n_iter} iterations')

            # Save the statistics to the csv file
            if save_stats:
                stats_row = [f'S{n_sub}', task, n_particles, n_mcmc_steps, lml, end_time - start_time]
                with open(stats_filename, 'a', newline='') as f:
                    writer = csv.writer(f, delimiter=dl)
                    writer.writerow(stats_row)

            partial_txt = '_partial' if partial else ''
            base_save_filename = f"bin_hyp_S{n_sub}_{task}_embedding_th{data_threshold:.2f}{partial_txt}"

            if make_plot:
                _z_positions = smc_embedding.particles['_z']
                z_positions = lorentz_to_poincare(get_attribute_from_trace(_z_positions, get_det_params, 'z', shape=(n_particles, N, D + 1)))

                ## TODO: ADD LABELS!
                # Plot posterior
                plt.figure()
                ax = plt.gca()
                plot_posterior(z_positions,
                               edges=obs[si, ti, 0],  # Dit is ook nog raar!
                               pos_labels=plt_labels,
                               ax=ax,
                               title=f"Posterior S{n_sub} {task}",
                               hyperbolic=True,
                               bkst=True)
                poincare_disk = plt.Circle((0, 0), 1, color='k', fill=False, clip_on=False)
                ax.add_patch(poincare_disk)
                # Save figure
                savefile = get_filename_with_ext(base_save_filename, ext='png', folder=figure_folder)
                plt.savefig(savefile)
                plt.close()

            # Save data
            embedding_filename = get_filename_with_ext(base_save_filename, partial=partial, bpf=bpf, folder=output_folder)
            info_filename = get_filename_with_ext(f"bin_hyp_S{n_sub}", ext='txt', folder=output_folder)
            with open(embedding_filename, 'wb') as f:
                pickle.dump(smc_embedding, f)
            with open(info_filename, 'a') as f:
                info_string = f'Task {task} at threshold {data_threshold} took {end_time - start_time:.4f}sec ({n_iter} iterations) with lml={jnp.sum(lml):.4f}\n'
                f.write(info_string)

        # Add an empty line between each subject in the info file
        with open(info_filename, 'a') as f:
            f.write('\n')

    ## Save the new seed
    with open(seed_file, 'w') as f:
        f.write(str(key[1]))
    if do_print:
        print('Done')
