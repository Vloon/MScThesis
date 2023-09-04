# Basics
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Ellipse
import matplotlib.colors as mcolors
import numpy as np
import time
import argparse 

# File stuff
import pickle

# Sampling
import jax
import jax.numpy as jnp
from jax.config import config
import jax.scipy as jsp
import jax.scipy.stats as jstats
# config.update("jax_enable_x64", True)

# Typings
from jax._src.prng import PRNGKeyArray
from jax._src.typing import ArrayLike
from blackjax.types import PyTree
from blackjax.mcmc.rmh import RMHState
from matplotlib import axes as Axes # I want types to be capitalized for some reason.
from typing import Callable, Tuple

## Parser
def set_GPU(gpu:str = '') -> None:
    """
    Sets the GPU safely in os.environ
    PARAMS:
    gpu : string format of the GPU used. Multiple GPUs can be seperated with commas, e.g. '0,1,2'
    """
    if gpu is None: # Safety, if visible divises is set to none in os, then all GPUs are used
        gpu = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

def get_cmd_params(parameter_list:list) -> dict:
    """
    Gets the parameters described in parameter_list from the command line.
    PARAMS:
    parameter_list : list of tuples containing (arg_name <str>, dest <str>, type <type>, default <any>, nargs <str> [OPTIONAL])
    default_parameters : the default parameters when not all parameters are given in the command line

    Example of parameter list:
        [('-m', 'mu', float, [1.,0.] '+'),
         ('-s', 'sigma', float, 1.)]
    """
    # Create parser
    parser = argparse.ArgumentParser()
    # Add parameters to parser
    for parameter in parameter_list:
        assert len(parameter) in [3, 4, 5], f'Parameter tuple must be length 3 (bool only), 4 or 5 but is length {len(parameter)}.'
        if len(parameter) == 3:
            arg_name, dest, arg_type = parameter
            assert arg_type == bool, f'Three parameters were passed, so arg_type should be bool but is {arg_type}'
            nargs = '?'
        elif len(parameter) == 4:
            arg_name, dest, arg_type, default = parameter
            nargs = '?' # If no nargs is given we default to single value
        elif len(parameter) == 5:
            arg_name, dest, arg_type, default, nargs = parameter
        if arg_type != bool:
            parser.add_argument(arg_name, dest=dest, nargs=nargs, type=arg_type, default=default)
        else:
            parser.add_argument(arg_name, dest=dest, action='store_true', default=False)
    # Parse arguments from CMD
    args = parser.parse_args()
    # Create global parameters dictionary
    global_params = {arg:getattr(args,arg) for arg in vars(args)}
    return global_params

## Data stuff
def is_valid(x:ArrayLike) -> bool:
    """
    Checks whether all values in an array are valid, and returns the bad indices
    PARAMS:
    x : input array
    """
    return np.all(np.isfinite(x)), np.where(~np.isfinite(x))

def load_observations(data_filename:str, task_filename:str, subject1:int, subjectn:int, M:int) -> Tuple[jnp.ndarray, list]:
    """
    Loads the observations from the filename into a jax.numpy array, seperated by task found in task_filename. Takes both encodings as seperate observations of the same stimulus. 
    PARAMS:
    data_filename : filename of the observed correlations data
    task_filename : name of the file containing a list of tasks plus a list of encodings (=observations per subject per task)
    subject1 : first subject
    subjectn : last subject
    M : number of edges
    """
    # Open the data file
    with open(data_filename, 'rb') as f:
        obs_corr_dict = pickle.load(f)

    # Open task list file
    with open(task_filename) as tf:
        tasks = tf.readline().rstrip('\n').split(',')
        encs = tf.readline().rstrip('\n').split(',')

    # Get observations for each subject for each task
    n_subjects = subjectn+1-subject1
    n_tasks = len(tasks)
    obs = jnp.zeros((n_subjects, n_tasks, len(encs), M))

    for si, n_sub in enumerate(range(subject1, subjectn+1)): # Doesn't have to be the same, you can go from subject 3 to subject 6
        for ti, task in enumerate(tasks):
            for n_enc in range(len(encs)):
                enc = encs[n_enc]
                dict_key = f'S{n_sub}_{task}_{enc}'
                obs = obs.at[si, ti, n_enc, :].set(obs_corr_dict[dict_key])
    return obs, tasks

def node_pos_dict2array(pos_dict:dict) -> np.ndarray:
    """
    Puts the dictionary latent positions {node: position} into an (n,D) array
    PARAMS:
    pos_dict : dictionary containing node integers as keys and positions (length D) as value
    """
    n = len(pos_dict)
    D = len(pos_dict[0])
    pos_array = np.zeros((n, D))
    for i in range(n):
        pos_array[i, :] = pos_dict[i]
    return pos_array

def triu2mat(v:ArrayLike) -> jnp.ndarray:
    """
    Fills a matrix from upper triangle vector
    PARAMS:
    v (ArrayLike) : upper triangle vector
    """
    m = len(v)
    n = int((1 + np.sqrt(1 + 8 * m))/2)
    mat = np.zeros((n, n))
    triu_indices = np.triu_indices(n, k=1)
    mat[triu_indices] = v
    return mat + mat.T

def get_trace_correlation(sampled_d:ArrayLike, ground_truth_d:ArrayLike) -> jnp.ndarray:
    """
    Gets the correlations between the distances in the sampled positions and the ground truth.
    PARAMS:
    sampled_d : the sampled distance matrices
    ground_truth_d : the ground truth distance matrix
    """
    N = sampled_d.shape[1]
    triu_indices = np.triu_indices(N, k=1)
    gt_d_triu = ground_truth_d[triu_indices]
    get_corr = lambda carry, d: (None, jnp.corrcoef(gt_d_triu, d[triu_indices])[0,1])

    _, corrs = jax.lax.scan(get_corr, None, sampled_d)
    return corrs

def get_attribute_from_trace(LSM_embeddings:ArrayLike, get_det_params:Callable, attribute:str='d_norm', shape:tuple=None, param_kwargs:dict={}) -> jnp.ndarray:
    """
    Calculates the distance for a whole trace of positions
    PARAMS:
    LSM_embeddings : (T, n_particles, N, D) or (n_particles, N, D) _z positions for all traces of all embeddings
    get_det_params : function to get the deterministic parameters according to the correct model
    attributes : dictionary key corresponding to the desired attribute
    shape : shape of the output, T x n_particles x "shape of attribute for 1 particle"
    param_kwargs : parameters for the get_det_params function (e.g. 'mu' = [0.,0.])
    """
    if len(LSM_embeddings.shape) == 4:
        T, n_particles, N, _ = LSM_embeddings.shape
    elif len(LSM_embeddings.shape) == 3:
        n_particles, N, _ = LSM_embeddings.shape
        T = 1
        LSM_embeddings = jnp.array([LSM_embeddings])
    else:
        print(f'Illegal format for LSM_embeddings: {LSM_embeddings.shape}')

    if shape is None:
        M = N * (N - 1) // 2
        shape = (T, n_particles, M)
    elif T == 1:  # implicit if shape
        shape = (T,) + shape

    def cond(carry):
        k, _, _ = carry
        return k < T

    @jax.jit
    def step(carry):
        k, i, attributes = carry
        attributes = attributes.at[k,i,:].set(get_det_params(LSM_embeddings[k,i,:,:], **param_kwargs)[attribute])
        i = (i+1)%n_particles
        k = k+ (i == 0)
        return k, i, attributes

    _, _, attributes = jax.lax.while_loop(
        cond, step, (0, 0, jnp.zeros(shape))
        )
    return attributes[0] if T == 1 else attributes

## Math
def invlogit(x:ArrayLike) -> ArrayLike:
    """
    Definition of the inverse-logit function (a.k.a. the logistic function)
    PARAMS:
    x : input variables
    """
    return 1 / (1 + jnp.exp(-x))

## Hyperbolic math
def lorentz_to_poincare(networks:ArrayLike) -> np.ndarray:
    """
    Convert Lorentz coordinates to Poincaré coordinates, eq 11 from Nickel & Kiela (2018).
    PARAMS:
    network (S,N,D) or (N,D): numpy array with Lorentzian coordinates [samples x nodes x dimensions] or [nodes x dimensions]
    """
    one_nw = len(networks.shape) == 2
    if one_nw: # We pass 1 network
        networks = np.array([networks])

    S, N, D_L = networks.shape
    calc_z_P = lambda c, nw: (None, nw[:,1:]/np.reshape(np.repeat(nw[:,0]+1, D_L-1), newshape=(N, D_L-1)))
    _, z_P = jax.lax.scan(calc_z_P, None, networks)

    return z_P[0,:,:] if one_nw else z_P

def hyp_pnt(X:ArrayLike) -> jnp.ndarray:
    """
    Create a 3D point [z,x,y] in hyperbolic space by projecting onto hyperbolic plane from 2D X=[x,y]
    PARAMS
    X : array containing 2D points to be projected up onto the hyperbolic plane
    """
    N, D = X.shape
    z = jnp.sqrt(jnp.sum(X**2, axis=1)+1)
    x_hyp = jnp.zeros((N,D+1))
    x_hyp = x_hyp.at[:,0].set(z)
    x_hyp = x_hyp.at[:,1:].set(X)
    return x_hyp

def lorentzian(v:ArrayLike, u:ArrayLike, keepdims:bool=False) -> jnp.ndarray:
    """
    Returns the Lorentzian prodcut of v and u, defined as -v_0*u_0 = SUM_{i=1}^N v_i*u_i
    PARAMS:
    v (N,D) : vector
    u (N,D) : vector
    """
    signs = jnp.ones_like(v)
    signs = signs.at[:,0].set(-1)
    return jnp.sum(v*u*signs, axis=1, keepdims=keepdims)

def lorentz_distance(z:ArrayLike) -> jnp.ndarray: #
    """
    Returns the hyperbolic distance between all N points in z as a N x N matrix
    PARAMS:
    z : points in hyperbolic space
    """
    def arccosh(x:ArrayLike) -> jnp.ndarray:
        """
        Definition of the arccosh function
        PARAMS:
        x : input
        """
        x_clip = jnp.maximum(x, jnp.ones_like(x, dtype=jnp.float32)) # Should leave sqrt(x_clip**2-1) = 0 at least
        return jnp.log(x_clip + jnp.sqrt(x_clip**2 - 1))
    signs = jnp.ones_like(z)
    signs = signs.at[:,0].set(-1)
    lor = jnp.dot(signs*z, z.T)
    # Due to numerical instability, we can get nan's on the diagonal, hence we force it to be zero
    dist = arccosh(-lor)
    dist = dist.at[jnp.diag_indices_from(dist)].set(0)
    return dist

def parallel_transport(v:ArrayLike, nu:ArrayLike, mu:ArrayLike) -> ArrayLike:
    """
    Parallel transports the points v sampled around nu to the tangent space of mu
    PARAMS:
    v  (N,D) : points on tangent space of nu [points on distribution around nu]
    nu (N,D) : point in hyperbolic space [center to move from] (mu_0 in Fig 2 of Nagano)
    mu (N,D) : point in hyperbolic space [center to move to]
    """
    alpha = -lorentzian(nu, mu, keepdims=True)
    u = v + lorentzian(mu - alpha*nu, v, keepdims=True)/(alpha+1) * (nu + mu)
    return u

def exponential_map(mu:ArrayLike, v:ArrayLike, eps:float=1e-6) -> ArrayLike:
    """
    Maps the points v on the tangent space of mu onto the hyperolic plane
    PARAMS:
    mu (N,D) : Transported middle points
    v (N,D) : Points to be mapped onto hyperbolic space
    eps : minimum value
    """
    # Euclidean norm from mu_0 to v is the same as from mu to u is the same as the Hyp-norm from mu to exp_mu(u), hence we can use the Euclidean norm of v.
    lor = lorentzian(v,v,keepdims=True)
    v_norm = jnp.sqrt(jnp.clip(lor, eps, lor))  ## If eps is too small, it gets rounded right back to zero and then we divide by zero
    return jnp.cosh(v_norm) * mu + jnp.sinh(v_norm) * v / v_norm

