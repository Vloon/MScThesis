## Basics
import jax
import jax.numpy as jnp

## Typing
from jax._src.prng import PRNGKeyArray
from jax._src.typing import ArrayLike
from blackjax.types import PyTree
from blackjax.smc.tempered import TemperedSMCState
from blackjax.mcmc.rmh import RMHState
from typing import Callable, Union, Tuple

def get_rigid_bookstein_anchors(n_dims:int = 2, offset:float = 0.3) -> jnp.ndarray:
    """
    Returns the Bookstein anchor coordinates for the first two positions, in n_dims dimensions.
    These Bookstein coordinates are rigid, in the sense that they are set rather than restricted, meaning the distance has to be relative.
    PARAMS:
    n_dims : number of dimensions of the latent space
    offset : offset on the x-axis that the Bookstein anchors are put. Node 1 is put on (-offset, 0), node 2 on (offset, 0).
    """
    assert n_dims > 0, f"Bookstein anchors must be used in a latent space with at least 1 dimension, but is being used in a {n_dims} dimensional LS"
    assert offset > 0, f"Offset must be bigger than 0 but is {offset}"
    bookstein_anchors = jnp.zeros(shape=(2, n_dims)) # For each dimension you need one bookstein coordinate, but only 2 can be rigid. Others must be restricted.
    bookstein_anchors = bookstein_anchors.at[:,0].set([-offset, offset])
    return bookstein_anchors

def get_bookstein_anchors(_zb_x:float, n_dims:int = 2, offset:float = 0.3) -> jnp.ndarray:
    """
    Returns the Bookstein anchor coordinates for the first positions, in n_dims dimensions.
    These Bookstein coordinates are restricted, meaning only 1 node is set, and the others are restricted and not dealt with in this function. 
    PARAMS:
    _zb_x : x position of the 2nd Bookstein anchor (restricted) of which the y-value must be 0.
    n_dims : number of dimensions of the latent space
    offset : offset on the x-axis that the Bookstein anchors are put. Node 1 is put on (-offset, 0).
    """
    assert n_dims > 0, f"Bookstein anchors must be used in a latent space with at least 1 dimension, but is being used in a {n_dims} dimensional LS"
    assert offset > 0, f"Offset must be bigger than 0 but is {offset}"
    bookstein_anchors = jnp.zeros(shape=(2, n_dims)) # Only 2 nodes are possible for now, didn't do the math yet for n_dims > 2.
    bookstein_anchors = bookstein_anchors.at[0,0].set(-offset) # First node's position is set rigidly.
    bookstein_anchors = bookstein_anchors.at[1,0].set(_zb_x[0]) # Second node's position is restricted to just a free x-coordinate, which is passed. Must unpack because a 1-dim array is not a float.
    return bookstein_anchors

def bookstein_position(pos:ArrayLike) -> ArrayLike:
    """
    Turns a set of positions into bookstein coordinates, where the first position is constrained. The two Bookstein anchors are implicit.
    PARAMS:
    pos : z or _z positions in the latent space.
    """
    n_nodes, n_dims = pos.shape
    ## Whether to flip this particle around the y-axis (in the right shape). This method is clunky for 1 particle.
    _do_flip = pos[0,1] < 0 # Is first (third when counting implicit anchors) _z below x-axis?
    do_flip = jnp.reshape(jnp.repeat(jnp.repeat(_do_flip, n_dims),n_nodes), (n_nodes, n_dims))

    ## The flip operator over the x-axis (i.e. in the y-direction).
    _x_flip = jnp.array([1,-1])
    x_flip = jnp.tile(_x_flip, (n_nodes,1))

    ## Then flip the positions over x-axis if we need to flip, or keep the positions the same if no flip.
    pos = pos*x_flip*do_flip + pos*(1-do_flip)
    return pos

def bookstein_init(bkst_init:RMHState, prior:dict, log_density:Callable, latpos:str='_z') -> PyTree:
    """
    Initializes the prior to start in Bookstein position.
    PARAMS:
    bkst_init : initialization function used of the RMH kernel.
    prior : prior dictionary containing positions _z.
    log_density : log density function to determine the probability of going to the proposed position.
    latpos : latent position variable name ('z' for Euclidean, '_z' for hyperbolic).
    """
    N,D = prior[latpos].shape
    assert N > D, f"Must have more than D positions to use Bookstein coordinates, but N={N} and D={D}"
    prior[latpos] = prior[latpos][2:,:] # Cut off first two positions, the bookstein targets are implicit
    initial_state = bkst_init(position=prior, logprob_fn=log_density)
    initial_state.position[latpos] = bookstein_position(initial_state.position[latpos])
    return initial_state

def smc_bookstein_position(position:ArrayLike) -> ArrayLike:
    """
    Turns a set of latent positions into bookstein coordinates, where the third position is constrained to be above the x-axis.
    Keeps particle dimension in mind.
    PARAMS:
    position : (P,N,D) latent positions for all P particles.
    """
    n_particles, n_nodes, n_dims = position.shape
    
    ## Whether to flip each particle around the x-axis. Takes the particle dimension into account.
    _do_flip = position[:,0,1] < 0 # Is first (third when counting implicit anchors) position below x-axis?
    do_flip = jnp.reshape(jnp.repeat(jnp.repeat(_do_flip, n_dims),n_nodes), (n_particles, n_nodes, n_dims))

    ## The flip operator over the x-axis.
    _x_flip = jnp.array([1,-1])
    x_flip = jnp.tile(_x_flip, (n_particles,n_nodes,1))

    ## Then flip the positions over x-axis if we need to flip, or keep the positions the same if no flip.
    position = position*x_flip*do_flip + position*(1-do_flip)
    return position

def add_bkst_to_smc_trace(trace:TemperedSMCState, bkst_dist:float, latpos:str='_z', D:int=2, is_array:bool=False) -> Union[TemperedSMCState,ArrayLike]:
    """
    Adds the Bookstein anchors to the start of each position in the SMC trace.
    PARAMS:
    trace : the smc trace, must be an TemperedSMCState OR be a (M x N x D) trace array.
    bkst_dist : Bookstein distance of the anchors.
    latpos : latent position variable name.
    D : number of latent space dimensions.
    is_array : whether the trace is an (n_iter, N-D, D) array.
    """
    ## Add Bookstein anchors to the array, the anchors must be rigid.
    if is_array:
        bkst_anchors = get_rigid_bookstein_anchors(D, bkst_dist)
        add_bkst = lambda c, s: (None, jnp.concatenate([bkst_anchors, s]))
        _, trace = jax.lax.scan(add_bkst, None, trace)
    ## Add Bookstein anchors to the SMC state.
    else:
        n_particles, _N, D = trace.particles[latpos].shape
        cond = lambda state: state[0] < n_particles

        @jax.jit
        def get_bkst(state):
            i, new_particles = state
            _zb_x = trace.particles[f"{latpos}b_x"][i]
            anchor = get_bookstein_anchors(_zb_x, D, bkst_dist)
            new_z = jnp.concatenate([anchor, trace.particles[latpos][i]])
            new_particles = new_particles.at[i].set(new_z)
            return i+1, new_particles

        new_particles = jnp.zeros(shape=(n_particles,_N+D,D))
        _, new_particles = jax.lax.while_loop(cond, get_bkst, (0, new_particles))
        trace.particles[latpos] = new_particles
    return trace

def smc_bkst_inference_loop(key: PRNGKeyArray, smc_kernel: Callable, initial_state: ArrayLike, max_iters: int=200) -> Union[Tuple[PRNGKeyArray, int, float, TemperedSMCState], Tuple[PRNGKeyArray, int, float, ArrayLike, TemperedSMCState]]:
    """
    Run the temepered SMC algorithm with Bookstein anchoring.

    Run the adaptive algorithm until the tempering parameter lambda reaches the value lambda=1.
    PARAMS:
    key: random key for JAX functions.
    smc_kernel: kernel for the SMC particles.
    initial_state: beginning position of the algorithm.
    max_iters : maximum number of sigma iterations to save (if we save sigma at all).
    """
    ## Define the step function for the continuous models.
    if 'sigma_beta_T' in initial_state.particles:
        n_particles = initial_state.particles['sigma_beta_T'].shape[0]
        start_carry = (key, 0, 0., jnp.zeros((max_iters, n_particles, 1)), initial_state)

        @jax.jit
        def step(carry):
            key, i, lml, sigma_trace, state = carry
            key, subkey = jax.random.split(key)
            state, info = smc_kernel(subkey, state)
            sigma_trace = sigma_trace.at[i].set(state.particles['sigma_beta_T'])
            lml += info.log_likelihood_increment
            return key, i+1, lml, sigma_trace, state
    ## Define the step function for the binary models.
    else:
        start_carry = (key, 0, 0., initial_state)

        @jax.jit
        def step(carry):
            key, i, lml, state  = carry
            key, subkey = jax.random.split(key)
            state, info = smc_kernel(subkey, state)
            lml += info.log_likelihood_increment
            return key, i+1, lml, state

    ## Run the inference.
    cond = lambda carry: carry[-1].lmbda < 1
    results  = jax.lax.while_loop(cond, step, start_carry)

    if results[1] > max_iters: # n_iter always stays in position 1
        print(f"Warning! Embedding took {n_iter} iterations but max_iters is only {max_iters}! Not all proposals are saved.")
    return results