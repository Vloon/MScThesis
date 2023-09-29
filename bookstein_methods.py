import jax
import jax.numpy as jnp

# Typing
from jax._src.prng import PRNGKeyArray
from jax._src.typing import ArrayLike
from blackjax.types import PyTree
from blackjax.smc.tempered import TemperedSMCState
from blackjax.mcmc.rmh import RMHState
from typing import Callable

def get_bookstein_anchors(n_dims:int = 2, offset:float=0.3) -> jnp.ndarray:
    """
    Returns the bookstein coordinate anchors for the first two positions, in n_dims dimensions
    PARAMS:
    n_dims : number of dimensions of the latent space
    offset : offset on the x-axis that the Bookstein anchors are put. Node 1 is put on (-offset, 0), node 2 on (offset, 0).
    """
    assert n_dims > 1, f"Bookstein anchors must be used in a latent space with more than 1 dim, but is being used in a {n_dims} dimensional LS"
    assert offset > 0., f"Offset must be bigger than 0 but is {offset}"
    bookstein_anchors = jnp.zeros(shape=(n_dims, n_dims)) # for each dimension you need one bookstein coordinate
    bookstein_anchors = bookstein_anchors.at[0,0].set(-offset) # First position is negative
    for n in range(1,n_dims):
        bookstein_anchors = bookstein_anchors.at[n,0].set(offset) # Dit klopt nog niet maar het gaat nooit uitmaken
    return bookstein_anchors

def bookstein_position(_z:ArrayLike) -> ArrayLike:
    """
    Turns a set of _z positions into bookstein coordinates, where the first position is constrained. The two Bookstein targets are implicit.
    PARAMS:
    position : _z position in the latent space
    """
    n_nodes, n_dims = _z.shape
    # Whether to flip that particle around the y-axis
    _do_flip = _z[0,1] < 0 # Is first (third when counting implicit bkst) _z below x-axis?
    do_flip = jnp.reshape(jnp.repeat(jnp.repeat(_do_flip, n_dims),n_nodes), (n_nodes, n_dims)) # Don't ask.

    # The flip operator over the x-axis
    _x_flip = jnp.array([1,-1])
    x_flip = jnp.tile(_x_flip, (n_nodes,1))

    # Then flip over x-axis if flip, or keep the same if no flip
    _z = _z*x_flip*do_flip + _z*(1-do_flip)
    return _z

def bookstein_init(bkst_init:RMHState, prior:dict,log_density:Callable) -> PyTree:
    """
    Initializes the prior to start in bookstein position.
    PARAMS:
    bkst_init : initialization function used of the RMH kernel
    prior : prior dictionary containing positions _z
    log_density : log density function to determine the probability of going to the proposed position
    """
    N,D = prior['_z'].shape
    assert N > D, "Must have more than D positions to use Bookstein coordinates, but N={} and D={}".format(N,D)
    prior['_z'] = prior['_z'][2:,:] # Cut off first two positions, the bookstein targets are implicit
    initial_state = bkst_init(position=prior, logprob_fn=log_density)
    initial_state.position['_z'] = bookstein_position(initial_state.position['_z'])
    return initial_state

def smc_bookstein_position(position:ArrayLike) -> ArrayLike:
    """
    Turns a set of _z positions into bookstein coordinates, where the first two positions are set, and the third position is constrained.
    Keeps particle dimension in mind.
    PARAMS:
    position : _z position in the latent space
    """
    n_particles, n_nodes, n_dims = position.shape
    
    # Whether to flip that particle around the y-axis
    _do_flip = position[:,0,1] < 0 # Is first (third with implicit bkst coords) position below x-axis?
    do_flip = jnp.reshape(jnp.repeat(jnp.repeat(_do_flip, n_dims),n_nodes), (n_particles, n_nodes, n_dims))

    # The flip operator over the x-axis
    _x_flip = jnp.array([1,-1])
    x_flip = jnp.tile(_x_flip, (n_particles,n_nodes,1))

    # Then flip over x-axis if flip, or keep the same if no flip
    position = position*x_flip*do_flip + position*(1-do_flip)
    return position

def add_bkst_to_smc_trace(trace:TemperedSMCState, D:int=2, is_array:bool=False) -> TemperedSMCState:
    """
    Adds the bookstein coordinates to the start of each position in the SMC trace.
    PARAMS:
    trace : the smc trace, must have a field .particles with key '_z'
    D : number of latent space dimensions
    is_array : whether the trace is an (n_steps, N-D, D) array
    """
    bkst_target = get_bookstein_target(D)
    add_bkst = lambda c, s: (None, jnp.concatenate([bkst_target, s]))
    if is_array:
        _, trace = jax.lax.scan(add_bkst, None, trace)
    else:
        _, bkst_z_pos = jax.lax.scan(add_bkst, None, trace.particles['_z'])
        trace.particles['_z'] = bkst_z_pos
    return trace