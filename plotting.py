# Basics
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Ellipse
import matplotlib.colors as mcolors
import jax

import numpy as np

from jax._src.prng import PRNGKeyArray
from jax._src.typing import ArrayLike
from matplotlib import axes as Axes # I want types to be capitalized for some reason.
from typing import Tuple, Callable
from blackjax.smc.tempered import TemperedSMCState

from helper_functions import *

# if __name__ == "__main__":
#     # Create cmd argument list (arg_name, var_name, type, default, nargs[OPT])
#     arguments = [('-if', 'input_folder', str, 'Embeddings'),  # filename of the embeddings
#                  ('-s1', 'subject1', int, 0), # first subject to be used
#                  ('-sn', 'subjectn', int, 25), # last subject to be used (inclusive)
#                  ('-et', 'edge_type', str, 'con'),  # Edge type: 'con' for continuous or 'bin' for binary edges
#                  ('-g', 'geometry', str, 'hyp'), # Latent space geometry: 'hyp' for hyperbolic or 'euc' for euclidean
#                  ('-tf', 'task_file', str, 'task_list.txt'), # filename of the list of task names
#                  ('-of', 'output_folder', str, 'Figures'), # folder where to dump the LSM embeddings
#                  ('-gpu', 'gpu', str, ''), # Number of gpu to use (in string form). If no GPU is specified, CPU is used.
#                  ('-fs', 'figsize', int, 10), # Figure size
#                  ]
#
#     # Get arguments from CMD
#     global_params = get_cmd_params(arguments)
#     input_folder = global_params['input_folder']
#     subject1 = global_params['subject1']
#     subjectn = global_params['subjectn']
#     edge_type = global_params['edge_type']
#     geometry = global_params['geometry']
#     task_file = global_params['task_file']
#     output_folder = global_params['output_folder']
#     gpu = global_params['gpu']
#     if gpu is None:
#         gpu = ''
#     figsize = global_params['figsize']
#
#     print(input_folder)
#     print(output_folder)
#
#     # Set before USING JAX to avoid issues with seeing GPUs!
#     os.environ['CUDA_VISIBLE_DEVICES'] = gpu

def plot_hyperbolic_edges(p:ArrayLike, A:ArrayLike, ax:Axes=None, R:float=1, linewidth:float=0.5, threshold:float=0.4) -> Axes:
    """
    PARAMS
    p (N,2) : points on the Poincaré disk
    A (N,N) or (N*(N-1)/2) : (upper triangle of the) adjacency matrix.
    ax : axis to be plotted on
    R : disk radius
    linewidth : Linewidth of the edges
    """
    def mirror(p:ArrayLike, R:float=1) -> ArrayLike:
        """
        Mirrors point p in circle with R = 1
        Based on: https://math.stackexchange.com/questions/1322444/how-to-construct-a-line-on-a-poincare-disk
        PARAMS:
            p (N,2) : N points on a 2-dimensional Poincaré disk
        RETURNS:
            p_inv (N,2) : Mirror of p
        """
        N, D = p.shape
        p_norm = np.sum(p**2,1)
        p_inv = R**2*p/np.reshape(np.repeat(p_norm,D), newshape=(N,D))
        return p_inv

    def bisectors(p:ArrayLike, R:float=1) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """
        Returns the function for the perpendicular bisector as ax + b
        Based on: https://www.allmath.com/perpendicularbisector.php
        PARAMS:
            p (N,2) : List of points
            R : Size of the disk
        RETURNS:
            a_self (N*(N+1)/2) : Slopes of the bisectors for each combination of points in p
            b_self (N*(N+1)/2) : Intersects of the bisectors for each combination of points in p
            a_inv (N) : Slopes of the bisectors for each point with its mirror
            b_inv (N) : Intersects of the bisectors for each point with its mirror
        """
        N, D = p.shape
        assert D == 2, 'Cannot visualize a Poincaré disk with anything other than 2 dimensional points'
        triu_indices = np.triu_indices(N, k=1)

        # Get mirror of points
        p_inv = mirror(p, R)

        # Tile to get all combinations
        x_rep = np.tile(p[:,0], N).reshape((N,N))
        y_rep = np.tile(p[:,1], N).reshape((N,N))

        # Get midpoints
        mid_x_self = ((x_rep.T+x_rep)/2)[triu_indices]
        mid_y_self = ((y_rep.T+y_rep)/2)[triu_indices]
        mid_x_inv  = (p[:,0]+p_inv[:,0])/2
        mid_y_inv  = (p[:,1]+p_inv[:,1])/2

        # Get slopes
        dx_self = (x_rep - x_rep.T)[triu_indices]
        dy_self = (y_rep- y_rep.T)[triu_indices]
        dx_inv  = p[:,0] - p_inv[:,0]
        dy_inv  = p[:,1] - p_inv[:,1]

        a_self = -1/(dy_self/dx_self)
        a_inv  = -1/(dy_inv/dx_inv)

        # Get intersects
        b_self = -a_self*mid_x_self + mid_y_self
        b_inv  = -a_inv*mid_x_inv + mid_y_inv

        return a_self, b_self, a_inv, b_inv

    if ax is None:
        plt.figure()
        ax = plt.gca()
    N, D = p.shape
    assert D == 2, 'Cannot visualize a Poincaré disk with anything other than 2 dimensional points'
    if len(A.shape) == 2: # Get the upper triangle
        A = A[np.triu_indices(N, k=1)]

    ## Calculate perpendicular bisectors for points in p with each other, and with their mirrors.
    a_self, b_self, a_inv, b_inv  = bisectors(p,R)

    # Repeat elements according to the upper triangle indices
    first_triu_idc = np.triu_indices(N,k=1)[0]
    a_inv_rep = np.array([a_inv[i] for i in first_triu_idc])
    b_inv_rep = np.array([b_inv[i] for i in first_triu_idc])
    px_rep = np.array([p[i,0] for i in first_triu_idc])
    py_rep = np.array([p[i,1] for i in first_triu_idc])

    # Get coordinates and radius of midpoint of the circle
    cx = (b_self-b_inv_rep)/(a_inv_rep-a_self)
    cy = a_self*cx + b_self
    cR = np.sqrt((px_rep-cx)**2 + (py_rep-cy)**2)

    second_triu_idc = np.triu_indices(N,k=1)[1]
    qx_rep = np.array([p[i,0] for i in second_triu_idc])
    qy_rep = np.array([p[i,1] for i in second_triu_idc])

    theta_p = np.degrees(np.arctan2(py_rep-cy, px_rep-cx))
    theta_q = np.degrees(np.arctan2(qy_rep-cy, qx_rep-cx))

    for i in range(int(N*(N-1)/2)):
        if A[i] >= threshold:
            # Honestly... can't really tell you why this works but it does so someone else can do the math.
            if cx[i] > 0:
                theta_p[i] = theta_p[i]%360
                theta_q[i] = theta_q[i]%360

            # Draw arc
            arc = Arc(xy=(cx[i], cy[i]), width=2*cR[i], height=2*cR[i], angle=0, theta1=min(theta_p[i],theta_q[i]), theta2=max(theta_p[i],theta_q[i]), linewidth=linewidth, alpha=A[i])
            ax.add_patch(arc)

    return ax

def plot_euclidean_edges(p:ArrayLike,
                         A:ArrayLike,
                         ax:Axes=None,
                         linewidth:float=0.5,
                         threshold:float=0.4) -> Axes:
    """
    Plots for all entries in A
    PARAMS:
    p : (N, 2) positions of the nodes
    A : (M,) or (N, N) adjacency matrix (or its upper triangle)
    ax : axis to plot the network on
    linewidth : width of the edges
    threshold : minimum edge weight for the edge to be plotted
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()
    if len(A.shape) == 2:
        N = A.shape[0]
        assert A.shape[1] == N, f'A must be of size ({N},{N}), but is {A.shape}.'
        assert p.shape == (N,2), f'p must be ({N},2) but is {p.shape}'
        triu_indices = np.triu_indices(N, k=1)
        A = A[triu_indices]
        M = int(N * (N - 1) / 2)
    elif len(A.shape) == 1:
        M = A.shape[0]
        N = int((1 + np.sqrt(1 + 8 * M))/2)
        triu_indices = np.triu_indices(N, k=1)
    else:
        raise ValueError(f'A should have 1 or 2 dimensions, but has {len(A.shape)}.')
        
    for m in range(M):
        if A[m] > treshold: # A must be in triu_indices format
            p1 = p[triu_indices[0][m], :]
            p2 = p[triu_indices[1][m], :]
            ax.plot(x=[p1[0], p2[0]],
                     y=[p1[1], p2[1]],
                     color='k',
                     linewidth=linewidth,
                     alpha=A[m])
    return ax

def plot_network(A:ArrayLike,
                 pos:ArrayLike,
                 ax:Axes=None,
                 title:str=None,
                 node_color:list=None,
                 node_size:list=None,
                 edge_width:float=0.5,
                 disk_radius:float=1.,
                 hyperbolic:bool=False,
                 continuous:bool=False,
                 bkst:bool=True,
                 threshold:float=0.4,
                 margin:float=0.1) -> Axes:
    """
    Plots a network with the given positions.
    PARAMS:
    A : (M,) or (N, N) adjacency matrix (or its upper triangle)
    pos : (N, 2) positions of the nodes
    ax : axis to plot the network on
    title : title to display
    node_color : list of valid plt colors for the nodes. defaults to N*['k']
    node_size : list of node sizes. defaults to scale with degree.
    edge_width : width of the edges
    disk_radius : the radius of the size of the plot
    hyperbolic : whether the network should be plotted in hyperbolic space or Euclidean space
    continuous : whether the edge weights are continuous or binary
    bkst : whether to deal with the first two nodes as Bookstein nodes
    threshold : minimum edge weight for the edge to be plotted
    margin : percentage of disk radius to be added as whitespace
    """
    # Convert possibly jax numpy arrays to numpy arrays
    A = np.array(A)
    pos = np.array(pos)
    
    if len(A.shape) == 1:
        A = triu2mat(A)
    N = A.shape[0]
    assert pos.shape[1] == 2, f'The 2nd dimension of pos must be 2 but is {pos.shape[1]}'
    assert A.shape[1] == N and pos.shape[0] == N, f'Invalid shapes between obs: {A.shape} and pos: {pos.shape}'

    if node_color is None:
        node_color = N*['k']
    clean_ax = True
    if ax is None:
        plt.figure(facecolor='w')
        ax = plt.gca()
        clean_ax = False
    if node_size is None:
        d = np.sum(A, axis=0)
        node_size = [v**2/10 for v in d]

    # Draw the nodes
    if bkst:
        node_color[:2] = 2*['r']
        node_color[2] = 'y'
    ax.scatter(pos[:,0], pos[:,1],
               s=node_size,
               c=node_color,
               linewidths=2.0 )

    if hyperbolic:
        if bkst: # Add jitter to Bookstein coordinates to avoid dividing by zero
            pos[:2,:] += np.random.normal(0, 1e-6, size=(2,2))
        plot_hyperbolic_edges(p=pos, A=A, ax=ax, R=disk_radius, linewidth=edge_width, threshold=threshold)
    else:
        plot_euclidean_edges(p=pos, A=A, ax=ax, linewidth=edge_width, threshold=threshold)

    if title is not None:
        ax.set_title(title, color='k', fontsize='24')
    margin = 1+margin
    lim = (-margin*disk_radius, margin*disk_radius)
    ax.set(xlim=lim,ylim=lim)
    ax.axis('off')
    if not clean_ax:
        plt.tight_layout()
    return ax

def plot_posterior(pos_trace:ArrayLike,
                   edges:ArrayLike=None,
                   ax:Axes=None,
                   title:str=None,
                   edge_width:float=0.5,
                   disk_radius:float=1.,
                   hyperbolic:bool=False,
                   continuous:bool=False,
                   bkst:bool=False,
                   legend:bool=False,
                   threshold:float=0.4,
                   margin:float=0.1,
                   s:float=0.5,
                   alpha_margin:float=0.005) -> Axes:
    """
    Plots a network with the given positions.
    PARAMS:
    pos_trace : (n_steps, N, D+1) trace of the positions
    edges : (M,) edge weight or binary edges between positions
    ax : axis to plot the network in
    title : title to display
    edge_width : width of the edges
    disk_radius : the radius of the size of the plot
    hyperbolic : whether the network should be plotted in hyperbolic space or Euclidean space
    continuous : whether the edge weights are continuous or binary
    bkst : whether to deal with the first two nodes as Bookstein nodes
    threshold : minimum edge weight for the edge to be plotted
    margin : percentage of disk radius to be added as whitespace
    s : point size for the scatter plot
    alpha_margin : transparancy margin to ensure the most variable position does not have alpha=0.
    """
    pos_trace = np.array(pos_trace) # Convert to Numpy, probably from JAX.Numpy
    edges = np.array(edges)
    n_steps, N, D = pos_trace.shape
    assert D == 2, 'Dimension must be 2 to be plotted. If plotting hyperbolic, convert to Poincaré coordinates beforehand.'
    clean_ax = True
    if ax is None:
        plt.figure(facecolor='w')
        ax = plt.gca()
        clean_ax = False

    pos_mean = np.mean(pos_trace, axis=0) # N,D average position
    pos_std = np.std(pos_trace, axis=0) # N,D position standard deviation
    pos_std_nml = pos_std/np.max(pos_std+alpha_margin) # Normalize so that the max standard deviation is (just under) 1 (which then corresponds to most transparant point)
    
    # BOTTOM: Add edges
    if not edges is None:
        if hyperbolic:
            if bkst:  # Add jitter to Bookstein coordinates for plottability
                pos_mean[:2, :] += np.random.normal(0, 1e-6, size=(2, 2))
            plot_hyperbolic_edges(p=pos_mean, A=edges, ax=ax, R=disk_radius, linewidth=edge_width, threshold=threshold, continuous=continuous, bkst=bkst)
        else:
            plot_euclidean_edges(pos_mean, edges, ax, edge_width)
    # MID: Plot standard deviations
    for n in range(N):
        std = float(np.max(pos_std_nml[n,:]))
        point = Ellipse((pos_mean[n,0], pos_mean[n,1]), width=pos_std[n,0], height=pos_std[n,1], alpha=1-std, color='r', fill=True)
        ax.add_patch(point)

    # TOP: Plot node means
    if bkst: # Make bookstein coordinates red
        ax.scatter(pos_mean[:2,0], pos_mean[:2,1], c='r', s=s, label='Bookstein coordinates')
        ax.scatter(pos_mean[2:,0], pos_mean[2:,1], c='k', s=s)
        if legend:
            ax.legend(loc='best', bbox_to_anchor=(1.01, 0.5))
    else:
        ax.scatter(pos_mean[:,0], pos_mean[:,1], c='k', s=s)

    if title is not None:
        ax.set_title(title, color='k', fontsize='24')
    margin = 1+margin
    ax.set(xlim=(-margin*disk_radius,margin*disk_radius),ylim=(-margin*disk_radius,margin*disk_radius))
    ax.axis('off')
    if not clean_ax:
        plt.tight_layout()
    return ax

def plot_log_marginal_likelihoods(lml:ArrayLike, n_particles:ArrayLike, n_mcmc_steps:ArrayLike, ax:Axes=None):
    """
    Plots a heatmap of the log marginal likelihoods for different numbers of particles x different numbers of mcmc steps
    PARAMS:
    lml : (T,N,M) matrix containing log marginal likelihoods for each task by each number of particles by each number of mcmc steps
    n_particles : (N,) list of number of particles used
    n_mcmc_steps : (M,) list of number of mcmc steps used
    """
    T, N, M = lml.shape
    assert len(n_particles) == N, f'n_particles must be of length {N} but is {len(n_particles)}.'
    assert len(n_mcmc_steps) == M, f'n_mcmc_steps must be of length {N} but is {len(n_mcmc_steps)}.'
    if ax is None:
        plt.figure(figsize=(figsize,figsize))
        ax = plt.gca()
    lml_avg = jnp.mean(lml, axis=0)
    ax.imshow(lml_avg)
    plt.xlabel('Number of particles')
    plt.ylabel('Number of MCMC steps')
    ax.set_xticks(n_particles)
    ax.set_yticks(n_mcmc_steps)
    return ax

def plot_metric(csv_file:str, x_name:str='n_particles', plt_x_name:str=None, y_name:str='lml', plt_y_name:str=None, label_by:str=None, plt_label_by:str=None, delim:str=';', plt_type:str='scatter', ax:Axes=None, color:str='#a3b23b', alpha:float=0.8):
    """
    Plots an independent parameter on the x-axis versus a dependent parameter on the y-axis. The latter can act as a metric of performance, like log-marginal likelihood or runtime
    Both should be found in the csv file (sub, n_particles, n_mcmc_steps, task, runtime)
    PARAMS:
    csv_file : the location of the csv file containing the data (incl. folder). The csv starts with a row of headers, then a row of types
    x_name : the name of the column in the CSV file to use as the value on the x-axis
    plt_x_name : plottable x-name to be used as label
    y_name : the name of the column in the CSV file to use as the value on the y-axis
    plt_y_name : plottable y-name to be used as label
    label_by : the name of the column in the CSV file to label by
    plt_label_by : plottable label name to be used in the legend
    delim : the delimiter of the csv file
    plt_type : type of plot, can only be 'scatter', 'bar' or 'box'
    ax : the axis to plot on
    color : the color of the points
    alpha : the alpha of the points
    """
    valid_plt_types = ['scatter', 'bar', 'box']
    assert plt_type in valid_plt_types, f"plt_type must be in {valid_plt_types} but is {plt_type}"
    data = np.loadtxt(csv_file, delimiter=delim, dtype=str) # Save everything as string so we can keep multiple types in one csv
    headers = data[0,:]
    assert x_name in headers, f'x_name should be in {headers} but is {x_name}'
    assert y_name in headers, f'y_name should be in {headers} but is {y_name}'
    if label_by is not None:
        assert label_by in headers, f'label_by should be in {headers} but is {label_by}'

    plt_x_name = x_name if plt_x_name is None else plt_x_name
    plt_y_name = y_name if plt_y_name is None else plt_y_name
    plt_label_by = label_by if plt_label_by is None else plt_label_by # If label_by is None this is still None

    types = [eval(d) for d in data[1,:]] # Eval casts the str version of a type to the type-cast function
    values = data[2:,:]

    # Create dictionary with for each header as key the correctly typecast value
    data_dict = {h:[types[i](v) for v in values[:,i]] for i, h in enumerate(headers)}
    y_val = np.array(data_dict[y_name])
    x_val = np.array(data_dict[x_name])
    # Asserting x/y lengths is unnecessary, otherwise the csv wouldn't load.
    if label_by is not None:
        label_val = np.array(data_dict[label_by])

    if ax is None:
        plt.figure()
        ax = plt.gca()

    if plt_type == 'scatter':
        if label_by is not None:
            unique_labels = np.unique(label_val)
            for i, ul in enumerate(unique_labels):
                idc = np.where(label_val == ul)
                plt.scatter(x_val[idc], y_val[idc], alpha=alpha, label=f'{plt_label_by} = {ul}')
                plt.legend()
        else:
            plt.scatter(x_val, y_val, c=color, alpha=alpha)
    elif plt_type == 'bar':
        # Get the means and std of the LML values, ordered by x value
        unique_x = np.unique(x_val)
        mean_y = np.zeros(len(unique_x))
        std_y = np.zeros(len(unique_x))
        for i, ux in enumerate(unique_x):
            idc = np.where(x_val == ux)
            mean_y[i] = np.mean(y_val[idc])
            std_y[i] = np.std(y_val[idc])
        x_plt = np.arange(len(unique_x))
        plt.bar(x_plt, mean_y, yerr=std_y, color=color)
        plt.xticks(x_plt, unique_x, rotation=45)
    elif plt_type == 'box':
        unique_x = np.unique(x_val)
        y_box = []
        for ux in unique_x:
            idc = np.where(x_val == ux)
            y_box.append(y_val[idc])
        plt.boxplot(y_box, notch=True)
        plt.xticks(np.arange(len(y_box))+1, labels=unique_x)
    plt.xlabel(plt_x_name)
    plt.ylabel(plt_y_name)
    return ax