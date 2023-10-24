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

from helper_functions import triu2mat

def plot_hyperbolic_edges(p:ArrayLike, A:ArrayLike, ax:Axes=None, R:float=1, linewidth:float=0.5, threshold:float=0.4) -> Axes:
    """
    PARAMS
    p (N,2) : points on the Poincaré disk
    A (N,N) or (N*(N-1)/2) : (upper triangle of the) adjacency matrix.
    ax : axis to be plotted on
    R : disk radius
    linewidth : linewidth of the edges
    threshold : minimum edge weight for the edge to be plotted
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
    N, D = p.shape
    assert D == 2, f'Dimension must be 2 to be plotted, but is {D}.'
    M = N*(N-1)//2
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
                 pos_labels:ArrayLike=None,
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
    pos_labels : (N,) list of labels for the nodes in the network.
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
    assert len(pos_labels) == N, f'Length of pos_labels should be {N} but is {len(pos_labels)}'
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
                   pos_labels:ArrayLike=None,
                   ax:Axes=None,
                   title:str=None,
                   edge_width:float=0.5,
                   disk_radius:float=1.,
                   hyperbolic:bool=False,
                   continuous:bool=False,
                   bkst:bool=False,
                   threshold:float=0.1,
                   max_th_digits:int=4,
                   margin:float=0.1,
                   s:float=0.5,
                   alpha_margin:float=5e-3,
                   marker_width:float=0.5,
                   hemisphere_symbols:ArrayLike=['s', '^']) -> Axes:
    """
    Plots a network with the given positions.
    PARAMS:
    pos_trace : (n_steps, N, D+1) trace of the positions
    edges : (M,) edge weight or binary edges between positions. If None, no edges are drawn.
    pos_labels : (N,) list of labels for the nodes in the network. All labels should start with 'Left' or 'Right'
    ax : axis to plot the network in
    title : title to display
    edge_width : width of the edges
    disk_radius : the radius of the size of the plot
    hyperbolic : whether the network should be plotted in hyperbolic space or Euclidean space
    continuous : whether the edge weights are continuous or binary
    bkst : whether to deal with the first two nodes as Bookstein nodes
    threshold : minimum edge weight for the edge to be plotted
    max_th_digits : maximum number of digits to show in the title for the threshold
    margin : percentage of disk radius to be added as whitespace
    s : point size for the scatter plot
    alpha_margin : transparancy margin to ensure the most variable position does not have alpha=0.
    marker_width : the width of the border around the marker
    hemisphere_symbols : list of plt marker symbols used for the different hemispheres
    """
    pos_trace = np.array(pos_trace) # Convert to Numpy, probably from JAX.Numpy
    n_steps, N, D = pos_trace.shape
    assert D == 2, f'Dimension must be 2 to be plotted, but is {D}. If plotting hyperbolic, convert to Poincaré coordinates beforehand.'
    if pos_labels is not None:
        pos_labels = np.array(pos_labels)
        assert len(pos_labels) == N, f'Length of pos_labels should be {N} but is {len(pos_labels)}'
    M = N*(N-1)//2
    if edges is not None:
        edges = np.array(edges)
        assert len(edges) == M, f'Length of edges must be {M} but is {len(edges)}'
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
            if bkst:  # Add small jitter to Bookstein coordinates for plottability
                pos_mean[:2, :] += np.random.normal(0, 1e-6, size=(2, 2))
            ax = plot_hyperbolic_edges(p=pos_mean, A=edges, ax=ax, R=disk_radius, linewidth=edge_width, threshold=threshold)
        else:
            ax = plot_euclidean_edges(pos_mean, edges, ax, edge_width)

    # MID: Plot standard deviations
    for n in range(N):
        std = float(np.max(pos_std_nml[n,:]))
        point = Ellipse((pos_mean[n,0], pos_mean[n,1]), width=pos_std[n,0], height=pos_std[n,1], alpha=1-std, color='r', fill=True)
        ax.add_patch(point)

    # TOP: Plot node means
    if pos_labels is not None: # Add position labels
        hemispheres = [lab[0] for lab in pos_labels]
        brain_region = [lab[1] for lab in pos_labels]
        unique_hemis = np.unique(hemispheres)
        unique_regions = np.unique(brain_region)
        assert len(unique_hemis) == len(hemisphere_symbols), f'Length of hemisphere_symbols should match unique number of hemispheres but they are {len(hemisphere_symbols)} and {len(unique_hemis)} respectively.'
        for i, hs in enumerate(unique_hemis):
            ax.scatter(0, 0, c='k', marker=hemisphere_symbols[i], label=hs, linewidth=.5, alpha=0) # Plot one invisible point per marker for plotting
        for j, br in enumerate(unique_regions):
            ax.scatter(0, 0, marker='.', edgecolor='k', label=br, alpha=0)  # Plot one invisible point per color for plotting
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
        # Now plot the actual data without labels
        for i, hs in enumerate(unique_hemis):
            for j, br in enumerate(brain_region):
                ax.scatter(x=pos_mean[:,0],
                            y=pos_mean[:,1],
                            s=s,
                            marker=hemisphere_symbols[i])
    else: # Just plot the positions as default
        if bkst:  # Make bookstein coordinates red
            ax.scatter(pos_mean[:2, 0], pos_mean[:2, 1], c='r', s=s)
            ax.scatter(pos_mean[2:, 0], pos_mean[2:, 1], c='k', s=s)
        else:
            ax.scatter(pos_mean[:, 0], pos_mean[:, 1], c='k', s=s)

    if title: # Both None and empty string will be False
        if threshold > 0:
            thr_tit = round(threshold, max_th_digits) if len(str(threshold)) > max_th_digits else threshold
            title = f"{title}\nthreshold = {thr_tit}"
        ax.set_title(title, color='k', fontsize='24')
    margin = 1+margin
    ax.set(xlim=(-margin*disk_radius,margin*disk_radius),ylim=(-margin*disk_radius,margin*disk_radius))
    ax.axis('off')
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

def plot_metric(csv_file:str,
                x_name:str='n_particles',
                plt_x_name:str=None,
                y_name:str='lml',
                plt_y_name:str=None,
                label_by:str=None,
                plt_label_by:str=None,
                delim:str=';',
                plt_type:str='scatter',
                ax:Axes=None,
                color:str='#a3b23b',
                alpha:float=0.8,
                ):
    """
    Plots an independent parameter on the x-axis versus a dependent parameter on the y-axis. The latter can act as a metric of performance, like log-marginal likelihood or runtime
    Both x and y values are indicated by a name, which should be found in the header of the csv file (sub, n_particles, n_mcmc_steps, task, runtime)
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
    # Asserting x/y lengths is unnecessary, because the csv wouldn't load if they were different.
    if label_by is not None:
        label_val = np.array(data_dict[label_by])

    if ax is None:
        plt.figure()
        ax = plt.gca()

    # Only rotate strings
    rotation = 45 if type(x_val[0]) in [np.str_, str] else 0 ## Okay this is ugly but x_val.dtype = '<U1' while this gives np.str_
    if plt_type == 'scatter':
        if label_by is not None:
            unique_labels = np.unique(label_val)
            for i, ul in enumerate(unique_labels):
                idc = np.where(label_val == ul)
                plt.scatter(x_val[idc], y_val[idc], alpha=alpha, label=f'{plt_label_by} = {ul}')
                plt.legend()
        else:
            plt.scatter(x_val, y_val, c=color, alpha=alpha)
            plt.xticks(rotation=rotation)
    elif plt_type == 'bar':
        bar_width = 0.8 # WE SHOULDN'T GIVE THIS AS A USER DEFINED PARAMETER. This is the default.
        if label_by is not None:
            # Get the means and std of the LML values, ordered by x value
            unique_x = np.unique(x_val)
            unique_labels = np.unique(label_val)
            mean_y = np.zeros((len(unique_x), len(unique_labels)))
            std_y = np.zeros((len(unique_x), len(unique_labels)))
            for i, ux in enumerate(unique_x):
                x_idc = np.array([i for i, x in enumerate(x_val) if x == ux])
                for j, ul in enumerate(unique_labels):
                    l_idc = np.array([i for i, l in enumerate(label_val) if l == ul]) # Use list comprehension because string == np.array returns a single Bool (probably because string is a character array)
                    idc_intersect = np.intersect1d(x_idc, l_idc)
                    mean_y[i,j] = np.mean(y_val[idc_intersect])
                    std_y[i,j] = np.std(y_val[idc_intersect])
            x_plt = np.arange(len(unique_x))*len(unique_labels)*1.5
            for j, ul in enumerate(unique_labels):
                x_offset = j-len(unique_labels)/2
                plt.bar(x = x_plt+x_offset,
                        height = mean_y[:,j],
                        yerr = std_y[:,j],
                        align='center',
                        label=f'{plt_label_by} = {ul}')
            plt.xticks(ticks=x_plt, labels=unique_x, rotation=rotation)
            plt.legend()
        else:
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
            plt.xticks(ticks=x_plt, labels=unique_x, rotation=rotation)
    elif plt_type == 'box':
        unique_x = np.unique(x_val)
        y_box = []
        for ux in unique_x:
            idc = np.where(x_val == ux)
            y_box.append(y_val[idc])
        x_plt = np.arange(len(unique_x))
        plt.boxplot(y_box,
                    notch=True,
                    positions=x_plt)
        plt.xticks(x_plt, labels=unique_x, rotation=rotation)
    plt.xlabel(plt_x_name)
    plt.ylabel(plt_y_name)
    return ax

def plot_correlations(corr:ArrayLike,
                      ax:Axes=None,
                      cmap:str='viridis',
                      add_colorbar:bool=False,
                      vmin:float=None,
                      vmax:float=None) -> Axes:
    """
    Plots the correlations as a heat map
    PARAMS:
    corr : (N,N) or (M,) correlation matrix, or its upper triangle
    ax : the axis to plot on
    cmap : the color map of the heat map
    add_colorbar : whether to add a colorbar as legend-ish thing
    """
    corr = np.array(corr)
    assert len(corr.shape) in [1,2], f'corr must have 1 or 2 dimensions, but has {len(corr.shape)}'
    if len(corr.shape) == 1:
        corr = triu2mat(corr)
    if ax is None:
        plt.figure()
        ax = plt.gca()

    im = ax.imshow(corr, cmap=cmap, vmin=vmin, vmax=vmax)

    if add_colorbar:
        ax.figure.colorbar(im, ax=ax)

    return ax