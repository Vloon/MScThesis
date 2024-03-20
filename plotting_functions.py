"""
This file contains a large number of plotting functions that are used. The functions in this file only return the plt axes, and don't actually save the plots themselves.
"""

## Basics
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Ellipse
import matplotlib.colors as mcolors
import jax.numpy as jnp
import numpy as np

## Typing
from jax._src.prng import PRNGKeyArray
from jax._src.typing import ArrayLike
from matplotlib import axes as Axes
from typing import Tuple, Callable
from blackjax.smc.tempered import TemperedSMCState

## Self-made functions
from helper_functions import triu2mat, invlogit

def plot_hyperbolic_edges(p:ArrayLike,
                          A:ArrayLike,
                          ax:Axes=None,
                          R:float=1,
                          linewidth:float=0.5,
                          threshold:float=0.4,
                          zorder:float=0,
                          overwrt_alpha:float=None) -> Axes:
    """
    Plots the edges on the Poincaré disk, meaning these will look curved.
    PARAMS
    p (N,2) : points on the Poincaré disk
    A (N,N) or (M) : (upper triangle of the) adjacency matrix.
    ax : axis to be plotted on
    R : disk radius
    linewidth : linewidth of the edges
    threshold : minimum edge weight for the edge to be plotted
    overwrt_alpha : overwrite the alpha value (can be used for binary edges to decrease the intensity)
    """
    def mirror(p:ArrayLike, R:float=1) -> ArrayLike:
        """
        Mirrors point p in circle with radius R
        Based on: https://math.stackexchange.com/questions/1322444/how-to-construct-a-line-on-a-poincare-disk
        PARAMS:
            p (N,2) : N points on a 2-dimensional Poincaré disk
            R : disk radius
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
            R : disk radius
        RETURNS:
            a_self (N*(N+1)/2) : Slopes of the bisectors for each combination of points in p
            b_self (N*(N+1)/2) : Intersects of the bisectors for each combination of points in p
            a_inv (N) : Slopes of the bisectors for each point with its mirror
            b_inv (N) : Intersects of the bisectors for each point with its mirror
        """
        N, D = p.shape
        assert D == 2, 'Cannot visualize a Poincaré disk with anything other than 2 dimensional points'
        triu_indices = np.triu_indices(N, k=1)

        ## Get mirror of points
        p_inv = mirror(p, R)

        ## Tile the points to get all combinations
        x_rep = np.tile(p[:,0], N).reshape((N,N))
        y_rep = np.tile(p[:,1], N).reshape((N,N))

        ## Get midpoints
        mid_x_self = ((x_rep.T+x_rep)/2)[triu_indices]
        mid_y_self = ((y_rep.T+y_rep)/2)[triu_indices]
        mid_x_inv  = (p[:,0]+p_inv[:,0])/2
        mid_y_inv  = (p[:,1]+p_inv[:,1])/2

        ## Get slopes
        dx_self = (x_rep - x_rep.T)[triu_indices]
        dy_self = (y_rep- y_rep.T)[triu_indices]
        dx_inv  = p[:,0] - p_inv[:,0]
        dy_inv  = p[:,1] - p_inv[:,1]
        a_self = -1/(dy_self/dx_self)
        a_inv  = -1/(dy_inv/dx_inv)

        ## Get intersects
        b_self = -a_self*mid_x_self + mid_y_self
        b_inv  = -a_inv*mid_x_inv + mid_y_inv

        return a_self, b_self, a_inv, b_inv

    if ax is None:
        plt.figure()
        ax = plt.gca()
    N, D = p.shape
    M = N*(N-1)//2
    assert D == 2, 'Cannot visualize a Poincaré disk with anything other than 2 dimensional points'
    if len(A.shape) == 2:
        A = A[np.triu_indices(N, k=1)]

    ## Calculate perpendicular bisectors for points in p with each other, and with their mirrors.
    a_self, b_self, a_inv, b_inv = bisectors(p,R)

    ## Repeat elements according to the upper triangle indices
    first_triu_idc = np.triu_indices(N,k=1)[0]
    a_inv_rep = np.array([a_inv[i] for i in first_triu_idc])
    b_inv_rep = np.array([b_inv[i] for i in first_triu_idc])
    px_rep = np.array([p[i,0] for i in first_triu_idc])
    py_rep = np.array([p[i,1] for i in first_triu_idc])

    ## Get coordinates and radius of midpoint of the circle
    cx = (b_self-b_inv_rep)/(a_inv_rep-a_self)
    cy = a_self*cx + b_self
    cR = np.sqrt((px_rep-cx)**2 + (py_rep-cy)**2)

    second_triu_idc = np.triu_indices(N,k=1)[1]
    qx_rep = np.array([p[i,0] for i in second_triu_idc])
    qy_rep = np.array([p[i,1] for i in second_triu_idc])

    ## Get starting and ending angles of the arcs
    theta_p = np.degrees(np.arctan2(py_rep-cy, px_rep-cx))
    theta_q = np.degrees(np.arctan2(qy_rep-cy, qx_rep-cx))

    for m in range(M):
        if A[m] >= threshold:
            ## Correct the angles for quadrant wraparound
            if cx[m] > 0:
                theta_p[m] = theta_p[m]%360
                theta_q[m] = theta_q[m]%360

            ## Draw the arc and add it to the axis
            alpha = A[m] if overwrt_alpha is None else overwrt_alpha
            arc = Arc(xy=(cx[m], cy[m]), width=2*cR[m], height=2*cR[m], angle=0, theta1=min(theta_p[m],theta_q[m]), theta2=max(theta_p[m],theta_q[m]), linewidth=linewidth, alpha=alpha, zorder=zorder)
            ax.add_patch(arc)

    return ax

def plot_euclidean_edges(p:ArrayLike,
                         A:ArrayLike,
                         ax:Axes=None,
                         linewidth:float=0.5,
                         threshold:float=0.4,
                         zorder:float=0,
                         overwrt_alpha:float=None) -> Axes:
    """
    Plots Euclidean edges between points in p.
    PARAMS:
    p : (N, 2) positions of the nodes
    A : (M,) or (N, N) adjacency matrix (or its upper triangle)
    ax : axis to plot the network on
    linewidth : width of the edges
    threshold : minimum edge weight for the edge to be plotted
    zorder : the z-order (depth) of the edges
    overwrt_alpha : overwrite the alpha value (can be used for binary edges to decrease the intensity)
    """
    N, D = p.shape
    assert D == 2, f'Dimension must be 2 to be plotted, but is {D}.'
    M = N*(N-1)//2
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ## Get variables in the right shape
    if len(A.shape) == 2:
        N = A.shape[0]
        assert A.shape[1] == N, f'A must be of size ({N},{N}), but is {A.shape}.'
        assert p.shape == (N,2), f'p must be ({N},2) but is {p.shape}'
        triu_indices = np.triu_indices(N, k=1)
        A = A[triu_indices]
        M = N*(N-1)//2
    elif len(A.shape) == 1:
        M = A.shape[0]
        N = int((1 + np.sqrt(1 + 8 * M))/2)
        triu_indices = np.triu_indices(N, k=1)
    else:
        raise ValueError(f'A should have 1 or 2 dimensions, but has {len(A.shape)}.')

    for m in range(M):
        if A[m] >= threshold:
            alpha = A[m] if overwrt_alpha is None else overwrt_alpha
            p1 = p[triu_indices[0][m], :]
            p2 = p[triu_indices[1][m], :]
            ## Plot edge as a straight line
            ax.plot([p1[0], p2[0]],
                    [p1[1], p2[1]],
                    color='k',
                    linewidth=linewidth, 
                    alpha=alpha,
                    zorder=zorder)
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
                   threshold:float=0,
                   edge_alpha:float=None,
                   edge_zorder:float=0,
                   std_zorder:float=5,
                   mean_zorder:float=10,
                   max_th_digits:int=4,
                   margin:float=0.1,
                   s:float=20,
                   alpha_margin:float=5e-3,
                   marker_color:str='0.6',
                   brainregion_cmap:str='jet',
                   one_region:bool=True,
                   hemisphere_symbols:ArrayLike=['+', 'x'],
                   legend_fontsize:float=18,
                   zoom:bool=False) -> Axes:
    """
    Plots the posterior positions, possibly including the networks' edges.
    PARAMS:
    pos_trace : (n_steps, N, D+1) trace of the positions
    edges : (M,) edge weight or binary edges between positions. If None, no edges are drawn.
    pos_labels : (N,) list of labels for the nodes in the network. All labels should start with 'Left' or 'Right'
    ax : axis to plot the network in
    title : title to display
    edge_width : width of the edges
    disk_radius : the disk radius, doubling as the x- and y-limits of the plot
    hyperbolic : whether the network should be plotted in hyperbolic space or Euclidean space
    continuous : whether the edge weights are continuous or binary
    bkst : whether to deal with the first two nodes as Bookstein nodes
    threshold : minimum edge weight for the edge to be plotted
    edge_alpha : alpha for plotting the binary edges to improve visibility
    max_th_digits : maximum number of digits to show in the title for the threshold
    margin : percentage of disk radius to be added as whitespace on the sides
    s : point size for the scatter plot
    alpha_margin : transparancy margin to ensure the most variable position does not have alpha=0.
    marker_color : color for the position means markers
    brainregion_cmap : cmap string to color the brain regions
    one_region : whether to commit each brain region to one lobe only, rather than splitting certain nodes over multiple regions
    hemisphere_symbols : list of plt marker symbols used for the different hemispheres
    legend_fontsize : fontsize used in the legend.
    zoom : whether the image is zoomed (meaning the legend should shift upwards)
    """
    ## Convert possible JAX.numpy arrays to numpy
    pos_trace = np.array(pos_trace)
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
    ## Prepare position labels and colors
    if pos_labels is not None:
        empty_labels = ['NaN', 'nan', '']
        hemispheres = [lab[0] for lab in pos_labels]
        brain_regions = [lab[1] for lab in pos_labels]
        empty_idc = [i for i, br in enumerate(brain_regions) if br in empty_labels]
        has_empty_labels = len(empty_idc) > 0
        if one_region:
            brain_regions = [br.split(';')[0] for br in brain_regions]
        unique_hemis = list(np.unique(hemispheres))
        unique_regions = list(np.unique(brain_regions))
        ubr_colors = [plt.get_cmap(brainregion_cmap)(i) for i in np.linspace(0,1,len(unique_regions))]

        br_colors = [ubr_colors[unique_regions.index(br)] if i not in empty_idc else '0.75' for i, br in enumerate(brain_regions)]
        assert len(unique_hemis) == len(hemisphere_symbols), f'Length of hemisphere_symbols should match unique number of hemispheres in the label file but they are {len(hemisphere_symbols)} and {len(unique_hemis)} respectively.'

    margin = 1 + margin

    ## Get position means and stds, and normalize so that the maximum standard deviation is just under 1 which then corresponds to most transparant point.
    pos_mean = np.mean(pos_trace, axis=0)
    pos_std = np.std(pos_trace, axis=0)
    pos_std_nml = pos_std/np.max(pos_std+alpha_margin)
    
    ## Bottom layer: Add edges
    if edges is not None:
        if hyperbolic:
            if bkst:
                ## Add small jitter to Bookstein coordinates for plottability
                pos_mean[:2, :] += np.random.normal(0, 1e-6, size=(2, 2))
            ax = plot_hyperbolic_edges(p=pos_mean, A=edges, ax=ax, R=disk_radius, linewidth=edge_width, threshold=threshold, zorder=edge_zorder, overwrt_alpha=edge_alpha)
        else:
            ax = plot_euclidean_edges(pos_mean, edges, ax, edge_width, threshold=threshold, zorder=edge_zorder, overwrt_alpha=edge_alpha)

    ## Middle layer: Plot standard deviations
    ell_colors = br_colors if pos_labels is not None else ['r']*N
    for n in range(N):
        alpha = 1 - float(np.max(pos_std_nml[n,:]))
        ell = Ellipse((pos_mean[n,0], pos_mean[n,1]), width=pos_std[n,0], height=pos_std[n,1], alpha=0.8, color=ell_colors[n], fill=True, zorder=std_zorder)
        ax.add_patch(ell)

    ## Top layer: Plot node means
    if pos_labels is not None:
        ## To create the legend, we plot one point per marker far outsize the range we look at.
        coord = margin*disk_radius*10 
        marker_edgewidth = min(np.sqrt(s), s**2)/5 # Edge width doesn't seem to visually scale linearly with the size of the marker, but we also need to deal with s < 1.
        s_lab = max(np.sqrt(s), s**2)*2 # (the point sizes are decided visually, since markers don't seem to be consistent in size)
        ## Create left vs right hemisphere markers
        for i, hs in enumerate(unique_hemis):
            ax.scatter(coord, coord,
                       s=s_lab,
                       marker=hemisphere_symbols[i],
                       color=marker_color,
                       label=hs,
                       )
        ## Create lobe markers
        for j, br in enumerate(unique_regions):
            if br not in empty_labels:
                ax.scatter(coord, coord,
                           s=s_lab,
                           marker='o',
                           color=ubr_colors[j],
                           edgecolor='k',
                           linewidth=marker_edgewidth,
                           label=br,
                           )
        ncol = (len(unique_hemis)+len(unique_regions))//2
        bbox_anchor = (0.5, 0.1) if zoom else (0.5, 0)
        ax.legend(loc='upper center', bbox_to_anchor=bbox_anchor, ncol=ncol, fontsize=legend_fontsize)
        ## Plot the actual data
        for n in range(N):
            hs_idx = unique_hemis.index(pos_labels[n][0])
            ax.scatter(x=pos_mean[n,0],
                       y=pos_mean[n,1],
                       s=s,
                       marker=hemisphere_symbols[hs_idx],
                       color=marker_color,
                       linewidth=marker_edgewidth,
                       zorder=mean_zorder,
                       )
    else:
        ## Without labels, just plot the positions as default
        if bkst:
            ## Color Bookstein anchors red
            ax.scatter(pos_mean[:2, 0], pos_mean[:2, 1], c='r', s=s, zorder=mean_zorder)
            ax.scatter(pos_mean[2:, 0], pos_mean[2:, 1], c='k', s=s, zorder=mean_zorder)
        else:
            ax.scatter(pos_mean[:, 0], pos_mean[:, 1], c='k', s=s, zorder=mean_zorder)

    if title:
        ## Create fitting title
        if threshold > 0 and continuous:
            thr_tit = round(threshold, max_th_digits) if len(str(threshold)) > max_th_digits else threshold
            title = f"{title}\nthreshold = {thr_tit}"
        ax.set_title(title, color='k', fontsize='24')
    ax.set(xlim=(-margin*disk_radius,margin*disk_radius),ylim=(-margin*disk_radius,margin*disk_radius))
    ax.axis('off')
    return ax

def plot_network(A: ArrayLike,
                 pos: ArrayLike,
                 pos_labels: ArrayLike = None,
                 ax: Axes = None,
                 title: str = None,
                 node_color: list = None,
                 node_size: list = None,
                 edge_width: float = 0.5,
                 disk_radius: float = 1.,
                 hyperbolic: bool = False,
                 continuous: bool = False,
                 bkst: bool = True,
                 threshold: float = 0.4,
                 margin: float = 0.1
                 ) -> Axes:
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
    ## Convert possibly JAX.numpy arrays to numpy arrays
    A = np.array(A)
    pos = np.array(pos)

    if len(A.shape) == 1:
        A = triu2mat(A)
    N = A.shape[0]
    assert pos.shape[1] == 2, f'The 2nd dimension of pos must be 2 but is {pos.shape[1]}'
    assert A.shape[1] == N and pos.shape[0] == N, f'Invalid shapes between obs: {A.shape} and pos: {pos.shape}'
    assert len(pos_labels) == N, f'Length of pos_labels should be {N} but is {len(pos_labels)}'
    if node_color is None:
        node_color = N * ['k']
        if bkst:
            node_color[:2] = 2 * ['r']
            node_color[2] = 'b'
    clean_ax = True
    if ax is None:
        plt.figure(facecolor='w')
        ax = plt.gca()
        clean_ax = False
    if node_size is None:
        d = np.sum(A, axis=0)
        node_size = [v ** 2 / 10 for v in d]

    ## Draw the nodes
    ax.scatter(pos[:, 0], pos[:, 1],
               s=node_size,
               c=node_color,
               linewidths=2.0)

    if hyperbolic:
        if bkst:
            ## Add jitter to Bookstein coordinates to avoid dividing by zero
            pos[:2, :] += np.random.normal(0, 1e-6, size=(2, 2))
        ## Plot edges
        ax = plot_hyperbolic_edges(p=pos, A=A, ax=ax, R=disk_radius, linewidth=edge_width, threshold=threshold)
    else:
        ax = plot_euclidean_edges(p=pos, A=A, ax=ax, linewidth=edge_width)

    if title is not None:
        ax.set_title(title, color='k', fontsize='24')
    margin = 1 + margin
    lim = (-margin * disk_radius, margin * disk_radius)
    ax.set(xlim=lim, ylim=lim)
    ax.axis('off')
    if not clean_ax:
        plt.tight_layout()
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
                label_fontsize:float=20,
                tick_fontsize:float=16
                ):
    """
    Plots an independent parameter on the x-axis versus a dependent parameter on the y-axis. The latter can act as a metric of performance, like log-marginal likelihood or runtime
    Both x and y values are indicated by a name, which should be found in the header of the csv file (sub, n_particles, n_mcmc_steps, task, runtime)
    PARAMS:
    csv_file : the location of the csv file containing the data (incl. folder). The csv starts with a row of headers, then a row of types, then the data
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
    label_fontsize : fontsize of the labels and legends
    tick_fontsize : fontsize of the x/y ticks
    """
    valid_plt_types = ['scatter', 'bar', 'box']
    assert plt_type in valid_plt_types, f"plt_type must be in {valid_plt_types} but is {plt_type}"
    data = np.loadtxt(csv_file, delimiter=delim, dtype=str) # Everything is saved as a string
    headers = data[0,:]
    assert x_name in headers, f'x_name should be in {headers} but is {x_name}'
    assert y_name in headers, f'y_name should be in {headers} but is {y_name}'
    if label_by is not None:
        assert label_by in headers, f'label_by should be in {headers} but is {label_by}'

    plt_x_name = x_name if plt_x_name is None else plt_x_name
    plt_y_name = y_name if plt_y_name is None else plt_y_name
    plt_label_by = label_by if plt_label_by is None else plt_label_by

    ## Cast the str version of a type to the type-cast function
    types = [eval(d) for d in data[1,:]]
    values = data[2:,:]

    ## Create dictionary with for each header as key the correctly typecast value
    data_dict = {h:[types[i](v) for v in values[:,i]] for i, h in enumerate(headers)}
    y_val = np.array(data_dict[y_name])
    x_val = np.array(data_dict[x_name])
    if label_by is not None:
        label_val = np.array(data_dict[label_by])

    if ax is None:
        plt.figure()
        ax = plt.gca()

    ## Rotate the strings
    rotation = 45 if type(x_val[0]) in [np.str_, str] else 0
    if plt_type == 'scatter':
        if label_by is not None:
            unique_labels = np.unique(label_val)
            for i, ul in enumerate(unique_labels):
                idc = np.where(label_val == ul)
                plt.scatter(x_val[idc], y_val[idc], alpha=alpha, label=ul)
                plt.legend(fontsize=label_fontsize, bbox_to_anchor=(0.6,1.0)).set_title(title=plt_label_by, prop={'size': label_fontsize})

        else:
            plt.scatter(x_val, y_val, c=color, alpha=alpha)
            plt.xticks(rotation=rotation)
    elif plt_type == 'bar':
        bar_width = 0.8 # This is the plt default, and should not be changed.
        if label_by is not None:
            ## Get the means and std of the dependent parameter, ordered by x value
            unique_x = np.unique(x_val)
            unique_labels = np.unique(label_val)
            mean_y = np.zeros((len(unique_x), len(unique_labels)))
            std_y = np.zeros((len(unique_x), len(unique_labels)))
            for i, ux in enumerate(unique_x):
                x_idc = np.array([i for i, x in enumerate(x_val) if x == ux])
                for j, ul in enumerate(unique_labels):
                    l_idc = np.array([i for i, l in enumerate(label_val) if l == ul])
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
                        label=ul)
            plt.xticks(ticks=x_plt, labels=unique_x, rotation=rotation)
            plt.legend(fontsize=label_fontsize).set_title(title=plt_label_by, prop={'size':label_fontsize})
        else:
            ## Get the means and std of the LML values, ordered by x value
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
    plt.xlabel(plt_x_name, fontsize=label_fontsize)
    plt.ylabel(plt_y_name, fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    return ax

def plot_timeseries(timeseries:ArrayLike, y_offset:float=0.1, ax:Axes=None) -> Axes:
    """
    Creates a plot of the timeseries for each subject, task, and encoding.
    PARAMS:
    timeseries : (N, T) timeseries data 
    y_offset : offset between timeseries 
    ax : figure axis
    """
    N, ts_len = timeseries.shape
    ## Normalize each timeseries to fit within a -1 to 1 range
    nrm_constant = np.tile(np.max(np.abs(timeseries), axis=1), (ts_len, 1)).T
    nrm_ts = timeseries / nrm_constant

    ## Plot the timeseries
    yticks = np.arange(N)*(2+y_offset)
    for n in range(N):
        ax.plot(np.arange(ts_len), nrm_ts[n]+yticks[n], color='k')
    plt.yticks(yticks, [])
    plt.ylim((yticks[0] - 1 - y_offset, yticks[-1] + 1 + y_offset))
    plt.xlim((0, ts_len - 1))
    
    return ax, 

                
def plot_sigma_convergence(sigma_chain:ArrayLike,
                           true_sigma:float = None,
                           ax:Axes=None,
                           n_bins:int = 100,
                           x_offset:float = 0.5,
                           x_tick_interval:int = None,
                           hcolor:str='tab:gray',
                           true_sigma_label:str=r"True $\sigma$/bound",
                           legend:bool=True,
                           label_fontsize:float=20,
                           tick_fontsize:float=16) -> Axes:
    """
    Plots the evolution of sigma as the sequence of distributions over SMC iterations.
    PARAMS:
    sigma_chain : (n_iter, n_particles) array of sigma proposals
    true_sigma : if passed, will plot a red line to indicate the true sigma value (in logit-form).
    ax : the axis to plot on
    n_bins : number of bins per histogram
    x_tick_interval : how many-th iteration is ticked. If None is passed, it autoscales.
    x_offset : distance in between each histogram
    hcolor : color of the histograms
    true_sigma_label : legend label for the true sigma line
    legend : whether to add a legend
    label_fontsize : fontsize of the labels and legend
    """
    if ax is None:
        plt.figure(20, 10)
        ax = plt.gca()

    ## Get tick values
    n_iter = len(sigma_chain)
    if x_tick_interval is None:
        x_tick_interval = n_iter//10

    for it in range(n_iter):
        ## Create histogram for this iteration
        sigma_hist, sigma_bins = jnp.histogram(invlogit(sigma_chain[it]), bins=n_bins)
        ## Normalize so the peak is at 1
        sigma_hist_nml = sigma_hist / jnp.max(sigma_hist)
        ## Add some offset to not have the distributions touch
        x_start = it * (1 + x_offset)
        ## Plot the distribution over the y-axis
        ax.stairs(sigma_hist_nml + x_start, sigma_bins, baseline=x_start, fill=True, color=hcolor, orientation='horizontal')

    ## Set ticks
    ax.set_xticks(ticks=[it * (1 + x_offset) for it in range(n_iter)][::x_tick_interval], labels=np.arange(n_iter)[::x_tick_interval], fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    ## Plot line for ground truth value
    xmin, xmax = 0, (n_iter + 1) * (1 + x_offset)
    if true_sigma is not None:
        ax.hlines(invlogit(true_sigma), xmin, xmax, 'tab:red', label=true_sigma_label)
    ax.set_xlim(xmin, xmax)
    if legend:
        ax.legend(fontsize=label_fontsize)
    return ax