import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import pickle
import os
import jax
import jax.numpy as jnp
from typing import Tuple
from jax._src.typing import ArrayLike

from helper_functions import get_cmd_params, set_GPU, is_valid, get_filename_with_ext, get_safe_folder, load_observations, triu2mat

### Create cmd argument list (arg_name, var_name, type, default[OPT], nargs[OPT]).
###  - arg_name is the name of the argument in the command line.
###  - var_name is the name of the variable in the returned dictionary (which we re-use as variable name here).
###  - type is the data-type of the variable.
###  - default is the default value it takes if nothing is passed to the command line. This argument is only optional if type is bool, where the default is always False.
###  - nargs is the number of arguments, where '?' (default) is 1 argument, '+' concatenates all arguments to 1 list. This argument is optional.
arguments = [('-s1', 'subject1', int, 1), # first subject
             ('-sn', 'subjectn', int, 100),  # last subject
             ('-df', 'data_folder', str, 'Data'), # path to get to the data
             ('-tf', 'task_file', str, 'task_list'), # task list name WITHOUT EXTENTION
             ('-if', 'input_file', str, 'processed_data_downsampled_evenly_spaced'), # filename of the processed data WITHOUT EXTENTION
             ('-of', 'output_file', str, 'binary_data'), # filename to save the binary data WITHOUT EXTENTION
             ('-minth', 'min_threshold', float, 1e-2), # minimum threshold to try
             ('-maxth', 'max_threshold', float, 1-1e-2), # maximum threshold to try (inlcuded)
             ('-nth', 'n_thresholds', int, 9), # number of thresholds to try
             ('-N', 'N', int, 164), # number of nodes
             ('--partial', 'partial', bool), # whether to use partial correlations
             ('--bpf', 'bpf', bool), # whether to use the band-pass filtered data
             ('--print', 'do_print', bool), # whether to print cute info
             ('--plot', 'make_plot', bool), # whether to make plots
             ('-ff', 'figure_folder', str, 'Figures/binary_data_distribution'),  # base folder where to dump the figures
             ('-infofile', 'info_filename', str, 'binary_info'), # filename of the info file
             ('-method', 'method', str, 'max_unconnected'), # Method to threshold the data. 'set_th' uses the same set threshold for each sub/task, 'top_edges' uses the top % edges, 'max_unconnected' leaves at most 1-% nodes unconnected
             ('-pedges', 'percent_edges', float, 0.95), # Percentage of edges that are included when using 'top_edges' method, or 1-percent_edges is used as maximum in 'max_unconnected'
             ('-nbins', 'n_bins', int, 50), # number of bins for the log-log degree distribution plot
             ('-palpha', 'plot_alpha', float, .6), # alpha for plotting stuff
             ('-cmap', 'cmap', str, 'bwr'), # cmap for plotting
             ('-lfs', 'label_fontsize', float, 20),  # fontsize of labels (and legend)
             ('-tfs', 'tick_fontsize', float, 16),  # fontsize of the tick labels
             ('-wsz', 'wrapsize', float, 20),  # wrapped text width
             ('-gpu', 'gpu', str, ''), # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]

valid_methods = ['set_th', 'top_edges', 'max_unconnected']

## Get arguments from command line and do a few basic checks.
global_params = get_cmd_params(arguments)
set_GPU(global_params['gpu'])
subject1 = global_params['subject1']
subjectn = global_params['subjectn']
assert subject1 <= subjectn, f"Subject 1 must be smaller than subject n but they are {subject1} and {subjectn} respectively."
data_folder = global_params['data_folder']
partial = global_params['partial']
bpf = global_params['bpf']
task_filename = get_filename_with_ext(global_params['task_file'], ext='txt', folder=data_folder)
data_filename = get_filename_with_ext(global_params['input_file'], partial, bpf, folder=data_folder)
min_threshold = global_params['min_threshold']
max_threshold = global_params['max_threshold']
n_thresholds = global_params['n_thresholds']
N = global_params['N']
M = N*(N-1)//2
output_file = global_params['output_file']
do_print = global_params['do_print']
make_plot = global_params['make_plot']
figure_folder = get_safe_folder(global_params['figure_folder'])
info_filename = get_filename_with_ext(global_params['info_filename'], ext='txt', folder=data_folder)
method = global_params['method']
assert method in valid_methods, f"method should be in {valid_methods} but is {method}"
percent_edges = global_params['percent_edges']
n_included = int(np.ceil(M*percent_edges)) # We want at least percent_edge, hence ceiling (ceil keeps it a float, hence int)
n_bins = global_params['n_bins']
plot_alpha = global_params['plot_alpha']
cmap = global_params['cmap']
label_fontsize = global_params['label_fontsize']
tick_fontsize = global_params['tick_fontsize']
wrapsize = global_params['wrapsize']

## Load continuous data
con_obs, tasks, encs = load_observations(data_filename, task_filename, subject1, subjectn, M)
subjects = range(subject1, subjectn + 1)
n_scans = len(subjects) * len(tasks) * len(encs)

## Some defaults
x_offset = 0.5
n_x_vals = len(subjects) * len(tasks)
x_tick_interval = max(n_x_vals//10, 1)
hcolor = [plt.get_cmap(cmap)(i) for i in np.linspace(0, 1, len(encs))]

def get_i(si:int, ti:int, ei:int, T:int=len(tasks), E:int=len(encs)) -> int:
    """
    Returns the flattened index based on subject, task, encoding indices
    PARAMS:
    si : subject index
    ti : task index
    ei : phase-encoding index
    T : number of tasks
    E : number of phase-encodings
    """
    return si * T * E + ti * E + ei


def custom_formatter(x:float, pos:float) -> str:
    """
    Custom string formatter for the xticks
    PARAMS:
    x : value
    pos : position (is not used, but is passed by the formatter so should be kept)
    """
    ma, ex = f'{x:.1e}'.split('e')
    return r"${}\times10^{}$".format(ma, ex[-1])

## Create correlation range plot
if make_plot:
    correlation_range = np.empty((n_scans, 2))

    for si, n_sub in enumerate(subjects):
        for ti, task in enumerate(tasks):
            for ei, enc in enumerate(encs):
                correlation_range[get_i(si, ti, ei), :] = [np.min(con_obs[si, ti, ei, :]), np.max(con_obs[si, ti, ei, :])]

    correlation_range = np.sort(correlation_range, axis=0)[::-1] ## Sort by correlations high to low

    ppi = 72 # points per inch, default in plt
    w = 8 # inch
    h = 2*w # inch
    boff = 1.1 # bar plot offset

    lw = h*72/(boff*n_scans)

    plt.figure(figsize=(w, h))
    for i in range(n_scans):
        plt.hlines(-i, correlation_range[i,0], correlation_range[i,1], color='0.05', alpha=plot_alpha, linewidth=lw)
    plt.ylim(-n_scans, 1)
    offset = .05
    plt.xlim(-offset,1+offset)
    plt.yticks(ticks=[])
    plt.xticks(fontsize=tick_fontsize)
    plt.ylabel('Scan', fontsize=label_fontsize)
    plt.xlabel('Correlation', fontsize=label_fontsize)
    savename = get_filename_with_ext(f"correlation_ranges", partial=partial, ext='png', folder=figure_folder)
    plt.savefig(savename, bbox_inches='tight')
    plt.close()

if method == 'set_th':
    ###
    ### Uses the min_threshold (cmd: minth), max_threshold (cmd: maxth), and n_threholds (cmd: nth) to get different thresholds.
    ### For each of those thresholds, all correlations larger than the threshold are set to an edge, all correlations smaller than the threhsold are set to a non-edge.
    ###
    thresholds = np.linspace(min_threshold, max_threshold, n_thresholds)
    nodes_excluded = np.zeros((n_thresholds, n_scans))

    for thi, threshold in enumerate(thresholds):
        if do_print:
           print(f"Binarizing at theta={threshold}")
        ### TODO: Number of digits dependent on the thresholds or something smart there
        bin_filename = get_filename_with_ext(f"{output_file}_th{threshold:.2f}", partial=partial, folder=data_folder)
        bin_data = {}
        ## Create binary observations matrix
        for si, n_sub in enumerate(subjects):
            for ti, task in enumerate(tasks):
                for ei, enc in enumerate(encs):
                    dict_key = f"S{n_sub}_{task}_{enc}"
                    bin_data[dict_key] = con_obs[si, ti, ei, :] > threshold

                    degree = np.sum(triu2mat(bin_data[dict_key]), axis=1)
                    nodes_excluded[thi, get_i(si, ti, ei)] = np.sum(degree == 0)

                    ## Plot degree distribution
                    if make_plot:
                        _n_bins = min(len(np.unique(degree)), n_bins)
                        degree_hist, degree_bins = jnp.histogram(degree, bins=_n_bins, density=True)
                        plot_position = (degree_bins[1:] + degree_bins[:-1]) / 2 # Plot in the middle of the bin
                        plt.plot(plot_position, degree_hist, color='k')
                        plt.xscale('log')
                        plt.yscale('log')
                        plt.gca().xaxis.set_major_locator(LogLocator())
                        plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_formatter))
                        plt.xlabel('Degree', fontsize=label_fontsize)
                        plt.xticks(fontsize=tick_fontsize)
                        plt.ylabel('Density', fontsize=label_fontsize)
                        plt.yticks(fontsize=tick_fontsize)
                        savename = get_filename_with_ext(f"{dict_key}_degree_dist_th{threshold:.2f}", partial=partial, ext='png', folder=figure_folder)
                        plt.savefig(savename, bbox_inches='tight')
                        plt.close()

        ## Save binarized data
        with open(bin_filename, 'wb') as f:
            pickle.dump(bin_data, f)
        if do_print:
            print(f"Saved data at {bin_filename}")

    ## Plot the percentage of nodes included per threshold
    if make_plot:
        plt.figure()
        for thi, threshold in enumerate(thresholds):
            plt.scatter(np.repeat(threshold, n_scans), 1-nodes_excluded[thi]/N, c='k', s=1, alpha=plot_alpha)
        offset = .05
        plt.hlines(percent_edges, min_threshold-offset, max_threshold+offset, color='r', linestyle='dashed')
        plt.xlabel('Threshold', fontsize=label_fontsize)
        nticks = 4
        ticks = [round(t, 1) for t in np.linspace(min_threshold, max_threshold, nticks)]
        plt.xticks(ticks, fontsize=tick_fontsize)
        plt.ylabel('Percentage nodes included', fontsize=label_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        savename = get_filename_with_ext(f"nodes_included", partial=partial, ext='png', folder=figure_folder)
        plt.savefig(savename, bbox_inches='tight')
        plt.close()

elif method == 'top_edges':
    ###
    ### Take the top percent_edges% of weighted edges as unweighted edges, leaving all other weighted edges as non-edges.
    ###
    degree_dist = np.zeros((len(subjects),len(tasks),len(encs), N))
    bin_filename = get_filename_with_ext(f"{output_file}_{percent_edges:.2f}percent_edges", partial=partial, folder=data_folder)
    bin_data = {}
    for si, n_sub in enumerate(subjects):
        for ti, task in enumerate(tasks):
            for ei, enc in enumerate(encs):
                dict_key = f"S{n_sub}_{task}_{enc}"
                edge_idc = np.argsort(con_obs[si, ti, ei, :])[::-1][:n_included] # Sort highest to lowest, then take the first n_included indices to be edges
                bin_obs = np.zeros((M,))
                bin_obs[edge_idc] = 1
                bin_data[dict_key] = bin_obs

                degree = np.sum(triu2mat(bin_data[dict_key]), axis=1)
                degree_dist[si, ti, ei] = degree

                ## Plot degree distribution
                if make_plot:
                    _n_bins = min(len(np.unique(degree)), n_bins)
                    degree_hist, degree_bins = jnp.histogram(degree, bins=_n_bins, density=True)
                    plot_position = (degree_bins[1:] + degree_bins[:-1]) / 2  # Plot in the middle of the bin

                    plt.figure()
                    plt.plot(plot_position, degree_hist, color='k')
                    plt.xscale('log')
                    plt.yscale('log')

                    # First set log spaced xticks, then re-write them as "a.b x 10^c"
                    plt.gca().xaxis.set_major_locator(LogLocator())
                    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_formatter))
                    plt.gca().xaxis.set_minor_locator(LogLocator())
                    
                    plt.xlabel('Degree',fontsize=label_fontsize)
                    plt.xticks(fontsize=tick_fontsize)
                    plt.ylabel('Density',fontsize=label_fontsize)
                    plt.yticks(fontsize=tick_fontsize)
                    savename = get_filename_with_ext(f"{dict_key}_degree_dist_top_{percent_edges:.2f}_edges", partial=partial, ext='png', folder=figure_folder)
                    plt.savefig(savename, bbox_inches='tight')
                    plt.close()

    ## Save binarized data
    with open(bin_filename, 'wb') as f:
        pickle.dump(bin_data, f)
    if do_print:
        print(f"Saved data at {bin_filename}")

elif method == 'max_unconnected':
    ###
    ### Set threshold in order of the correlations. This means the first iteration, the network is fully connected, in the second network, one edge is removed (the one with the lowest correlation).
    ### Repeat this process until at most (1-percent_edges)% are no longer part of the main network component. 
    ###
    def cond(state) -> bool:
        """
        PARAMS:
        state
            th_idx : index of the ordered threshold
            degree : (N,) degree distribution given the threshold
        """
        _, degree = state
        return jnp.sum(degree==0)/N < 1-percent_edges

    def get_degree_dist(state:Tuple[int, ArrayLike], obs:ArrayLike, sorted_idc:ArrayLike) -> Tuple[int, ArrayLike]:
        """
        Returns the threshold's index in the sorted list and the degree distribution of the binary network
        PARAMS:
        state
            th_idx : index of the ordered threshold
            degree : (N,) degree distribution given the threshold
        obs : (M,) continuous observation
        sorted_idc : (M,) indices of the sorted observation
        """
        th_idx, _ = state
        threshold = obs[sorted_idc[th_idx]]
        bin_obs = obs > threshold
        degree = jnp.sum(triu2mat(bin_obs), axis=1)
        return th_idx+1, degree

    degree_dist = np.zeros((len(subjects),len(tasks),len(encs), N))
    bin_filename = get_filename_with_ext(f"{output_file}_max_{1-percent_edges:.2f}unconnected", partial=partial, folder=data_folder)
    bin_data = {}
    for si, n_sub in enumerate(subjects):
        for ti, task in enumerate(tasks):
            for ei, enc in enumerate(encs):
                dict_key = f"S{n_sub}_{task}_{enc}"

                ### Increases the threshold to the next highest edge value one by one, until more than 5% of the nodes are unconnected.
                ### The while loop runs until degree is too low, which means th_idx is one more than we want, but we also return th_idx + 1, so it's 2 too high.
                this_obs = jnp.array(con_obs[si, ti, ei, :])
                sorted_idc = jnp.argsort(this_obs)
                print(f"Sorted idc: {sorted_idc}")
                get_degree_dist_func = lambda state: get_degree_dist(state, obs=this_obs, sorted_idc=sorted_idc)
                th_idx_plus_two, _ = jax.lax.while_loop(cond, get_degree_dist_func, (0, jnp.ones(N)))

                threshold = this_obs[sorted_idc[th_idx_plus_two - 2]]
                bin_obs = this_obs > threshold

                bin_data[dict_key] = bin_obs

                if do_print:
                    print(f"{dict_key} has {jnp.sum(bin_obs)} total edges.")

                degree = jnp.sum(triu2mat(bin_obs), axis=1)
                degree_dist[si, ti, ei] = degree

                info_string = f"{dict_key} has threshold {threshold:.4f} which leaves {100 * jnp.sum(degree == 0) / N:.2f}% unconnected"
                with open(info_filename, 'a') as f:
                    f.write(f"{info_string}\n")
                if do_print:
                    print(info_string)

                ## Plot degree distribution.
                if make_plot:
                    degree_hist, degree_bins = jnp.histogram(degree, bins=n_bins, density=True)
                    plot_position = (degree_bins[1:] + degree_bins[:-1]) / 2  # Plot in the middle of the bin
                    plt.plot(plot_position, degree_hist, color='k')
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.xlabel('Degree', fontsize=label_fontsize)
                    plt.xticks(fontsize=tick_fontsize)
                    plt.ylabel('Density', fontsize=label_fontsize)
                    plt.yticks(fontsize=tick_fontsize)
                    savename = get_filename_with_ext(f"{dict_key}_degree_dist_max_{1-percent_edges:.2f}_unconnected", partial=partial, ext='png', folder=figure_folder)
                    plt.savefig(savename, bbox_inches='tight')
                    plt.close()

    with open(info_filename, 'a') as f:
        f.write('\n')
    ## Save binarized data
    with open(bin_filename, 'wb') as f:
        pickle.dump(bin_data, f)
    if do_print:
        print(f"Saved data at {bin_filename}")