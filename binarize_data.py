import numpy as np
import pickle
from helper_functions import get_cmd_params, set_GPU, is_valid, get_filename_with_ext, get_safe_folder, load_observations, triu2mat
import os
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Create cmd argument list (arg_name, var_name, type, default, nargs[OPT])
arguments = [('-s1', 'subject1', int, 1), # first subject
             ('-sn', 'subjectn', int, 100),  # last subject
             ('-df', 'data_folder', str, 'Data'), # path to get to the data
             ('-tf', 'task_file', str, 'task_list'), # task list name WITHOUT EXTENTION
             ('-if', 'input_file', str, 'processed_data'), # filename of the processed data WITHOUT EXTENTION
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
             ('-nbins', 'n_bins', int, 20), # number of bins for the log-log degree distribution plot
             ('-palpha', 'plot_alpha', float, .6), # alpha for plotting stuff
             ('-gpu', 'gpu', str, ''), # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]

# Get arguments from CMD
global_params = get_cmd_params(arguments)
subject1 = global_params['subject1']
subjectn = global_params['subjectn']
assert subject1 <= subjectn, f"Subject 1 must be smaller than subject n but they are {subject1} and {subjectn} respectively."
data_folder = global_params['data_folder']
partial = global_params['partial']
bpf = global_params['bpf']
task_filename = get_filename_with_ext(global_params['task_file'], ext='txt', folder=data_folder)
input_file = global_params['input_file']
data_filename = get_filename_with_ext(input_file, partial, bpf, folder=data_folder)
min_threshold = global_params['min_threshold']
max_threshold = global_params['max_threshold']
n_thresholds = global_params['n_thresholds']
N = global_params['N']
M = N*(N-1)//2
output_file = global_params['output_file']
do_print = global_params['do_print']
make_plot = global_params['make_plot']
figure_folder = get_safe_folder(global_params['figure_folder'])
n_bins = global_params['n_bins']
plot_alpha = global_params['plot_alpha']
set_GPU(global_params['gpu'])

## Load continuous data
con_obs, tasks, encs = load_observations(data_filename, task_filename, subject1, subjectn, M)
subjects = range(subject1, subjectn + 1)

def get_i(si:int, ti:int, ei:int, T:int=len(tasks), E:int=len(encs)) -> int:
    """
    Returns the flattened index based on subject, task, encoding indices
    """
    return si * T * E + ti * E + ei 

n_scans = len(subjects) * len(tasks) * len(encs)

## Create correlation range plot
if make_plot:
    correlation_range = np.empty((n_scans, 2))

    for si, n_sub in enumerate(subjects):
        for ti, task in enumerate(tasks):
            for ei, enc in enumerate(encs):
                correlation_range[get_i(si, ti, ei), :] = [np.min(con_obs[si, ti, ei, :]), np.max(con_obs[si, ti, ei, :])]

    correlation_range = np.sort(correlation_range, axis=0)[::-1] ## Sort by correlations high to low

    plt.figure(figsize=(7, 14)) # Nice if it's 1:2 ratio i think?
    for i in range(n_scans):
        plt.hlines(-i, correlation_range[i,0], correlation_range[i,1], color='0.05', alpha=plot_alpha, linewidth=10)
    plt.ylim(-n_scans, 1)
    offset = .05
    plt.xlim(0-offset,1+offset)
    plt.yticks(ticks=[])
    plt.ylabel('Scan')
    plt.xlabel('Correlation')
    savename = get_filename_with_ext(f"correlation_ranges", partial=partial, ext='png', folder=figure_folder)
    plt.savefig(savename)
    plt.close()

thresholds = np.linspace(min_threshold, max_threshold, n_thresholds)
nodes_excluded = np.zeros((n_thresholds, n_scans))

for thi, threshold in enumerate(thresholds):
    if do_print:
       print(f"Binarizing at theta={threshold}")
    ### TODO: Number of digits dependent on the thresholds
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

                ## Plot log-log degree distribution ### HOW MANY BINS???
                if make_plot:
                    degree_hist, degree_bins = jnp.histogram(degree, bins=n_bins, density=True)
                    plot_position = (degree_bins[1:] + degree_bins[:-1]) / 2 # Plot in the middle of the bin
                    plt.plot(plot_position, degree_hist, color='k')
                    plt.xscale('log')
                    plt.yscale('log')
                    savename = get_filename_with_ext(f"degree_dist_th{threshold:.2f}", partial=partial, ext='png', folder=figure_folder)
                    plt.savefig(savename)
                    plt.close()

    ## Save binarized data
    with open(bin_filename, 'wb') as f:
        pickle.dump(bin_data, f)

plt.figure()
for thi, threshold in enumerate(thresholds):
    plt.scatter(np.repeat(threshold, n_scans), 1-nodes_excluded[thi]/N, c='k', s=1, alpha=plot_alpha)
offset = .05
plt.hlines(.95, min_threshold-offset, max_threshold+offset, color='r', linestyle='dashed')
plt.xlabel('Threshold')
plt.ylabel('Percentage nodes included')
savename = get_filename_with_ext(f"nodes_included", partial=partial, ext='png', folder=figure_folder)
plt.savefig(savename)
plt.close()