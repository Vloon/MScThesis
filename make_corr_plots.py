"""
This file makes plot of the correlation matrices of the specified subjects and tasks
"""

import numpy as np
import matplotlib.pyplot as plt
from plotting_functions import plot_correlations
from helper_functions import load_observations, get_cmd_params, set_GPU, get_filename_with_ext, get_safe_folder, triu2mat

### Create cmd argument list (arg_name, var_name, type, default[OPT], nargs[OPT]).
###  - arg_name is the name of the argument in the command line.
###  - var_name is the name of the variable in the returned dictionary (which we re-use as variable name here).
###  - type is the data-type of the variable.
###  - default is the default value it takes if nothing is passed to the command line. This argument is only optional if type is bool, where the default is always False.
###  - nargs is the number of arguments, where '?' (default) is 1 argument, '+' concatenates all arguments to 1 list. This argument is optional.
arguments = [('-bdf', 'base_data_filename', str, 'processed_data_downsampled_evenly_spaced'),  # the most basic version of the filename of the saved data
             ('-tf', 'task_filename', str, 'task_list'), # filename of the list of task names
             ('-of', 'output_folder', str, 'Figures/correlations'), # folder where to dump figures
             ('-df', 'data_folder', str, 'Data'), # path to get to the data
             ('-lab', 'label_location', str, 'Figures/lobelabels.npz'),  # file location of the labels
             ('-N', 'N', int, 164), # number of nodes
             ('-s1', 'subject1', int, 1), # first subject to plot
             ('-sn', 'subjectn', int, 25), # last subject to plot
             ('-cm', 'cmap', str, 'bwr'), # colormap
             ('--partial', 'partial', bool), # whether to use partial correlations
             ('--bpf', 'bpf', bool), # whether to use the band-pass filtered data
             ('--abs', 'abs', bool), # whether to use absolute (partial) correlations
             ('--print', 'do_print', bool), # whether to print
             ('--bar', 'add_bar', bool), # whether to add a colorbar
             ('-vvals', 'v_values', tuple, (-1,1), '+'), # minimum plotting value
             ('-lfs', 'label_fontsize', float, 20),  # fontsize of labels (and legend)
             ('-tfs', 'tick_fontsize', float, 16),  # fontsize of the tick labels
             ('-wsz', 'wrapsize', float, 20),  # wrapped text width
             ('-gpu', 'gpu', str, ''),  # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]

## Get arguments from command line.
global_params = get_cmd_params(arguments)
set_GPU(global_params['gpu']) ## MUST BE RUN FIRST
data_folder = global_params['data_folder']
partial = global_params['partial']
bpf = global_params['bpf']
abs = global_params['abs']
task_filename = get_filename_with_ext(global_params['task_filename'], ext='txt', folder=data_folder)
data_filename = get_filename_with_ext(global_params['base_data_filename'], partial=partial, bpf=bpf, folder=data_folder)
output_folder = get_safe_folder(global_params['output_folder'])
N = global_params['N']
M = N*(N-1)//2
subject1 = global_params['subject1']
subjectn = global_params['subjectn']
cmap = global_params['cmap']
label_location = global_params['label_location']
label_fontsize = global_params['label_fontsize']
tick_fontsize = global_params['tick_fontsize']
wrapsize = global_params['wrapsize']
vmin, vmax = global_params['v_values']
do_print = global_params['do_print']
add_bar = global_params['add_bar']

## Load observations and tasks
obs, tasks, encs = load_observations(data_filename, task_filename, subject1, subjectn, M)

## Load labels for lobe ordering
label_data = np.load(label_location)
plt_labels = label_data[label_data.files[0]]
if len(plt_labels) != N:
    plt_labels = None

## Get the lobe ordering
lobes = [l[1].split(';')[0] for l in plt_labels]
idc = np.argsort(lobes)
ordered = np.sort(lobes)
uordered = np.unique(ordered)
first_idc = [list(ordered).index(ul) for ul in uordered]

for si, n_sub in enumerate(range(subject1, subjectn + 1)):
    for ti, task in enumerate(tasks):
        for ei, enc in enumerate(encs):
            if do_print:
                print(f"Plotting S{n_sub}_{task}_{enc}")
            plt.figure(figsize=(10,10))

            ## Get the (N x N) correlation matrix
            _corr = obs[si, ti, ei, :]
            if abs:
                _corr = np.abs(corr)
            corr = triu2mat(_corr)

            ## Sort correlation matrix by lobe
            c_sorted = np.zeros((N, N))
            for row in range(N):
                c_sorted[row] = corr[idc[row]][idc]

            ## Plot the sorted correlations
            plt.imshow(c_sorted, cmap=cmap, vmin=vmin, vmax=vmax)

            plt.yticks(ticks=first_idc, labels=uordered, fontsize=tick_fontsize)
            plt.xticks(ticks=first_idc, labels=uordered, rotation=75, fontsize=tick_fontsize, ha='right')
            if add_bar:
                cbar = plt.colorbar()
                cbar.ax.tick_params(labelsize=tick_fontsize)
            ax = plt.gca()
            ax.xaxis.tick_top()

            abs_txt = '_abs' if abs else ''
            savefile = get_filename_with_ext(f"S{n_sub}_{task}_{enc}{abs_txt}", partial=partial, bpf=bpf, ext='png', folder=output_folder)
            plt.savefig(savefile, bbox_inches='tight')
            plt.close()
            

