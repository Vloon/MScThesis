import numpy as np
import matplotlib.pyplot as plt
from plotting_functions import plot_correlations
from helper_functions import load_observations, get_cmd_params, set_GPU, get_filename_with_ext
from tqdm import tqdm

arguments = [('-df', 'base_data_filename', str, 'processed_data'),  # the most basic version of the filename of the saved data
             ('-tf', 'task_filename', str, 'task_list.txt'), # filename of the list of task names
             ('-of', 'output_folder', str, 'Figures/correlations'), # folder where to dump figures
             ('-N', 'N', int, 164), # number of nodes
             ('-s1', 'subject1', int, 1), # first subject to plot
             ('-sn', 'subjectn', int, 25), # last subject to plot
             ('-cm', 'cmap', str, 'viridis'), # colormap
             ('--partial', 'partial', bool), # whether to use partial correlations
             ('--bpf', 'bpf', bool), # whether to use the band-pass filtered data
             ('--abs', 'abs', bool), # whether to use absolute (partial) correlations
             ('--balanced', 'balanced', bool),  # whether to set the vmin, vmax in a balanced way (0 in the middle)
             ('-gpu', 'gpu', str, ''),  # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]

global_params = get_cmd_params(arguments)
base_data_filename = global_params['base_data_filename']
task_filename = global_params['task_filename']
output_folder = global_params['output_folder']
N = global_params['N']
M = N*(N-1)//2
subject1 = global_params['subject1']
subjectn = global_params['subjectn']
cmap = global_params['cmap']
partial = global_params['partial']
bpf = global_params['bpf']
abs = global_params['abs']
balanced = global_params['balanced']
set_GPU(global_params['gpu'])

# Load data
data_filename = get_filename_with_ext(base_data_filename, partial, bpf)
obs, tasks, encs = load_observations(data_filename, task_filename, subject1, subjectn, M)

for si, n_sub in tqdm(enumerate(range(subject1, subjectn + 1))):
    for ti, task in enumerate(tasks):
        for ei, enc in enumerate(encs):
            plt.figure(figsize=(10,10))
            ax = plt.gca()
            corr = obs[si, ti, ei, :]
            if abs:
                corr = np.abs(corr)
            if balanced:
                vmax = np.max(np.abs(corr))
                vmin = -vmax
            else:
                vmin, vmax = None, None
            ax = plot_correlations(corr, ax, cmap=cmap, add_colorbar = True, vmin=vmin, vmax=vmax)
            # Create a fitting title
            title = f"S{n_sub} {task} {enc}"
            savename = f"{output_folder}/S{n_sub}_{task}_{enc}"
            if abs:
                title = f"{title} absolute"
                savename = f"{savename}_abs"
            if bpf:
                title = f"{title} band-pass filtered"
                savename = f"{savename}_bpf"
            if partial:
                title = f"{title} partial"
                savename = f"{savename}_partial"
            title = f"{title} correlations"
            savename = f"{savename}.png"

            plt.title(title)
            plt.xlabel('Node index')
            plt.ylabel('Node index')

            plt.savefig(savename)
            plt.close()
            

