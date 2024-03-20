"""
Calling this file calculates the upper triangle of the correlations matrix (=continuous data) for all subjects/tasks and saves it for later use.
"""

## Basics
import numpy as np
import pickle
import jax

## Self-made functions
from helper_functions import set_GPU, get_cmd_params, get_filename_with_ext, get_safe_folder, open_taskfile, get_attribute_from_trace, load_observations

### Create cmd argument list (arg_name, var_name, type, default[OPT], nargs[OPT]).
###  - arg_name is the name of the argument in the command line.
###  - var_name is the name of the variable in the returned dictionary (which we re-use as variable name here).
###  - type is the data-type of the variable.
###  - default is the default value it takes if nothing is passed to the command line. This argument is only optional if type is bool, where the default is always False.
###  - nargs is the number of arguments, where '?' (default) is 1 argument, '+' concatenates all arguments to 1 list. This argument is optional.
arguments = [('-datfol', 'data_folder', str, 'Data'),  # folder where the data is stored
             ('-conbdf', 'con_base_data_filename', str, 'processed_data_downsampled_evenly_spaced'), # the most basic version of the filename of the continuous saved data
             ('-binbdf', 'bin_base_data_filename', str, 'binary_data_downsampled_evenly_spaced_max_0.05unconnected'), # the most basic version of the filename of the binary saved data
             ('-ef', 'embedding_folder', str, 'Embeddings'),  # base input folder of the embeddings
             ('-tf', 'task_filename', str, 'task_list'), # filename of the list of task names
             ('-np', 'n_particles', int, 2000), # number of particles used in the embedding
             ('-nm', 'n_mcmc_steps', int, 500), # number of mcmc steps used in the embedding
             ('-N', 'N', int, 164), # number of nodes
             ('-s1', 'subject1', int, 1), # first subject to plot
             ('-sn', 'subjectn', int, 100), # last subject to plot
             ('-et', 'edge_type', str, 'con'),  # edge type ('con' or 'bin')
             ('-geo', 'geometry', str, 'hyp'),  # LS geometry ('hyp' or 'euc')
             ('--partial', 'partial', bool), # whether to use partial correlations
             ('--bpf', 'bpf', bool), # whether to use band-pass filtered rs-fMRI data
             ('--onerest', 'one_rest', bool), # whether to give both REST tasks the same class label
             ('-gpu', 'gpu', str, ''),  # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]

## Get arguments from command line.
global_params = get_cmd_params(arguments)
set_GPU(global_params['gpu']) ## MUST BE RUN FIRST
data_folder = global_params['data_folder']
edge_type = global_params['edge_type']
geometry = global_params['geometry']
embedding_folder = f"{global_params['embedding_folder']}/{global_params['n_particles']}p{global_params['n_mcmc_steps']}s"
base_data_filename = global_params['bin_base_data_filename'] if edge_type == 'bin' else global_params['con_base_data_filename']
task_filename = global_params['task_filename']
N = global_params['N']
M = N*(N-1)//2
subject1 = global_params['subject1']
subjectn = global_params['subjectn']
partial = global_params['partial']
bpf = global_params['bpf']
one_rest = global_params['one_rest']

## Load continuous data
data_filename = get_filename_with_ext(base_data_filename, partial, bpf, folder=data_folder)
task_filename = get_filename_with_ext(task_filename, ext='txt', folder=data_folder)
obs, tasks, encs = load_observations(data_filename, task_filename, subject1, subjectn, M)
subjects = range(subject1, subjectn + 1)

n_tasks = len(tasks)
n_subjects = subjectn+1-subject1
n_embeddings = n_tasks * n_subjects

## Calculate average correlations
avg_correlations = np.mean(obs, axis=2).reshape((n_embeddings, M))

## Re-name the class-labels for REST2 to that of REST1 if we want the two resting tasks to be counted as the same task
class_labels = np.tile(np.arange(n_tasks), n_subjects)
if one_rest:
    class_labels[class_labels == 1] = 0

## Save average correlations and class labels
one_rest_txt = '_one_rest' if one_rest else ''
avg_correlations_filename = get_filename_with_ext(f"{edge_type}_{geometry}_avg_correlations{one_rest_txt}", partial, bpf, folder=embedding_folder)
with open(avg_correlations_filename, 'wb') as f:
    pickle.dump(avg_correlations, f)
class_label_filename = get_filename_with_ext(f"{edge_type}_{geometry}_corr_class_labels{one_rest_txt}", partial, bpf, folder=embedding_folder)
with open(class_label_filename, 'wb') as f:
    pickle.dump(class_labels, f)
print(f"Saved correlations to {avg_correlations_filename}")
print(f"Saved class labels to {class_label_filename}")





