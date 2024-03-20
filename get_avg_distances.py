"""
Calling this file calculates the upper triangle of the distance matrix for all embeddings and saves it for later use. This script exists because this takes a long time to calculate.
"""

## Basics
import numpy as np
import pickle
import jax

## Self-made functions
from helper_functions import set_GPU, get_cmd_params, get_filename_with_ext, get_safe_folder, open_taskfile, get_attribute_from_trace
from binary_euclidean_LSM import get_det_params as bin_euc_det_params
from binary_hyperbolic_LSM import get_det_params as bin_hyp_det_params
from continuous_euclidean_LSM import get_det_params as con_euc_det_params
from continuous_hyperbolic_LSM import get_det_params as con_hyp_det_params

### Create cmd argument list (arg_name, var_name, type, default[OPT], nargs[OPT]).
###  - arg_name is the name of the argument in the command line.
###  - var_name is the name of the variable in the returned dictionary (which we re-use as variable name here).
###  - type is the data-type of the variable.
###  - default is the default value it takes if nothing is passed to the command line. This argument is only optional if type is bool, where the default is always False.
###  - nargs is the number of arguments, where '?' (default) is 1 argument, '+' concatenates all arguments to 1 list. This argument is optional.
arguments = [('-datfol', 'data_folder', str, 'Data'),  # folder where the data is stored
             ('-conbdf', 'con_base_data_filename', str, 'processed_data_downsampled_evenly_spaced'), # the most basic version of the filename of the continuous saved data
             ('-binbdf', 'bin_base_data_filename', str, 'binary_data_downsampled_evenly_spaced_max_0.05unconnected'), # the most basic version of the filename of the binary saved data
             ('-ef', 'embedding_folder', str, 'Embeddings'), # base input folder of the embeddings
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
n_particles = global_params['n_particles']
embedding_folder = f"{global_params['embedding_folder']}/{n_particles}p{global_params['n_mcmc_steps']}s"
data_folder = global_params['data_folder']
edge_type = global_params['edge_type']
geometry = global_params['geometry']
base_data_filename = global_params['bin_base_data_filename'] if edge_type == 'bin' else global_params['con_base_data_filename']
task_filename = global_params['task_filename']
N = global_params['N']
M = N*(N-1)//2
subject1 = global_params['subject1']
subjectn = global_params['subjectn']
partial = global_params['partial']
bpf = global_params['bpf']
one_rest = global_params['one_rest']

## Define a number of variables based on geometry or edge type
det_params_dict = {'bin_euc':bin_euc_det_params,
                   'bin_hyp':bin_hyp_det_params,
                   'con_euc':con_euc_det_params,
                   'con_hyp':con_hyp_det_params}
det_params_func = det_params_dict[f"{edge_type}_{geometry}"]
latpos = '_z' if geometry == 'hyp' else 'z'

## Load tasks
task_filename = get_filename_with_ext(task_filename, ext='txt', folder=data_folder)
tasks, _  = open_taskfile(task_filename)
n_tasks = len(tasks)
n_subjects = subjectn+1-subject1
n_embeddings = n_tasks * n_subjects

## Calculate the distances and then average them over particles
avg_distances = np.zeros((n_embeddings, M))
class_labels = np.zeros((n_embeddings), dtype=int)
for si, n_sub in enumerate(range(subject1, subjectn+1)):
    print(f"Running {n_sub}/{subjectn+1-subject1} subjects")
    for ti, task in enumerate(tasks):
        ## We see each subject as a seperate observation of the embedding. This means we of course disregard any inter-subject variability in our model.
        embedding_filename = get_filename_with_ext(f"{edge_type}_{geometry}_S{n_sub}_{task}_embedding_{base_data_filename}", partial, bpf, folder=embedding_folder)
        with open(embedding_filename, 'rb') as f:
            embedding = pickle.load(f)
        idx = si*n_tasks + ti
        avg_distances[idx,:] = np.mean(get_attribute_from_trace(embedding.particles[latpos], det_params_func, 'd'), axis=0)
        class_labels[idx] = ti-1 if one_rest and task == 'REST2' else ti

## Save the average distances and class labels
one_rest_txt = '_one_rest' if one_rest else ''
avg_distances_filename = get_filename_with_ext(f"{edge_type}_{geometry}_avg_distances{one_rest_txt}", partial, bpf, folder=embedding_folder)
with open(avg_distances_filename, 'wb') as f:
    pickle.dump(avg_distances, f)
class_label_filename = get_filename_with_ext(f"{edge_type}_{geometry}_class_labels{one_rest_txt}", partial, bpf, folder=embedding_folder)
with open(class_label_filename, 'wb') as f:
    pickle.dump(class_labels, f)
print(f"Saved distances to {avg_distances_filename}")
print(f"Saved class labels to {class_label_filename}")