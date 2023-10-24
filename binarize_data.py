import numpy as np
import pickle
from helper_functions import get_cmd_params, set_GPU, is_valid, get_filename_with_ext, load_observations
import os
import matplotlib.pyplot as plt

# Create cmd argument list (arg_name, var_name, type, default, nargs[OPT])
arguments = [('-s1', 'subject1', int, 1), # first subject
             ('-sn', 'subjectn', int, 100),  # last subject
             ('-df', 'data_folder', str, 'Data'), # path to get to the data
             ('-tf', 'task_file', str, 'task_list'), # task list name WITHOUT EXTENTION
             ('-if', 'input_file', str, 'processed_data'), # filename of the processed data WITHOUT EXTENTION
             ('-of', 'output_file', str, 'binary_data'), # filename to save the binary data WITHOUT EXTENTION
             ('-minth', 'min_threshold', float, 1e-3), # minimum threshold to try
             ('-maxth', 'max_threshold', float, 1-1e-3), # maximum threshold to try (inlcuded)
             ('-nth', 'n_thresholds', int, 10), # number of thresholds to try
             ('-N', 'N', int, 164), # number of nodes
             ('--partial', 'partial', bool), # whether to use partial correlations
             ('--bpf', 'bpf', bool), # whether to use the band-pass filtered data
             ('--print', 'do_print', bool), # whether to print cute info
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
task_filename = get_filename_with_ext(global_params['task_file'], partial=partial, bpf=bpf, ext='txt', folder=data_folder)
input_file = global_params['input_file']
data_filename = get_filename_with_ext(input_file, partial, bpf, folder=data_folder)
min_threshold = global_params['min_threshold']
max_threshold = global_params['max_threshold']
n_thresholds = global_params['n_thresholds']
N = global_params['N']
M = N*(N-1)//2
output_file = global_params['output_file']
do_print = global_params['do_print']
set_GPU(global_params['gpu'])

con_obs, tasks, encs = load_observations(data_filename, task_filename, subject1, subjectn, M)

thresholds = np.linspace(min_threshold, max_threshold, n_thresholds)
n_subjects = subjectn+1-subject1

for thi, threshold in enumerate(thresholds):
    print(f"Binarizing at theta={threshold}")
    ### TODO: Number of digits dependent on the thresholds
    bin_filename = get_filename_with_ext(f"{output_file}_th{threshold:.2f}", folder=data_folder)
    bin_data = {}
    ## Re-create observations matrix, but thresholded
    for si, n_sub in enumerate(range(subject1, subjectn + 1)):
        for ti, task in enumerate(tasks):
            for ei, enc, in enumerate(encs):
                dict_key = f'S{n_sub}_{task}_{enc}'
                bin_data[dict_key] = con_obs[si, ti, ei, :] > threshold
    ## Save binarized data
    with open(bin_filename, 'wb') as f:
        pickle.dump(bin_data, f)
            