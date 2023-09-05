import numpy as np
from zipfile import ZipFile
import pickle
from jax._src.typing import ArrayLike
import jax.numpy as jnp
import jax
from helper_functions import get_cmd_params, set_GPU, is_valid
import os

# Create cmd argument list (arg_name, var_name, type, default, nargs[OPT])
arguments = [('-ns', 'n_subjects', int, 100), # number of subjects
             ('-d', 'data_path', str, './Data'), # path to get to the data
             ('-tf', 'task_file', str, 'task_list.txt'), # task list name
             ('-of', 'output_file', str, 'processed_data'), # filename to save the processed data WITHOUT EXTENTION
             ('--partial', 'partial', bool), # whether to use partial correlations
             ('-gpu', 'gpu', str, ''), # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]

# Get arguments from CMD
global_params = get_cmd_params(arguments)
n_subjects = global_params['n_subjects']
data_path = global_params['data_path']
task_file = global_params['task_file']
output_file = global_params['output_file']
partial = global_params['partial']
set_GPU(global_params['gpu'])

def partial_correlation(x:ArrayLike) -> ArrayLike:
    """
    Returns the (N,N) partial correlations matrix. If A,B is correlated, and B,C, then A,C are also correlated. Partial correlations remove this A,C correlation.
    Of course, based on https://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion
    PARAMS:
    x : (N, T) matrix containing timeseries.
    """
    precision = jnp.linalg.inv(jnp.corrcoef(x, rowvar=False))
    diag = jnp.diagonal(precision)
    div = jnp.sqrt(jnp.outer(diag, diag))
    par_cor = - precision /div
    return par_cor

def import_data(subjects:ArrayLike=[], tasks:ArrayLike=[], phase_enc:ArrayLike=[], data_path:str=data_path, incl_bpf:bool=False) -> dict:
    """
    Imports the data and saves it in a dict with keys 'S{n_sub}_{task}_{enc}'
    PARAMS:
    subjects : list of subjects.
    tasks : list of tasks
    phase_enc : list of phase encoding directions
    data_path : path to the folder where the subjects' zip files can be found
    incl_bpf : If timeseries that have a band-pass filtered version, include this one if bpf=True.
    """
    data = {}
    for subject in subjects:
        with ZipFile('{}/{}.zip'.format(data_path, subject), 'r') as zip:
            zip.extractall() # This overwrites the previous files with same name.
        
        for task in tasks:
            task_type = 'r' if task in ['REST1', 'REST2'] else 't'
            for enc in phase_enc:
                filename = 'timeseries_{}fMRI_{}_{}.npz'.format(task_type,task,enc)
                keyname = '{}_{}_{}'.format(subject, task, enc)
                timeseries_dic = dict(np.load(filename))
                for series_type in timeseries_dic.keys(): ## Contains both non-filtered and bandpass-filtered timeseries.
                    is_bpf = series_type[-3:] == 'bpf'
                    if ('{}_bpf'.format(keyname) not in data and incl_bpf) or (not incl_bpf and not is_bpf):
                        data[keyname] = timeseries_dic[series_type]
                # Delete files from disc again
                os.remove(filename)
    return data

def get_abs_corr(data:dict, partial:bool=False) -> dict:
    """
    Takes the upper triangle of the absolute of the correlations of the timeseries data.
    PARAMS:
    data : dictionary with timeseries data as values
    N : number of nodes in the network
    partial : whether to take the partial correlations (filter out the dependencies)
    """
    corr_fun = partial_correlation if partial else lambda x: jnp.corrcoef(x, rowvar=False)
    abs_corr_data = {}
    for key in data.keys():
        timeseries_corr = corr_fun(data[key].T)
        N = len(timeseries_corr)
        abs_corr_data[key] = jnp.absolute(timeseries_corr[jnp.triu_indices(N, k=1)])
    return abs_corr_data

# Create list of subjects
subjects = [f'S{i+1}' for i in range(n_subjects)]

# Open task list file
with open(task_file) as tf:
    tasks = tf.readline().rstrip('\n').split(',') # List of tasks
    phase_encs = tf.readline().rstrip('\n').split(',') # List of encodings

data = import_data(subjects, tasks, phase_encs, data_path, incl_bpf=False)
data_bpf = import_data(subjects, tasks, phase_encs, data_path, incl_bpf=True)

data = get_abs_corr(data, partial)
data_bpf = get_abs_corr(data_bpf, partial)

partial_txt = '_partial' if partial else ''
output_file_nf = f'{output_file}{partial_txt}.pkl'
output_file_bpf = f'{output_file}{partial_txt}_bpf.pkl'

with open(output_file_nf, 'wb') as f:
    pickle.dump(data, f)

with open(output_file_bpf, 'wb') as f:
    pickle.dump(data_bpf, f)