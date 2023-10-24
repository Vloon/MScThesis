import numpy as np
from zipfile import ZipFile
import pickle
from jax._src.typing import ArrayLike
from typing import Tuple
import jax.numpy as jnp
import jax
import time
from sklearn.covariance import GraphicalLassoCV
from helper_functions import get_cmd_params, set_GPU, is_valid, get_filename_with_ext
import os

# Create cmd argument list (arg_name, var_name, type, default, nargs[OPT])
arguments = [('-s1', 'subject1', int, 1), # first subject
             ('-sn', 'subjectn', int, 100),  # last subject
             ('-df', 'data_folder', str, 'Data'), # path to get to the data
             ('-tf', 'task_file', str, 'task_list'), # task list name
             ('-of', 'output_file', str, 'processed_data'), # filename to save the processed data WITHOUT EXTENTION
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
task_file = get_filename_with_ext(global_params['task_file'], ext='txt', folder=data_folder)
output_file = global_params['output_file']
partial = global_params['partial']
bpf = global_params['bpf']
do_print = global_params['do_print']
set_GPU(global_params['gpu'])

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
            zip.extractall()
        for task in tasks:
            task_type = 'r' if task in ['REST1', 'REST2'] else 't'
            for enc in phase_enc:
                filename = f"timeseries_{task_type}fMRI_{task}_{enc}.npz"
                keyname = '{}_{}_{}'.format(subject, task, enc)
                timeseries_dic = dict(np.load(filename))
                for series_type in timeseries_dic.keys(): # Contains both non-filtered and bandpass-filtered timeseries.
                    is_bpf = series_type[-3:] == 'bpf'
                    # Only rest tasks have band-pass filtered versions
                    if (task_type == 'r' and is_bpf and incl_bpf) or (task_type == 'r' and not incl_bpf and not is_bpf) or (task_type == 't'):
                        data[keyname] = timeseries_dic[series_type]
                # Delete files from disc again
                os.remove(filename)
    return data

def partial_correlation(x:ArrayLike) -> ArrayLike:
    """
    Returns the (N,N) partial correlations matrix. If A,B is correlated, and B,C, then A,C are also correlated. Partial correlations remove this A,C correlation.
    Graphical Lasso is used to sparsify the matrix first.
    Of course, based on https://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion
    PARAMS:
    x : (N, T) matrix containing timeseries.
    """
    print(f'\tCalculating partial correlation on {x.shape} matrix')
    cov_obj = GraphicalLassoCV(verbose=do_print).fit(x)
    print('\tDone with Graphical Lassoing')
    precision = cov_obj.precision_ # Get sparse precision matrix
    diag = jnp.diagonal(precision)
    div = jnp.sqrt(jnp.outer(diag, diag))
    par_cor = -precision /div
    return par_cor

def get_corr(data:dict, partial:bool=partial) -> dict:
    """
    Takes the upper triangle of the absolute of the correlations of the timeseries data.
    PARAMS:
    data : dictionary with timeseries data as values
    partial : whether to take the partial correlations (filter out the dependencies)
    """
    corr_fun = partial_correlation if partial else lambda x: jnp.corrcoef(x, rowvar=False)
    corr_data = {}
    for key in data.keys():
        ts_data = data[key].T
        if is_valid(ts_data)[0]:
            if do_print:
                print(f"Calculating {key}")
            start = time.time()
            timeseries_corr = corr_fun(ts_data)
            end = time.time()
        else:
            print(f"Invalid data found: {key} at {is_valid(ts_data)[1]}")

        if do_print:
            print(f"Took {end-start:.1f} seconds")
        N = len(timeseries_corr)
        corr_data[key] = timeseries_corr[jnp.triu_indices(N, k=1)]
    return corr_data

def save_corr(data:dict, partial:bool=partial) -> Tuple[bool, list]:
    """
    Runs the same as get_corr, but instead saves each key individually. Use when running partial correlations, since it's SLOW.
    data : dictionary with timeseries data as values
    partial : whether to take the partial correlations (filter out the dependencies)
    """
    corr_fun = partial_correlation if partial else lambda x: jnp.corrcoef(x, rowvar=False)
    invalid_data = []
    all_good = True
    for key in data.keys():
        # Skip the ones that already exist

        partial_filename = get_filename_with_ext(f"{output_file}_{key}", partial=partial, bpf=bpf, folder=data_path)
        if not os.path.isfile(partial_filename):
            ts_data = data[key].T
            if is_valid(ts_data)[0]:
                if do_print:
                    print(f"Calculating {key}")
                start = time.time()
                timeseries_corr = corr_fun(ts_data)
                end = time.time()
            else:
                all_good, idc = is_valid(ts_data)
                invalid_data.append(idc)
                print(f"Invalid data found: {key} at {idc}")
            if do_print:
                print(f"Took {end-start:.1f} seconds")
            N = len(timeseries_corr)
            corr_data = timeseries_corr[jnp.triu_indices(N, k=1)]
            with open(partial_filename, 'wb') as f:
                pickle.dump(corr_data, f)
    return all_good, invalid_data

def all_partial_corrs_exist(data:dict) -> bool:
    """
    Returns whether all the partial correlation files exist based on the timeseries data dictionary keys.
    PARAMS:
    data : dictionary with timeseries data as values
    """
    return np.all([os.path.isfile(get_filename_with_ext(f"{output_file}_{key}", partial=partial, bpf=bpf, folder=data_path)) for key in data.keys()])

def combine_partial_corrs(data:dict) -> None:
    """
    Combines all existing partial correlation files in a single dictionary and saves that dictionary.
    PARAMS:
    data : dictionary with timeseries data as values
    """
    corr_data = {}
    for key in data.keys():
        partial_filename = get_filename_with_ext(f"{output_file}_{key}", partial=partial, bpf=bpf, folder=data_path)
        with open(partial_filename, 'rb') as f:
            corr_data[key] = pickle.load(f)
    output_file = get_big_dict_filename()
    with open(output_file, 'wb') as f:
        pickle.dump(corr_data, f)

# Create list of subjects
subjects = [f'S{i}' for i in range(subject1, subjectn+1)]

# Open task list file
with open(task_file) as tf:
    tasks = tf.readline().rstrip('\n').split(',') # List of tasks
    phase_encs = tf.readline().rstrip('\n').split(',') # List of encodings

# Import the timeseries data into a dictionary. If incl_bpf is true, where possible the band-pass filtered version will be used
data = import_data(subjects, tasks, phase_encs, data_path, incl_bpf=bpf)

if partial:
    if all_partial_corrs_exist(data):
        combine_partial_corrs(data)
    else:
        # Make the partial correlation files
        big_start_time = time.time()
        no_invalid_data, bad_indices = save_corr(data)
        big_end_time = time.time()
        if no_invalid_data:
            print(f"Succesfully created all files in {big_end_time-big_start_time:.1f} seconds")
        else:
            print(f"Found invalid data at {bad_indices}. Ran the rest in {big_end_time-big_start_time:.1f} seconds")
else:
    # Take the correlation matrix from the timeseries
    corr_data = get_corr(data)
    output_file = get_filename_with_ext(output_file, partial, bpf)
    with open(output_file, 'wb') as f:
        pickle.dump(corr_data, f)