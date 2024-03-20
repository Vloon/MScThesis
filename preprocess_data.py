"""
Calling this file performs the preprocessing of the raw HCP data.
"""

import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
import pickle
from jax._src.typing import ArrayLike
from typing import Tuple
import jax.numpy as jnp
import jax
import time
import os
from sklearn.covariance import GraphicalLassoCV
from helper_functions import get_cmd_params, set_GPU, is_valid, get_filename_with_ext, get_safe_folder, open_taskfile
from plotting_functions import plot_timeseries

### Create cmd argument list (arg_name, var_name, type, default[OPT], nargs[OPT]).
###  - arg_name is the name of the argument in the command line.
###  - var_name is the name of the variable in the returned dictionary (which we re-use as variable name here).
###  - type is the data-type of the variable.
###  - default is the default value it takes if nothing is passed to the command line. This argument is only optional if type is bool, where the default is always False.
###  - nargs is the number of arguments, where '?' (default) is 1 argument, '+' concatenates all arguments to 1 list. This argument is optional.
arguments = [('-s1', 'subject1', int, 1), # first subject
             ('-sn', 'subjectn', int, 100),  # last subject
             ('-outfol', 'output_folder', str, 'Data'), # where to save the data
             ('-infol', 'input_folder', str, 'Data/RAW'), # where the raw .zip files are stored
             ('-tf', 'task_file', str, 'task_list'), # task list name
             ('-of', 'output_file', str, 'processed_data'), # filename to save the processed data, without extension
             ('-ff', 'figure_folder', str, 'Figures/timeseries'), # figure output folder
             ('--partial', 'partial', bool), # whether to use partial correlations
             ('--bpf', 'bpf', bool), # whether to use the band-pass filtered data
             ('--print', 'do_print', bool), # whether to print cute info
             ('--plot', 'make_plot', bool), # whether to make plots
             ('-yoffset', 'y_offset', float, 0.1), # yoffset for the timeseries plot
             ('-lfs', 'label_fontsize', float, 20),  # fontsize of labels (and legend)
             ('-tfs', 'tick_fontsize', float, 16),  # fontsize of the tick labels
             ('-wsz', 'wrapsize', float, 20),  # wrapped text width
             ('--downsample', 'downsample', bool), # whether to equalize the lengths of the timeseries
             ('-dsmethod', 'downsample_method', str, 'evenly_spaced'), # which method to use to downsample the signals
             ('-gpu', 'gpu', str, ''), # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]

## Get arguments from command line.
global_params = get_cmd_params(arguments)
set_GPU(global_params['gpu']) ## MUST BE RUN FIRST
subject1 = global_params['subject1']
subjectn = global_params['subjectn']
assert subject1 <= subjectn, f"Subject 1 must be smaller than subject n but they are {subject1} and {subjectn} respectively."
output_folder = get_safe_folder(global_params['output_folder'])
input_folder = global_params['input_folder']
task_file = get_filename_with_ext(global_params['task_file'], ext='txt', folder=output_folder)
figure_folder = get_safe_folder(global_params['figure_folder'])
partial = global_params['partial']
if partial: # Assert the folder exists but don't overwrite the output folder
    p_output_folder = get_safe_folder(f"{output_folder}/partial") 
bpf = global_params['bpf']
do_print = global_params['do_print']
make_plot = global_params['make_plot']
y_offset = global_params['y_offset']
downsample = global_params['downsample']
downsample_method = global_params['downsample_method']
downsample_txt = f"_downsampled_{downsample_method}" if downsample else ''
output_file = f"{global_params['output_file']}{downsample_txt}"
label_fontsize = global_params['label_fontsize']
tick_fontsize = global_params['tick_fontsize']
wrapsize = global_params['wrapsize']

def import_data(subjects:ArrayLike=[], tasks:ArrayLike=[], phase_enc:ArrayLike=[], data_path:str=input_folder, incl_bpf:bool=False) -> dict:
    """
    Imports the data and saves it in a dictionary with keys 'S{n_sub}_{task}_{enc}'
    PARAMS:
    subjects : list of subjects.
    tasks : list of tasks
    phase_enc : list of phase encoding directions
    data_path : path to the folder where the subjects' zip files can be found
    incl_bpf : If timeseries that have a band-pass filtered version, include this one if bpf=True.
    """
    data = {}
    for subject in subjects:
        ## Unzip data
        with ZipFile(f"{data_path}/{subject}.zip", 'r') as zip:
            zip.extractall()
        for task in tasks:
            task_type = 'r' if task in ['REST1', 'REST2'] else 't' # Is the task a resting scan?
            for enc in phase_enc:
                filename = f"timeseries_{task_type}fMRI_{task}_{enc}.npz"
                keyname = f"{subject}_{task}_{enc}"
                timeseries_dic = dict(np.load(filename))
                for series_type in timeseries_dic.keys(): # Contains both non-filtered and bandpass-filtered timeseries.
                    is_bpf = series_type[-3:] == 'bpf'
                    ## Only rest tasks have band-pass filtered versions, therefore only save those (when desired)
                    if (task_type == 'r' and is_bpf and incl_bpf) or (task_type == 'r' and not incl_bpf and not is_bpf) or (task_type == 't'):
                        data[keyname] = timeseries_dic[series_type]
                ## Delete the unzipped files from disc again
                os.remove(filename)
    return data

def partial_correlation(x:ArrayLike) -> ArrayLike:
    """
    Returns the (N,N) partial correlations matrix. If A,B is correlated, and B,C, then A,C are also correlated. Partial correlations remove this A,C correlation.
    Graphical Lasso is used to sparsify the matrix first.
    Based partially on https://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion
    PARAMS:
    x : (N, T) timeseries matrix.
    """
    if do_print:
        print(f'\tCalculating partial correlation on {x.shape} matrix')
    ## Perform graphical lasso
    cov_obj = GraphicalLassoCV(verbose=do_print).fit(x)
    if do_print:
        print('\tDone with Graphical Lassoing')
    ## Get sparse precision matrix, and calculate the partial correlation based on it
    precision = cov_obj.precision_
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
    ## Define the correct correlation function
    corr_fun = partial_correlation if partial else lambda x: jnp.corrcoef(x, rowvar=False)
    corr_data = {}
    for key in data.keys():
        ts_data = data[key].T
        if is_valid(ts_data)[0]:
            start = time.time()
            timeseries_corr = corr_fun(ts_data)
            end = time.time()
        else:
            print(f"Invalid data found: {key} at {is_valid(ts_data)[1]}")
        N = len(timeseries_corr)
        corr_data[key] = timeseries_corr[jnp.triu_indices(N, k=1)]
    return corr_data

def save_corr(data:dict, partial:bool=partial) -> Tuple[bool, list]:
    """
    Does the same as the get_corr function, but instead saves each key individually. Use when running partial correlations, since that is time consuming.
    As a double check, also finds NaN values in the timeseries if they exist and warns the user.
    PARAMS:
    data : dictionary with timeseries data as values
    partial : whether to take the partial correlations (filter out the dependencies)
    """
    ## Define the correct correlation function
    corr_fun = partial_correlation if partial else lambda x: jnp.corrcoef(x, rowvar=False)
    invalid_data = []
    all_good = True
    _output_folder = p_output_folder if partial else output_folder
    for key in data.keys():
        partial_filename = get_filename_with_ext(f"{output_file}_{key}", partial=partial, bpf=bpf, folder=_output_folder)
        if not os.path.isfile(partial_filename):
            ts_data = data[key].T
            if is_valid(ts_data)[0]:
                ## Caculate the (partial) correlations
                start = time.time()
                timeseries_corr = corr_fun(ts_data)
                end = time.time()
            else:
                ## Sanity check
                _all_good, idc = is_valid(ts_data)
                invalid_data.append(idc)
                all_good = np.logical_and(all_good, _all_good)
                print(f"Invalid data found: {key} at {idc}")
            N = len(timeseries_corr)
            corr_data = timeseries_corr[jnp.triu_indices(N, k=1)]
            ## Save the (partial) correlations
            with open(partial_filename, 'wb') as f:
                pickle.dump(corr_data, f)
    return all_good, invalid_data

def all_partial_corrs_exist(data:dict) -> bool:
    """
    Returns whether all the partial correlation files exist based on the timeseries data dictionary keys.
    PARAMS:
    data : dictionary with timeseries data as values
    """
    return np.all([os.path.isfile(get_filename_with_ext(f"{output_file}_{key}", partial=partial, bpf=bpf, folder=p_output_folder)) for key in data.keys()])

def combine_partial_corrs(data:dict) -> None:
    """
    Combines all existing partial correlation files in a single dictionary and saves that dictionary.
    PARAMS:
    data : dictionary with timeseries data as values
    """
    corr_data = {}
    for key in data.keys():
        partial_filename = get_filename_with_ext(f"{output_file}_{key}", partial=partial, bpf=bpf, folder=p_output_folder)
        with open(partial_filename, 'rb') as f:
            corr_data[key] = pickle.load(f)
    _output_file = get_filename_with_ext(output_file, partial, bpf, folder=output_folder)
    with open(_output_file, 'wb') as f:
        pickle.dump(corr_data, f)
    if do_print:
        print(f"Combined partial data and saved it at {_output_file}")

## Create list of subjects
subjects = [f'S{i}' for i in range(subject1, subjectn+1)]

## Open task and encoding file
tasks, phase_encs = open_taskfile(task_file)

# Import the timeseries data into a dictionary. If incl_bpf is true, where possible the band-pass filtered version will be used
ts_data = import_data(subjects, tasks, phase_encs, input_folder, incl_bpf=bpf)

if downsample:
    ## Calculate the minimum length
    _ts_lengths = {key:ts.shape[1] for key, ts in ts_data.items()}
    ts_lengths = [ts.shape[1] for ts in ts_data.values()]
    min_length = np.min(ts_lengths)
    if do_print:
        print(f"Minimum timeseries length is {min_length}")
        print(f"All lengths are {_ts_lengths}")
    for key, timeseries in ts_data.items():
        ## Downsample the timesries
        if downsample_method == 'evenly_spaced':
            ## Take linearly space samples
            ts_data[key] = timeseries[:, np.linspace(0, timeseries.shape[1]-1, min_length, dtype=int)]
        elif downsample_method == 'cut_off':
            ## Take only the first samples
            ts_data[key] = timeseries[:, :min_length]

## Plot the timeseries
if make_plot:
    for subject in subjects:
        for task in tasks:
            for enc in phase_encs:
                key = f"{subject}_{task}_{enc}"
                if do_print:
                    print(f"Plotting timeseries for {key}")
                timeseries = ts_data[key]
                plt.figure(figsize=(20, 30))
                ax = plt.gca()
                plt.xlabel('Sample index', fontsize=label_fontsize)
                plt.ylabel('Brain region', fontsize=label_fontsize)
                ax = plot_timeseries(timeseries, y_offset, ax)
                plt.yticks(fontsize=tick_fontsize)
                plt.xticks(fontsize=tick_fontsize)
                downsample_txt = f"_downsampled_{downsample_method}" if downsample else ''
                savefile = get_filename_with_ext(f"timeseries_{key}{downsample_txt}", partial=partial, bpf=bpf, ext='png', folder=figure_folder)
                plt.savefig(savefile, bbox_inches='tight')
                plt.close()

## For dealing with partial correlations, we first check if all individual partial correlation files exist, and if so combine them. If not, we calculate the partial correlations while saving the invidual files.
if partial:
    if all_partial_corrs_exist(ts_data):
        combine_partial_corrs(ts_data)
    else:
        ## Make the partial correlation files
        big_start_time = time.time()
        no_invalid_data, bad_indices = save_corr(ts_data)
        big_end_time = time.time()
        if no_invalid_data and do_print:
            print(f"Succesfully created all files in {big_end_time-big_start_time:.1f} seconds")
        elif do_print:
            print(f"Found invalid data at {bad_indices}. Ran the rest in {big_end_time-big_start_time:.1f} seconds")
else:
    ## Take the (regular) correlation matrix from the timeseries
    corr_data = get_corr(ts_data)
    output_file = get_filename_with_ext(output_file, partial, bpf, folder=output_folder)
    with open(output_file, 'wb') as f:
        pickle.dump(corr_data, f)
    if do_print:
        print(f"Saved data at {output_file}")
