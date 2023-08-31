import numpy as np
from zipfile import ZipFile
import pickle
from jax._src.typing import ArrayLike
from helper_functions import get_cmd_params
import os

def import_data(subjects:ArrayLike=[], tasks:ArrayLike=[], phase_enc:ArrayLike=[], data_path:str='./', incl_bpf:bool=False) -> dict:
    """
    Imports the data and saves it in a dict with keys 'S{n_sub}_{task}_{enc}'
    PARAMS:
    subjects : list of subjects.
    tasks : list of tasks
    phase_enc : list of phase encoding directions
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
                for series_type in timeseries_dic.keys(): ## Allows bandpass-filtered timeseries.
                    if series_type[-3:] == 'bpf':
                        new_keyname = '{}_bpf'.format(keyname)
                        is_bpf = True
                    else:
                        new_keyname =  keyname
                        is_bpf = False
                    if ('{}_bpf'.format(new_keyname) not in data and incl_bpf) or (not incl_bpf and not is_bpf):
                        data[new_keyname] = timeseries_dic[series_type]
                # Delete files from disc again
                os.remove(filename)
        print(f'Succesfully processed {subject}')
    return data

def get_abs_corr(data:dict) -> dict:
    """
    Takes the upper triangle of the absolute of the correlations of the timeseries data.
    PARAMS:
    data : dictionary with timeseries data as values
    """
    for key in data.keys():
        timeseries_corr = np.corrcoef(data[key])
        N = len(timeseries_corr)
        data[key] = np.abs(timeseries_corr)[np.triu_indices(N, k=1)]
    return data


# Create cmd argument list (arg_name, var_name, type, default, nargs[OPT])
arguments = [('-ns', 'n_subjects', int, 100), # number of subjects
             ('-d', 'data_path', str, './Data'), # path to get to the data
             ('-tf', 'task_file', str, 'task_list.txt'), # task list name
             ('-f', 'filename', str, 'processed_data'), # filename to save the processed data WITHOUT EXTENTION
             ]

# Get arguments from CMD
global_params = get_cmd_params(arguments)
n_subjects = global_params['n_subjects']
data_path = global_params['data_path']
task_file = global_params['task_file']
filename = global_params['filename']

# Create list of subjects
subjects = [f'S{i+1}' for i in range(n_subjects)]

# Open task list file
with open(task_file) as tf:
    tasks = tf.readline().rstrip('\n').split(',') # List of tasks
    phase_encs = tf.readline().rstrip('\n').split(',') # List of encodings

data = import_data(subjects, tasks, phase_encs, data_path, incl_bpf=False)
data_bpf = import_data(subjects, tasks, phase_encs, data_path, incl_bpf=True)

data = get_abs_corr(data)
data_bpf = get_abs_corr(data_bpf)

with open(filename+'.pkl', 'wb') as f:
    pickle.dump(data, f)

with open(filename+'_bpf.pkl', 'wb') as f:
    pickle.dump(data_bpf, f)
