import numpy as np
import matplotlib as mpl
from matplotlib import patches
import matplotlib.pyplot as plt
import pickle
import jax
import jax.numpy as jnp
from sklearn import svm
import time

from helper_functions import set_GPU, get_cmd_params, get_filename_with_ext, get_safe_folder, open_taskfile

"""
Calling this file runs k-fold cross validation using an SVM to classify the distance matrices of the embeddings to their task. 
Each subject is a seperate observation for the SVM. The distances are calculated previously (get_avg_distances.py) because this takes a LONG time. 
The assumption is that the distances are ordered by subject, then task, so that the class labels are np.tile(np.arange(n_tasks), n_subjects). 

The created file contains (n_folds x n_test x 2) values, where the last axis contains a (predicted_label, true_label) tuple. 
"""

arguments = [('-datfol', 'data_folder', str, 'Data'),  # folder where the data is stored
             ('-distf', 'distance_filename', str, 'avg_distances'), # filename containing the distance matrices
             ('-clf', 'class_label_filename', str, 'class_labels'), # filename containing the class labels for CV (which is the task index)
             ('-ef', 'embedding_folder', str, 'Embeddings'), # base input folder of the embeddings
             ('-ff', 'figure_folder', str, 'Figures/SVM'), # figure output folder
             ('-tf', 'task_filename', str, 'task_list'), # filename of the list of task names
             ('-of', 'output_folder', str, 'Statistics'), # output folder for the pickle file containing the cv results
             ('-cvinfo', 'cv_info_filename', str, 'cv_info'), # Cross validation info file
             ('-np', 'n_particles', int, 2000), # number of particles used in the embedding
             ('-nm', 'n_mcmc_steps', int, 500), # number of mcmc steps used in the embedding
             ('-N', 'N', int, 164), # number of nodes
             ('-s1', 'subject1', int, 1), # first subject to plot
             ('-sn', 'subjectn', int, 100), # last subject to plot
             ('-et', 'edge_type', str, 'con'),  # edge type ('con' or 'bin')
             ('-geo', 'geometry', str, 'hyp'),  # LS geometry ('hyp' or 'euc')
             ('-nfold', 'n_folds', int, 10), # number of folds in k-fold cross validation
             ('-cmap', 'cmap', str, 'bwr'), # colormap for the imshows of the real vs predicted observations
             ('--partial', 'partial', bool), # whether to use partial correlations
             ('--bpf', 'bpf', bool), # whether to use band-pass filtered rs-fMRI data
             ('--plot', 'make_plot', bool), # whether to plot figures
             ('--print', 'do_print', bool), # whether to print
             ('-seed', 'seed', int, 1234), # starting random key
             ('-gpu', 'gpu', str, ''),  # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]

global_params = get_cmd_params(arguments)
set_GPU(global_params['gpu']) ### MUST BE RUN FIRST
n_particles = global_params['n_particles']
embedding_folder = f"{global_params['embedding_folder']}/{n_particles}p{global_params['n_mcmc_steps']}s"
data_folder = global_params['data_folder']
output_folder = global_params['output_folder']
edge_type = global_params['edge_type']
geometry = global_params['geometry']
partial = global_params['partial']
bpf = global_params['bpf']

distance_filename = get_filename_with_ext(f"{edge_type}_{geometry}_{global_params['distance_filename']}", partial, bpf, folder=embedding_folder)
class_label_filename = get_filename_with_ext(f"{edge_type}_{geometry}_{global_params['class_label_filename']}", partial, bpf, folder=embedding_folder)
cv_info_filename = get_filename_with_ext(f"{edge_type}_{geometry}_{global_params['cv_info_filename']}", partial, bpf, ext='txt', folder=output_folder)
figure_folder = get_safe_folder(global_params['figure_folder'])
N = global_params['N']
M = N*(N-1)//2
subject1 = global_params['subject1']
subjectn = global_params['subjectn']
n_subjects = subjectn+1-subject1
n_folds = global_params['n_folds']

cmap = global_params['cmap']
make_plot = global_params['make_plot']
do_print = global_params['do_print']

key = jax.random.PRNGKey(global_params['seed'])

## Load distances, labels and calculate + check cross validation values
task_filename = get_filename_with_ext(global_params['task_filename'], ext='txt', folder=data_folder)
tasks, _ = open_taskfile(task_filename)
n_tasks = len(tasks) # 9
n_embeddings = n_tasks * n_subjects # 9 * 100 = 900
assert n_embeddings%n_folds == 0, f"The total number of embeddings (= n_tasks * n_subjects) must be divisible by the number of folds, but these are {n_embeddings} (= {n_tasks} * {n_subjects}) and {n_folds} respectively"
n_test = n_embeddings//n_folds # 900 / 10 = 90
n_train = n_embeddings - n_test # 900-90 = 810

with open(distance_filename, 'rb') as f:
    avg_distances = pickle.load(f) # -> (n_embeddings x M)

with open(class_label_filename, 'rb') as f: # We could also make the class labels manually here [e.g. np.tile(np.arange(n_tasks), n_subjects)] but so far I've been a bit sloppy with being consistent so I'm leaving this in.
    class_labels = pickle.load(f) # -> (n_embeddings)

assert avg_distances.shape == (n_embeddings, M), f"Shape of avg_distances should be {(n_embeddings, M)} but is {avg_distances.shape}"
assert len(class_labels) == n_embeddings, f"Length of class_labels should be {n_embeddings} but is {len(class_labels)}"
assert n_test >= len(np.unique(class_labels)), f"n_test must be larger or equal to the number of unique classes, but they are {n_test} and {len(np.unique(class_labels))} respectively"

## Shuffle average distances and class labels (in the same way) ## <- for this we assume the data is tiled!
shuffle_avg_distances = np.zeros((n_embeddings, M))
shuffle_class_labels = np.zeros((n_embeddings), dtype=int)
for i in range(n_folds):
    fr, to = i * n_test, (i + 1) * n_test
    key, subkey = jax.random.split(key)
    shuffle_idx = jax.random.permutation(subkey, jnp.arange(n_test), independent=True)
    shuffle_avg_distances[fr:to] = avg_distances[fr:to][shuffle_idx]
    shuffle_class_labels[fr:to] = class_labels[fr:to][shuffle_idx]

## K-fold Cross validation
predictions = np.zeros((n_folds, n_test, 2), dtype=int)
for i in range(n_folds):
    fr, to = i * n_test, (i + 1) * n_test
    ## Seperate data into train and test-set
    test_incl = np.repeat(False, n_embeddings)
    test_incl[fr:to] = True
    train_incl = np.logical_not(test_incl)

    train_set = shuffle_avg_distances[train_incl, :] # -> (n_train x M)
    test_set = shuffle_avg_distances[test_incl, :] # -> (n_test x M)

    train_class_labels = shuffle_class_labels[train_incl] # (n_train x 1) -> (n_train)
    test_class_labels = shuffle_class_labels[test_incl] # (n_test x M) -> (n_test)

    ## Train SVM on training set
    start_time = time.time()
    clf = svm.SVC()
    clf.fit(train_set, train_class_labels)
    end_time = time.time()

    ## Predict class labels on test set
    predicted_labels = clf.predict(test_set)

    predictions[i,:,0] = predicted_labels
    predictions[i,:,1] = test_class_labels
    accuracy = predicted_labels == test_class_labels

    info_string = f"CV iteration {i+1}/{n_folds} ({end_time-start_time:.1f}s): {np.sum(accuracy)}/{n_test} correct"
    if do_print:
        print(info_string)
    with open(cv_info_filename, 'a') as f:
        f.write(f"{info_string}\n")
with open(cv_info_filename, 'a') as f:
    f.write("\n")

basename = f"cv_accuracy_{edge_type}_{geometry}"
filename = get_filename_with_ext(basename, partial, bpf, folder=output_folder)
with open(filename, 'wb') as f:
    pickle.dump(predictions, f)

if make_plot:
    plt.figure(figsize=(20,10))
    plt.imshow(accuracy, cmap=cmap)
    cmap_colors = mpl.colormaps[cmap]
    rpatch = patches.Patch(color=cmap(1.0), label='Correct')
    bpatch = patches.Patch(color=cmap(0.0), label='Incorrect')
    # for (j, i), label in np.ndenumerate(shuffle_class_labels.reshape((n_folds, n_test))):
    #     plt.text(i, j, label, ha='center', va='center')
    plt.legend(handles=[rpatch, bpatch], loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2, fancybox=True, title='Correctness')
    plt.ylabel('CV iteration')
    plt.xlabel('Test')
    filename = get_filename_with_ext(basename, partial, bpf, ext='png', folder=figure_folder)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()