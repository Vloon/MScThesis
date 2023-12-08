import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
import os

from helper_functions import get_filename_with_ext, load_observations, get_cmd_params, set_GPU, get_safe_folder, get_attribute_from_trace, lorentz_to_poincare
from plotting_functions import plot_posterior
from binary_euclidean_LSM import get_det_params as bin_euc_det_params
from binary_hyperbolic_LSM import get_det_params as bin_hyp_det_params
from continuous_euclidean_LSM import get_det_params as con_euc_det_params
from continuous_hyperbolic_LSM import get_det_params as con_hyp_det_params

arguments = [('-overwritedf', 'overwrite_data_filename', str, None),  # if used, it overwrites the default filename
             ('-datfol', 'data_folder', str, 'Data'),  # folder where the data is stored
             ('-conbdf', 'con_base_data_filename', str, 'processed_data'), # the most basic version of the filename of the continuous saved data
             ('-binbdf', 'bin_base_data_filename', str, 'binary_data_max_0.05unconnected'), # the filename of the binary saved data
             ('-datath', 'data_threshold', float, 0.5),  # threshold for the binary data
             ('-if', 'base_input_folder', str, 'Embeddings'), # base input folder of the embeddings
             ('-np', 'n_particles', int, 1000), # number of particles used in the embedding
             ('-nm', 'n_mcmc_steps', int, 100), # number of mcmc steps used in the embedding
             ('-tf', 'task_filename', str, 'task_list'), # filename of the list of task names
             ('-of', 'output_folder', str, 'Figures'), # folder where to dump figures
             ('-N', 'N', int, 164), # number of nodes
             ('-D', 'D', int, 2), # LS dimensionality
             ('-s1', 'subject1', int, 1), # first subject to plot
             ('-sn', 'subjectn', int, 25), # last subject to plot
             ('-et', 'edge_type', str, 'con'), # edge type ('con' or 'bin')
             ('-geo', 'geometry', str, 'hyp'), # LS geometry ('hyp' or 'euc')
             ('--bkst', 'is_bookstein', bool), # Whether the trace uses Bookstein anchors
             ('--partial', 'partial', bool), # whether to use partial correlations
             ('--bpf', 'bpf', bool), # whether to use band-pass filtered correlations
             ('--print', 'do_print', bool), # whether to print
             ('-lab', 'label_location', str, 'Figures/lobelabels.npz'),  # file location of the labels
             ('--nolabels', 'no_labels', bool), # whether to not use labels
             ('-plotth', 'plot_threshold', float, 0.4), # threshold for plotting edges
             ('-figsz', 'figsz', float, 10), # figure size in both x and y direction
             ('-gpu', 'gpu', str, ''),  # number of gpu to use (as str). If no GPU is specified, CPU is used.
             ]

global_params = get_cmd_params(arguments)
set_GPU(global_params['gpu'])

overwrite_data_filename = global_params['overwrite_data_filename']
data_folder = global_params['data_folder']
edge_type = global_params['edge_type']
geometry = global_params['geometry']

data_threshold = global_params['data_threshold']
base_data_filename = global_params['bin_base_data_filename'] if edge_type == 'bin' else global_params['con_base_data_filename']
n_particles = global_params['n_particles']
N = global_params['N']
M = N*(N-1)//2
D = global_params['D']
input_folder = f"{global_params['base_input_folder']}/{n_particles}p{global_params['n_mcmc_steps']}s"
task_filename = get_filename_with_ext(global_params['task_filename'], ext='txt', folder=data_folder)
output_folder = get_safe_folder(f"{global_params['output_folder']}/{n_particles}p{global_params['n_mcmc_steps']}s")
subject1 = global_params['subject1']
subjectn = global_params['subjectn']
is_bookstein = global_params['is_bookstein']
partial = global_params['partial']
bpf = global_params['bpf']
do_print = global_params['do_print']
label_location = global_params['label_location']
no_labels = global_params['no_labels']
plot_threshold = global_params['plot_threshold']
figsz = global_params['figsz']

det_params_dict = {'bin_euc':bin_euc_det_params,
                   'bin_hyp':bin_hyp_det_params,
                   'con_euc':con_euc_det_params,
                   'con_hyp':con_hyp_det_params}
det_params_func = det_params_dict[f"{edge_type}_{geometry}"]

if not overwrite_data_filename:
    data_filename = get_filename_with_ext(base_data_filename, partial, bpf, folder=data_folder)
else:
    data_filename = overwrite_data_filename
obs, tasks, encs = load_observations(data_filename, task_filename, subject1, subjectn, M)

# Load labels
if not no_labels:
    label_data = np.load(label_location)
    plt_labels = label_data[label_data.files[0]]
    if len(plt_labels) is not N:
        plt_labels = None
else:
    plt_labels = None

for si, n_sub in enumerate(range(subject1, subjectn + 1)):
    for ti, task in enumerate(tasks):
        if do_print:
            print(f"Making plot for S{n_sub} {task}")
        # Load embedding
        bin_txt = f"_{base_data_filename}" if edge_type == 'bin' else ''
        embedding_filename = get_filename_with_ext(f"{edge_type}_{geometry}_S{n_sub}_{task}_embedding{bin_txt}", partial=partial, bpf=bpf, folder=input_folder)
        with open(embedding_filename, 'rb') as f:
            embedding = pickle.load(f)

        if geometry == 'hyp':
            _z_positions = embedding.particles['_z']
            z_positions = lorentz_to_poincare(get_attribute_from_trace(_z_positions, det_params_func, 'z', shape=(n_particles, N, D + 1)))
        elif geometry == 'euc':
            z_positions= embedding.particles['z']

        ## TODO: ADD LABELS!
        # Plot posterior
        plt.figure(figsize=(figsz, figsz*.75))
        ax = plt.gca()
        if geometry == 'hyp':
            max_r = 1
            r_margin = 0
        else:
            radii = jnp.sqrt(jnp.sum(z_positions ** 2, axis=2))
            max_r = jnp.max(radii)
            r_margin = 0.1
        plot_posterior(z_positions,
                       edges=obs[si, ti, 0],
                       pos_labels=plt_labels,
                       ax=ax,
                       title=f"Proposal S{n_sub} {task}",
                       hyperbolic=geometry=='hyp',
                       bkst=is_bookstein,
                       disk_radius=max_r,
                       margin=r_margin,
                       threshold=plot_threshold)
        poincare_disk = plt.Circle((0, 0), max_r*(1+r_margin), color='k', fill=False, clip_on=False)
        ax.add_patch(poincare_disk)
        # Save figure
        partial_txt = '_partial' if partial else ''
        base_save_filename = f"{edge_type}_{geometry}_S{n_sub}_{task}_embedding_{base_data_filename}{partial_txt}"
        savefile = get_filename_with_ext(base_save_filename, ext='png', folder=output_folder)
        plt.savefig(savefile, bbox_inches='tight')
        plt.close()