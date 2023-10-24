import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
import os

from helper_functions import get_filename_with_ext, load_observations, get_cmd_params, set_GPU, get_safe_folder, get_attribute_from_trace
from plotting_functions import plot_posterior
from continuous_hyperbolic_LSM import get_det_params as con_hyp_det_params

arguments = [('-overwritedf', 'overwrite_data_filename', str, None),  # if used, it overwrites the default filename
             ('-if', 'base_input_folder', str, 'Embeddings'), # base input folder of the embeddings
             ('-np', 'n_particles', int, 1000), # number of particles used in the embedding
             ('-nm', 'n_mcmc_steps', int, 100), # number of mcmc steps used in the embedding
             ('-tf', 'task_filename', str, 'task_list.txt'), # filename of the list of task names
             ('-of', 'output_folder', str, 'Figures'), # folder where to dump figures
             ('-N', 'N', int, 164), # number of nodes
             ('-s1', 'subject1', int, 1), # first subject to plot
             ('-sn', 'subjectn', int, 25), # last subject to plot
             ('-et', 'edge_type', str, 'con'), # edge type ('con' or 'bin')
             ('-geo', 'geometry', str, 'hyp'), # LS geometry ('hyp' or 'euc')
             ('--bkst', 'is_bookstein', bool), # Whether the trace uses Bookstein anchors
             ('--partial', 'partial', bool), # whether to use partial correlations
             ('--bpf', 'bpf', bool), # whether to use band-pass filtered correlations
             ('-plotth', 'plot_threshold', float, plot_threshold), # threshold for plotting edges
             ('-gpu', 'gpu', str, ''),  # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]

global_params = get_cmd_params(arguments)
overwrite_data_filename = global_params['overwrite_data_filename']
n_particles = global_params['n_particles']
N = global_params['N']
M = N*(N-1)//2
input_folder = f"{global_params['base_input_folder']}/{n_particles}p{global_params['n_mcmc_steps']}s"
task_filename = global_params['task_filename']
output_folder = get_safe_folder(f"{global_params['output_folder']}/{n_particles}p{global_params['n_mcmc_steps']}s")
subject1 = global_params['subject1']
subjectn = global_params['subjectn']
edge_type = global_params['edge_type']
geometry = global_params['geometry']
is_bookstein = global_params['is_bookstein']
partial = global_params['partial']
bpf = global_params['bpf']
plot_threshold = global_params['plot_threshold']
set_GPU(global_params['gpu'])

det_params_dict = {'con_hyp':con_hyp_det_params}
det_params_func = det_params_dict[f"{edge_type}_{geometry}"]

if not overwrite_data_filename:
    data_filename = get_filename_with_ext(base_data_filename, partial, bpf)
else:
    data_filename = overwrite_data_filename
obs, tasks, encs = load_observations(data_filename, task_filename, subject1, subjectn, M)

for si, n_sub in enumerate(range(subject1, subjectn + 1)):
    for ti, task in enumerate(tasks):
        # Load embedding
        embedding_filename = f"{input_folder}/{edge_type}_{geometry}_S{n_sub}_{task}_embedding.pkl"
        with open(embedding_filename, 'rb') as f:
            embedding = pickle.load(f)

        if geometry == 'hyp'
            ## TODO: add bkst to trace
            _z_positions = embedding.particles['_z']
            z_positions = lorentz_to_poincare(get_attribute_from_trace(_z_positions, det_params_func, 'z', shape=(n_particles, N, D + 1)))
        elif geometry == 'euc':
            z_positions= embedding.particles['z']

        ## TODO: ADD LABELS!
        # Plot posterior
        plt.figure()
        ax = plt.gca()
        plot_posterior(z_positions,
                       edges=obs[si, ti, 0],
                       pos_labels=plt_labels,
                       ax=ax,
                       title=f"Proposal S{n_sub} {task}",
                       hyperbolic=geometry=='hyp',
                       bkst=is_bookstein,
                       threshold=plot_threshold)
        poincare_disk = plt.Circle((0, 0), 1, color='k', fill=False, clip_on=False)
        ax.add_patch(poincare_disk)
        # Save figure
        savefile = f'./{figure_folder}/{base_save_filename}.png'
        plt.savefig(savefile)
        plt.close()