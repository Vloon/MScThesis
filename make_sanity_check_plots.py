import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from helper_functions import get_filename_with_ext, load_observations, get_cmd_params, set_GPU, get_attribute_from_trace
from continuous_hyperbolic_LSM import get_det_params as con_hyp_det_params
from plotting_functions import plot_distance_vs_correlations

arguments = [('-df', 'base_data_filename', str, 'processed_data'),  # the most basic version of the filename of the saved data
             ('-if', 'base_input_folder', str, 'Embeddings'), # base input folder of the embeddings
             ('-np', 'n_particles', int, 1000), # number of particles used in the embedding
             ('-nm', 'n_mcmc_steps', int, 100), # number of mcmc steps used in the embedding
             ('-tf', 'task_filename', str, 'task_list.txt'), # filename of the list of task names
             ('-of', 'output_folder', str, 'Figures/sanity_checks'), # folder where to dump figures
             ('-nps', 'number_plot_samples', int, 5), # number of particles to sample which will be plotted
             ('-N', 'N', int, 164), # number of nodes
             ('-s1', 'subject1', int, 1), # first subject to plot
             ('-sn', 'subjectn', int, 25), # last subject to plot
             ('-et', 'edge_type', str, 'con'), # edge type ('con' or 'bin')
             ('-geo', 'geometry', str, 'hyp'), # LS geometry ('hyp' or 'euc')
             ('--partial', 'partial', bool), # whether to use partial correlations
             ('--bpf', 'bpf', bool), # whether to use band-pass filtered correlations
             ('--balanced', 'balanced', bool),  # whether to set the vmin, vmax in a balanced way (0 in the middle)
             ('-gpu', 'gpu', str, ''),  # number of gpu to use (in string form). If no GPU is specified, CPU is used.
             ]

global_params = get_cmd_params(arguments)
base_data_filename = global_params['base_data_filename']
n_particles = global_params['n_particles']
input_folder = f"{global_params['base_input_folder']}/{n_particles}p{global_params['n_mcmc_steps']}s"
task_filename = global_params['task_filename']
output_folder = global_params['output_folder']
number_plot_samples = global_params['number_plot_samples']
N = global_params['N']
M = N*(N-1)//2
subject1 = global_params['subject1']
subjectn = global_params['subjectn']
edge_type = global_params['edge_type']
geometry = global_params['geometry']
partial = global_params['partial']
bpf = global_params['bpf']
balanced = global_params['balanced']
set_GPU(global_params['gpu'])

data_filename = get_filename_with_ext(base_data_filename, partial, bpf)
obs, tasks, encs = load_observations(data_filename, task_filename, subject1, subjectn, M)

for si, n_sub in tqdm(enumerate(range(subject1, subjectn + 1))):
    for ti, task in enumerate(tasks):
        # Load embedding
        embedding_filename = f"{input_folder}/{edge_type}_{geometry}_S{n_sub}_{task}_embedding.pkl"
        with open(embedding_filename, 'rb') as f:
            embedding = pickle.load(f)
        distance_trace = get_attribute_from_trace(embedding.particles['_z'], con_hyp_det_params, 'd_norm') # n_particles x M
        particle_plot_sample_idc = sorted(np.random.choice(np.arange(n_particles), number_plot_samples, replace=False))
        for pi, ppsi in enumerate(particle_plot_sample_idc):
            distances = distance_trace[ppsi, :]
            for ei, enc in enumerate(encs):
                correlations = obs[si, ti, ei, :] # M observed correlations

                output_file = get_filename_with_ext(f"dist_vs_corr_S{n_sub}_{task}_{enc}_sample{pi}",partial,bpf,ext='png',folder=output_folder)

                partial_txt = 'partial ' if partial else ''
                title = f"Distance vs {partial}correlation\nS{n_sub} {task} {enc}\nParticle {ppsi}"
                plt.figure()
                ax = plt.gca()
                ax = plot_distance_vs_correlations(distances, correlations, ax)
                ax.set_title(title)
                plt.tight_layout()
                plt.savefig(output_file)
                plt.close()