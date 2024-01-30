import os
import numpy as np
import matplotlib.pyplot as plt
from plotting_functions import plot_metric
from helper_functions import get_cmd_params

arguments = [('-y', 'y_name', str, 'lml'),
             ('-x', 'x_name', str, 'n_particles'),
             ('-fn', 'filename', str, 'Statistics/statistics.csv'),
             ('-fol', 'figure_folder', str, 'Figures/metric_plots'),
             ('-fig', 'figure_file', str, 'metric'),
             ('-t', 'plt_type', str, 'scatter'),
             ('-lfs', 'label_fontsize', float, 20),  # fontsize of labels (and legend)
             ('-tfs', 'tick_fontsize', float, 16),  # fontsize of the tick labels
             ('-wsz', 'wrapsize', float, 20),  # wrapped text width
             ('--all', 'plot_all', bool),
             ]

global_params = get_cmd_params(arguments)
x_name = global_params['x_name']
filename = global_params['filename']
figure_folder = global_params['figure_folder']
figure_file = global_params['figure_file']
plt_type = global_params['plt_type']
plot_all = global_params['plot_all']
label_fontsize = global_params['label_fontsize']
tick_fontsize = global_params['tick_fontsize']
wrapsize = global_params['wrapsize']

# All possible names we want to plot against lml
x_names = ['n_particles', 'n_mcmc_steps', 'task']
y_names = ['lml', 'runtime']
label_bys = ['n_particles', 'n_mcmc_steps', 'task'] ## Adding None also plots averaged versions.

# To pretty plot
plt_names = {'n_particles' : 'number of particles',
             'n_mcmc_steps' : 'number of mcmc steps',
             'lml' : 'log-marginal likelihood',
             'runtime' : 'runtime (sec)'}

# This does not run on GPU so keep it like this
os.environ['CUDA_VISIBLE_DEVICES'] = ''

if plot_all:
    for x_name in x_names:
        plt_x_name = plt_names[x_name] if x_name in plt_names else None
        for y_name in y_names:
            plt_y_name = plt_names[y_name] if y_name in plt_names else None
            for label_by in label_bys:
                if label_by != x_name and label_by != y_name: # We're not sorting by a value already plotted
                    plt_label_by = plt_names[label_by] if label_by in plt_names else None
                    for plt_type in ['scatter', 'bar', 'box']:
                        plt.figure(figsize=(13,10))
                        ax = plt.gca()
                        ax = plot_metric(csv_file=filename,
                                         x_name=x_name,
                                         plt_x_name=plt_x_name,
                                         y_name=y_name,
                                         plt_y_name=plt_y_name,
                                         label_by=label_by,
                                         plt_label_by=plt_label_by,
                                         delim=';',
                                         ax=ax,
                                         plt_type=plt_type)
                        if label_by is not None and plt_type not in ['box']:
                            plt.savefig(f'{figure_folder}/{figure_file}_{x_name}_vs_{y_name}_by_{label_by}_{plt_type}.png')
                        else:
                            plt.savefig(f'{figure_folder}/{figure_file}_{x_name}_vs_{y_name}_{plt_type}.png')
                        plt.close()
else:
    plt_x_name = plt_x_names[x_name] if x_name in plt_x_names else None
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax = plot_metric(filename, x_name=x_name, plt_x_name=plt_x_name, y_name=y_name, plt_y_name=plt_y_name, delim=';', ax=ax, plt_type=plt_type)
    plt.savefig(f'{figure_folder}/{figure_file}_{x_name}_vs_{y_name}_{plt_type}.png')
    plt.close()