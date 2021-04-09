from os import mkdir
from os.path import isdir, exists
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import pandas as pd

from list_all_electrode import list_all_electrodes

def transform_location_str(loc_str, replace_dict={}):
    basic_change_dict = {
        'ctx': '',
        '_': ' ',
        '-': ' '}

    for to_replace, replacement in replace_dict.items():
        loc_str = loc_str.replace(to_replace, replacement)
    
    for to_replace, replacement in basic_change_dict.items():
        loc_str = loc_str.replace(to_replace, replacement)
        
    return loc_str.strip().capitalize()

def autolabel(ax, rects, error):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax.annotate('{:0.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height + error[i]),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize='large')

def autolabel_strs(ax, rects, strs, errors):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax.annotate(strs[i],
                    xy=(rect.get_x() + rect.get_width() / 2, height + errors[i]),
                    xytext=(0, 0),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize='large')

def plot_average_trial(data, savepath):
    '''
    Input = [trial x sample x columns]
    '''

     # get all trials per outcome
    # Get average of those trials
    # plot in subplots

    # rest_avg = data['rest'].mean(axis=0)

    labels = np.unique(data['non_rest_labels'])

    fig, axes = plt.subplots(nrows=1, 
                        ncols=len(labels)+1)
    axes[0].plot(data['rest'].mean(axis=0))
    axes[0].set_title('Rest')
    axes[0].set_ylabel('mV')
    axes[0].set_xlabel('time [.1s]')
    for i, label in enumerate(labels):
        mask = np.where(data['non_rest_labels']==label)[0]
        axes[i+1].plot(data['non_rest'][mask, :, :].mean(axis=0))
        axes[i+1].set_title(label)
        axes[i+1].set_xlabel('time [.1s]')
    # fig.suptitle('Average trial per label')

    return fig, ax

def plot_powerspectrograms_per_electrode(data, savepath):
    ##################

    # Output folder
    folder = '{}/channels_x_electrodes_spectrograms'.format(savepath)

    if not exists(folder):
        mkdir(folder)

    electrodes, counts = np.unique([ch.rstrip('0123456789') \
                                        for ch in data['channel_names'] \
                                        if '+' not in ch],
                                   return_counts=True)

    # Create a list of channels numbers per electrode
    channels = [[ch.lstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for ch in data['channel_names'] 
                                                        if electrode in ch]
                for electrode in electrodes]
    freqs = np.linspace(0, data['fs']/2, data['psd']['non_rest'].shape[-1])

    # Ensure that the colors in all plots are equal. Scales all values to
    #   the overall minimum and overall maximum
    get_extreme = lambda x, d: x([x(d[label].mean(axis=0)) for label in d.keys() if 'labels' not in label]) 
    color_scale = cm.colors.Normalize(vmin=get_extreme(np.min, data['psd']),
                                      vmax=get_extreme(np.max, data['psd']),
                                      clip=True)
    n_xticks = 3
    n_yticks = 3

    for label in data['psd'].keys():
        if 'labels' in label:
            continue
        # fig, axes = plt.subplots(nrows=max(counts), ncols=len(electrodes),
        #                          sharex=True, sharey=True,
        #                          frameon=False, figsize=(19.2, 10.8))
        fig = plt.figure(figsize=(19.2, 10.8), frameon=False)
        spec = gridspec.GridSpec(nrows=max(counts), ncols=len(electrodes),
                                 figure=fig)

        for j, electrode in enumerate(electrodes):
            for i, ch in enumerate(reversed(channels[j])):
                ax = fig.add_subplot(spec[i, j]) 

                current_ch_name = '{}{}'.format(electrode, ch)
                current_ch_idx = data['channel_names'].index(current_ch_name)
                
                # Form = [window x freq x power]
                mean_over_trials = data['psd'][label].mean(axis=0) 
                ax.imshow(mean_over_trials[:, current_ch_idx, :].T,
                                    aspect='auto',
                                    origin='lower',
                                    norm=color_scale)
                
                
                ax.set_xticks(np.linspace(0, mean_over_trials.shape[0]-1, n_xticks))
                x_ticklabels = np.linspace(1, mean_over_trials.shape[0], 3, dtype=np.int32)
                ax.set_xticklabels(x_ticklabels)

                ax.set_yscale('log')
                # Np log = natural log. for log 10 use np.log10()
                ax.set_yticks(np.logspace(0, np.log(mean_over_trials.shape[2]), n_yticks, 
                                          base=np.e, dtype=np.int32))
                y_ticklabels = np.logspace(-1, np.log(data['fs']/2), 3,
                                           base=np.e, dtype=np.int32)
                ax.set_ylim(1, mean_over_trials.shape[2])
                ax.set_yticklabels(y_ticklabels)

                # ax.set_yticks(np.linspace(0, mean_over_trials.shape[2], n_yticks))
                # y_ticklabels = np.linspace(0, data['fs']/2, 3, dtype=np.int32)
                # ax.set_yticklabels(y_ticklabels)

                show_x = False
                show_y = False
                if i==0:
                    ax.set_title('{}'.format(electrode))
                if i==len(channels[j])-1 and j==0:
                    show_x = True
                    ax.set_xlabel('Window [.1s]')
                    show_y = True
                    ax.set_ylabel('Frequency [~10Hz]')

                ax.tick_params(axis='both', which='both',
                               bottom=show_x, labelbottom=show_x,
                               left=show_y, labelleft=show_y)
                # ax.set_ylim(0, 22)

        plt.suptitle('Average Spectrogram [channel x electrode]\n{}'.format(label.upper()))

        # Save it
        # plt.show()
        plt.savefig('{}/{}_{}.png'.format(folder, 'psd', label))

def plot_per_electrode_window(data, savepath):
    plot_individual_trials = False
    plot_significant_windows = False

    freq_bands = {
        "delta": [0.5, 4],
        "theta": [4, 8],
        "alpha": [8, 12],
        "beta":  [12, 30],
        "gamma": [30, 80],
        "hfo":   [80, 200],
        "uhfo":  [200, 600]
    }

    replace_dict = {
                '_lh': '',
                '_rh': '',
                'Left-Cerebral-White-Matter': 'White matter',
                'Right-Cerebral-White-Matter': 'White matter'}

    ##################

    # Output folder
    folder = '{}/channels_x_electrodes_avg'.format(savepath)

    if plot_individual_trials:
        folder = '{}_and_individual_trials'.format(folder)

    if not exists(folder):
        mkdir(folder)

    color_list = ['r', 'b', 'g', 'y'] # Max four colors

    electrodes, counts = np.unique([ch.rstrip('0123456789') \
                                        for ch in data['channel_names'] \
                                        if '+' not in ch],
                                    return_counts=True)
    # Create a list of channels numbers per electrode
    channels = [[ch.lstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for ch in data['channel_names'] 
                                                        if electrode in ch]
                for electrode in electrodes]

    # for k, feature in enumerate(features):
    fig = plt.figure(figsize=(19.2, 10.8), frameon=False)
    spec = gridspec.GridSpec(nrows=max(counts), ncols=len(electrodes),
                                figure=fig, hspace=1)

    # plt.suptitle('{} {}\nTrial + average // page ({}/{})\n\nNote: Y-axis not equal'\
    #                 .format(feature.upper(), freq_bands[feature.lower()],
    # #                         k+1, len(features)))
    # plt.suptitle('{} {}Hz \nAverage\n\n'\
    #                 .format(feature.upper(), freq_bands[feature.lower()]))


    for j, electrode in enumerate(electrodes):
        for i, ch in enumerate(reversed(channels[j])):
            ax = fig.add_subplot(spec[i, j]) 
            
            current_data = data['Original']['data']
            labels = data['Original']['label']
            n_windows = current_data.shape[0]

            current_ch_name = '{}{}'.format(electrode, ch)
            current_ch_idx = data['feature_names'].index(current_ch_name)
            # current_ch_name = '{}_{}{}'.format(feature, electrode, ch)
            # current_ch_idx = data['feature_names'].index(current_ch_name)

            x = np.arange(current_data.shape[0])
            for i, label in enumerate(np.unique(labels)):
                data_masked = current_data[:, current_ch_idx].copy()
                data_masked = np.ma.masked_where(label!=labels, data_masked)
                ax.plot(x, data_masked, label=label,
                        color=color_list[i], 
                        linewidth=0.1)
            
            if plot_significant_windows:
                x = np.linspace(0, n_windows-1, n_windows)
                ylims = ax.get_ylim()
                ax.fill_between(x[:data['rest_vs_non_rest'].shape[0]], ylims[0], ylims[1],
                                        where=data['rest_vs_non_rest'][:, current_ch_idx]==1,
                                        facecolor='green', alpha=0.4)

            show_x = False
            show_y = False
            
            xticks = np.linspace(0, n_windows, 3)
            xticklabels = xticks*data['frameshift'] # TODO: Check is this holds true when changing window size and frameshift
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)

            if i==0:
                if 'channel_locations' in data.keys():
                    title = '{}'.format(electrode[0])
                    ax.annotate(title, xy=(0.5, 1.75), xytext=(0, 5),
                                xycoords='axes fraction', textcoords='offset points',
                                size='large', ha='right', va='center')
                else:
                    ax.set_title('{}'.format(electrode))

            if i==len(channels[j])-1 and j==0:
                show_x = True
                ax.set_xlabel('Time [s]')
                show_y = True
                ax.set_ylabel('Voltage [mV]')
                ax.spines['right'].set_visible(False)            
                ax.spines['top'].set_visible(False)
            else:
                # ax.axis('off')
                show_y=True
                ax.spines['right'].set_visible(False)            
                ax.spines['top'].set_visible(False)
                
            ax.tick_params(axis='both',
                            which='both',
                            bottom=show_x,
                            labelbottom=show_x,
                            left=show_y,
                            labelleft=show_y)
                            
            if 'channel_locations' in data.keys():
                title = transform_location_str(data['channel_locations'] \
                                                .get('{}{}'.format(electrode, ch), 'NID'),
                                                replace_dict=replace_dict)
                ax.set_title(title, fontsize=7, pad=2)

            if i==0 and j==len(electrodes)-1:
                plt.legend(bbox_to_anchor=(1, 1.05))

    # Save it
    # plt.show()
    plt.savefig('{}/{}.png'.format(folder, 'windowed'))
    plt.close('all')    

def plot_per_electrode(data, savepath):
    # Shape of plot: Channel (reversed) x Electrode. Channel reversed such that the lowest plot is the deepest electrode
    # opt 1) Plot per trial (alpha=0.5) and average (alpha=1)
    # opt 2) Plot average per trial per frequency band
    # TODO: Remove E channels before subplot declaration

    plot_individual_trials = True
    plot_significant_windows = False
    freq_bands = {
        "delta": [0.5, 4],
        "theta": [4, 8],
        "alpha": [8, 12],
        "beta":  [12, 30],
        "gamma": [30, 80],
        "hfo":   [80, 200],
        "uhfo":  [200, 600]
    }

    replace_dict = {
                '_lh': '',
                '_rh': '',
                'Left-Cerebral-White-Matter': 'White matter',
                'Right-Cerebral-White-Matter': 'White matter'}

    ##################

    # Output folder
    folder = '{}/channels_x_electrodes_avg'.format(savepath)

    if plot_individual_trials:
        folder = '{}_and_individual_trials'.format(folder)

    if not exists(folder):
        mkdir(folder)

    color_list = ['r', 'b', 'g', 'y'] # Max four colors

    electrodes, counts = np.unique([ch.rstrip('0123456789') \
                                        for ch in data['channel_names'] \
                                        if '+' not in ch],
                                    return_counts=True)
    # Create a list of channels numbers per electrode
    channels = [[ch.lstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for ch in data['channel_names'] 
                                                        if electrode in ch]
                for electrode in electrodes]
    features = np.unique([ch.split('_')[0] for ch in data['feature_names']])

    for k, feature in enumerate(features):
        fig = plt.figure(figsize=(19.2, 10.8), frameon=False)
        spec = gridspec.GridSpec(nrows=max(counts), ncols=len(electrodes),
                                    figure=fig, hspace=1)

        plt.suptitle('{} {}\nTrial + average // page ({}/{})\n\nNote: Y-axis not equal'\
                        .format(feature.upper(), freq_bands[feature.lower()],
                                k+1, len(features)))
        plt.suptitle('{} {}Hz \nAverage\n\n'\
                        .format(feature.upper(), freq_bands[feature.lower()]))

        y_low = np.inf
        y_high = -np.inf

        for j, electrode in enumerate(electrodes):
            for i, ch in enumerate(reversed(channels[j])):
                ax = fig.add_subplot(spec[i, j]) 
                
                current_ch_name = '{}_{}{}'.format(feature, electrode, ch)
                current_ch_idx = data['feature_names'].index(current_ch_name)

                n_windows = max(data['non_rest'].shape[1], data['rest'].shape[1])

                data['left'] = data['non_rest'][np.where(data['non_rest_labels']=='Links')[0], :, :]
                data['right'] = data['non_rest'][np.where(data['non_rest_labels']=='Rechts')[0], :, :]
                
                for h, label in enumerate(['Rest', 'Left', 'Right']):
                    avg = data[label].mean(axis=0)
                    if avg[:, current_ch_idx].min() < y_low:
                        y_low = avg[:, current_ch_idx].min()
                    if avg[:, current_ch_idx].max() > y_high:
                        y_high = avg[:, current_ch_idx].max()
                    if plot_individual_trials:
                        for trial_idx in range(data[label].shape[0]):
                            ax.plot(data[label][trial_idx, :, current_ch_idx],
                                            alpha=0.5, color=color_list[h])
                    ax.plot(avg[:, current_ch_idx],
                            color=color_list[h], linewidth=2,
                            label=label.capitalize())
                
                if plot_significant_windows:
                    x = np.linspace(0, n_windows-1, n_windows)
                    ylims = ax.get_ylim()
                    ax.fill_between(x[:data['rest_vs_non_rest'].shape[0]], ylims[0], ylims[1],
                                            where=data['rest_vs_non_rest'][:, current_ch_idx]==1,
                                            facecolor='green', alpha=0.4)

                show_x = False
                show_y = False
                
                xticks = np.linspace(0, n_windows, 3)
                xticklabels = xticks*data['frameshift'] # TODO: Check is this holds true when changing window size and frameshift
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)

                if i==0:
                    if 'channel_locations' in data.keys():
                        title = '{}'.format(electrode[0])
                        ax.annotate(title, xy=(0.5, 1.75), xytext=(0, 5),
                                    xycoords='axes fraction', textcoords='offset points',
                                    size='large', ha='right', va='center')
                    else:
                        ax.set_title('{}'.format(electrode))

                if i==len(channels[j])-1 and j==0:
                    show_x = True
                    ax.set_xlabel('Time [s]')
                    show_y = True
                    ax.set_ylabel('Voltage [mV]')
                    ax.spines['right'].set_visible(False)            
                    ax.spines['top'].set_visible(False)
                else:
                    # ax.axis('off')
                    show_y=True
                    ax.spines['right'].set_visible(False)            
                    ax.spines['top'].set_visible(False)
                    
                ax.tick_params(axis='both',
                               which='both',
                               bottom=show_x,
                               labelbottom=show_x,
                               left=show_y,
                               labelleft=show_y)
                                
                if 'channel_locations' in data.keys():
                    title = transform_location_str(data['channel_locations'] \
                                                  .get('{}{}'.format(electrode, ch), 'NID'),
                                                  replace_dict=replace_dict)
                    ax.set_title(title, fontsize=7, pad=2)

                if i==0 and j==len(electrodes)-1:
                    plt.legend(bbox_to_anchor=(1, 1.05))

            # Z-score data?
            # for j, electrode in enumerate(electrodes):    
            #     for i, ch in enumerate(reversed(channels[j])):
            #         ax = fig.add_subplot(spec[i, j]).set_ylim(y_low, y_high)

        # Save it
        plt.show()
        # plt.savefig('{}/{}.png'.format(folder, feature))
    plt.close('all')

def plot_bargrid(data, results, channel_locs, savepath, plot_all=False):
    exps_all = ['grasp', 'imagine']
    # Extract scores and put them in useful format
    exps = []
    scores = {}
    for ppt, exp_data in results.items():
        for exp in exps_all:
            if exp not in exp_data.keys():
                continue
            exps += [exp]
            if exp not in scores.keys():
                scores[exp] = {}
            for score in exp_data[exp].values():
                for name in score.keys():
                    if name not in scores[exp].keys():
                        scores[exp][name] = []
                    scores[exp][name] += [score[name]] 


    # Init grid
    fig = plt.figure(figsize=(16, 8), dpi=100)
    gs = gridspec.GridSpec(2, 3, wspace=.5)

    for i, exp in enumerate(exps):
        ax = fig.add_subplot(gs[i, 0])            

        names = [name[5:-1] for name in scores[exp].keys() if '_ci' not in name]

        n_bars = np.arange(len(names))
        n_bars_sub = len(results[ppt][exp].keys())
        bar_width = 0.7 / n_bars_sub

        colors = ['tab:blue', 'tab:red', 'tab:orange']    
        for j, date in enumerate(results[ppt][exp].keys()):
            score = []
            score_ci = []
            # Somehow two list comprehensions didn't work...
            for name, value in scores[exp].items():
                if '_ci' not in name:
                    score += [value[j]]
                if '_ci' in name:
                    score_ci += [value[j]]
            rects = ax.bar(n_bars + j*bar_width, score, bar_width,
                            bottom=0, yerr=score_ci,
                            color=colors[j],
                            label='{}-{}'.format(date[4:6], date[6:]))
            autolabel(ax, rects, score_ci)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(0.5, color='k', linestyle='dashed', label='chance')
        ax.set_ylim(0, 1)
        ax.set_xticks(n_bars+bar_width/2)
        ax.set_xticklabels(names, fontsize=6.5)
        ax.tick_params(axis='x',
                    labelrotation=0)

        ax.set_ylabel('ROC AUC', fontsize=12)
        ax.set_title(exp.capitalize())
        if i==0 and n_bars_sub > 1:
            ax.legend(bbox_to_anchor=(1, .75, .2, .2))

    # Plot locations
    ax_loc = fig.add_subplot(gs[:, 1:])
    ppt = ppt if not plot_all else None
    ax_loc = list_all_electrodes(ax=ax_loc, ppt=ppt,
                                ch_locs=channel_locs) # imported

    plt.savefig('{}/overall_result.png'.format(savepath))

    # plt.show()

def plot_bargrid_averaged_score(data, results, channel_locs, savepath, plot_all=False):
    exps_all = ['grasp', 'imagine']
    # Extract scores and put them in useful format
    exps = []
    scores = {}
    for ppt, exp_data in results.items():
        for exp in exps_all:
            if exp not in exp_data.keys():
                continue
            exps += [exp]
            if exp not in scores.keys():
                scores[exp] = {}
            for score in exp_data[exp].values():
                for name in score.keys():
                    if name not in scores[exp].keys():
                        scores[exp][name] = []
                    scores[exp][name] += [score[name]] 


    # Init grid
    fig = plt.figure(figsize=(16, 8), dpi=100)
    gs = gridspec.GridSpec(2, 3, wspace=.5)

    for i, exp in enumerate(exps_all):
        score = [np.mean(value) for name, value in scores[exp].items() if '_ci' not in name]
        score_ci = [np.mean(value) for name, value in scores[exp].items() if '_ci' in name]
        names = [name[5:-1] for name in scores[exp].keys() if '_ci' not in name]

        ax = fig.add_subplot(gs[i, 0]) 
        # ax = ax_1 if exp == 'grasp' else ax_2
                
        n_bars = np.arange(len(names))
        bar_width = 0.7
        rects = ax.bar(n_bars+bar_width/2, score, bar_width,
            bottom=0, yerr=score_ci,
            color='tab:blue', label='Execute')
        
        autolabel(ax, rects, score_ci)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(0.5, color='k', linestyle='dashed', label='chance')
        ax.set_ylim(0, 1)
        ax.set_xticks(n_bars+bar_width/2)
        ax.set_xticklabels(names, fontsize=6.5)
        ax.tick_params(axis='x',
                    labelrotation=0)

        ax.set_ylabel('ROC AUC', fontsize=12)
        ax.set_title(exp.capitalize())


        # Plot locations

    ax_loc = fig.add_subplot(gs[:, 1:])
    ppt = ppt if not plot_all else None
    ax_loc = list_all_electrodes(ax=ax_loc, ppt=ppt,
                                    ch_locs=channel_locs) # imported

    plt.savefig('{}/overall_result_averaged_score.png'.format(savepath))
    # plt.show()
    # print('')

def plot_bargrid_score_per_metric(data, results, channel_locs, savepath, plot_all=False):
    exps_all = ['grasp', 'imagine']
    # Extract scores and put them in useful format
    scores = {}
    for ppt, exps in results.items():
        if ppt not in scores.keys():
            scores[ppt] = {}
        for exp, dates in exps.items():
            if exp not in scores[ppt]:
                scores[ppt][exp] = {}
            current_aucs = {}
            for dates, aucs in dates.items():
                for name, auc in aucs.items():
                    if name not in current_aucs.keys():
                        current_aucs[name] = []
                    current_aucs[name] += [auc]
            
            scores[ppt][exp] = {name: np.mean(aucs) for name, aucs in current_aucs.items()}


    # Init grid
    fig = plt.figure(figsize=(16, 8), dpi=100)
    gs = gridspec.GridSpec(2, 3, wspace=.5)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    for i, exp in enumerate(exps_all):
        ax = fig.add_subplot(gs[i, :2]) 
        n_ppts = len(scores.keys())
        total_bar_width = .8
        bar_width = total_bar_width / n_ppts
        total_n_bars = 0
        for j, (ppt, aucs) in enumerate(scores.items()):
            if exp not in aucs.keys():
                continue
            names = [name for name in aucs[exp].keys() if '_ci' not in name]
            score = []
            score_ci = []
            for name in names:
                score += [aucs[exp][name]]
                score_ci += [aucs[exp]['{}_ci'.format(name)]]

            n_bars = np.arange(len(names))
            total_n_bars = max(total_n_bars, len(names))

            rects = ax.bar(n_bars + j*bar_width, score, bar_width,
                        bottom=0, yerr=score_ci,
                        color=colors[j], label=ppt)
            autolabel(ax, rects, score_ci)
            # print('done: {} {}'.format(ppt, exp))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(0.5, color='k', linestyle='dashed', label='chance')
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(total_n_bars)+total_bar_width/2)
        ax.set_xticklabels([name[5:-1] for name in names], fontsize=8)
        ax.tick_params(axis='x',
                    labelrotation=0)

        ax.set_ylabel('ROC AUC', fontsize=12)
        ax.set_title(exp.capitalize())
        if i==0 and total_n_bars > 1:
            ax.legend(bbox_to_anchor=(-0.1, 1), prop={'size': 7})

        # Plot locations
    ax_loc = fig.add_subplot(gs[:, 2:])
    ax_loc = list_all_electrodes(ax=ax_loc, ppt=None, ch_locs=channel_locs) # imported

    plt.savefig('{}/overall_result_score_per_metric.png'.format(savepath))
    # plt.show()
    # print('')

def plot_bargrid_score_per_ppt(data, results, channel_locs, savepath, plot_all=False):
    exps_all = ['grasp', 'imagine']
    # Extract scores and put them in useful format
    scores = {}

    for ppt, exps in results.items():
        if ppt not in scores.keys():
            scores[ppt] = {}
        for exp, dates in exps.items():
            if exp not in scores[ppt]:
                scores[ppt][exp] = {}
            current_aucs = {}
            for dates, aucs in dates.items():
                for name, auc in aucs.items():
                    if name not in current_aucs.keys():
                        current_aucs[name] = []
                    current_aucs[name] += [auc]
            
            scores[ppt][exp] = {name: np.mean(aucs) for name, aucs in current_aucs.items()}

    scores2 = {}
    for ppt, exps in scores.items():
        for exp, aucs in exps.items():
            for name, auc in aucs.items():
                if exp not in scores2.keys():
                    scores2[exp] = {}
                if name not in scores2[exp].keys():
                    scores2[exp][name] = {}
                scores2[exp][name][ppt] = auc
    scores = scores2

    # Init grid
    fig = plt.figure(figsize=(16, 8), dpi=100)
    gs = gridspec.GridSpec(2, 3, wspace=.5)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # for i, exp in enumerate(exps_all):

    for i, (exp, aucs) in enumerate(scores.items()):
        ax = fig.add_subplot(gs[i, :2]) 
        n_ppts = len(results.keys())
        total_n_bars = n_ppts
        total_bar_width = .8
        
        auc_names = [name for name in aucs.keys() if '_ci' not in name]
        bar_width = total_bar_width / len(auc_names)
        for j, name in enumerate(auc_names):
            auc_scores = []
            auc_scores_ci = []
            ppts = []
            for ppt in aucs[name].keys():
                auc_scores += [aucs[name][ppt]]
                auc_scores_ci += [aucs['{}_ci'.format(name)][ppt]]
                ppts += [ppt]
            
            # names = ppts
            n_bars = np.arange(len(ppts))
            # total_n_bars = max(total_n_bars, len(names))
            rects = ax.bar(n_bars + j*bar_width, auc_scores, bar_width,
                        bottom=0, yerr=auc_scores_ci,
                        color=colors[j], label=name[5:-1])
            autolabel(ax, rects, auc_scores_ci)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(0.5, color='k', linestyle='dashed', label='chance')
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(total_n_bars)+total_bar_width/2)
        ax.set_xticklabels(ppts, fontsize=8)
        ax.tick_params(axis='x',
                    labelrotation=0)

        ax.set_ylabel('ROC AUC', fontsize=12)
        ax.set_title(exp.capitalize())
        if i==0 and total_n_bars > 1:
            ax.legend(bbox_to_anchor=(-0.1, 1), prop={'size': 7})

        # Plot locations
    ax_loc = fig.add_subplot(gs[:, 2:])
    ax_loc = list_all_electrodes(ax=ax_loc, ppt=None, ch_locs=channel_locs) # imported

    plt.savefig('{}/overall_result_score_per_ppt.png'.format(savepath))
    # plt.show()
    # print('')

def plot_bargrid_score_per_ppt_no_loc(all_results, savepath, plot_all=False):
    score_name_dict = {
        'AUC [Links vs Rechts]': 'Left vs Right',
        'AUC [Links vs Rest]': 'Left vs Rest',
        'AUC [Rechts vs Rest]': 'Right vs Rest',
        # 'AUC [Avg 1v1]': 'Average',
        'AUC [Move vs Rest]': 'Move vs Rest'
    }
    score_order = ['AUC [Move vs Rest]', 
                #    'AUC [Rechts vs Rest]', 'AUC [Links vs Rest]',
                   'AUC [Links vs Rechts]',
                #    'AUC [Avg 1v1]'
    ]

    ppt_order = ['kh09', 'kh10', 'kh11', 'kh12', 'kh13', 'kh14', 'kh15', 'kh18']
    ppt_name_dict = {'kh09': 'P9', 'kh10': 'P10',
                     'kh11': 'P11', 'kh12': 'P12',
                     'kh13': 'P13', 'kh14': 'P14',
                     'kh15': 'P15', 'kh18': 'P18'}
                    
    ## EXEC IMAG
    results = all_results['beta_gamma']
    scores = {}
    for current_exp in ['grasp', 'imagine']:
        for ppt, exps in results.items():
            if ppt not in scores.keys():
                scores[ppt] = {}
            for exp, dates in exps.items():
                if exp != current_exp:
                    continue
                # if 'gamma' not in scores[ppt]:
                #     scores[ppt]['gamma'] = {}
                current_aucs = {}
                # first_date = np.inf
                for dates, aucs in dates.items():
                    # TODO: What to do with the second measurement???
                    # if int(dates) < first_date:
                    #     first_date = int(dates)

                    for name, auc in aucs.items():
                        if name not in current_aucs.keys():
                            current_aucs[name] = []
                        current_aucs[name] += [auc]
                scores[ppt] = {name: aucs[0] for name, aucs in current_aucs.items()} # Take first occurance (date NOTE: Not guaranteed order)            
                # scores[ppt] = {name: np.mean(aucs) for name, aucs in current_aucs.items()} # Average the two attempts
        if current_exp == 'grasp':
            grasp = scores.copy()
        else:
            imag = scores.copy()



    # Extract scores and put them in useful format
    # results = all_results['gamma']
    # scores = {}
    # for ppt, exps in results.items():
    #     if ppt not in scores.keys():
    #         scores[ppt] = {}
    #     for exp, dates in exps.items():
    #         # if 'gamma' not in scores[ppt]:
    #         #     scores[ppt]['gamma'] = {}
    #         current_aucs = {}
    #         # first_date = np.inf
    #         for dates, aucs in dates.items():
    #             # TODO: What to do with the second measurement???
    #             # if int(dates) < first_date:
    #             #     first_date = int(dates)

    #             for name, auc in aucs.items():
    #                 if name not in current_aucs.keys():
    #                     current_aucs[name] = []
    #                 current_aucs[name] += [auc]
    #         scores[ppt] = {name: aucs[0] for name, aucs in current_aucs.items()} # Take first occurance (date NOTE: Not guaranteed order)            
    #         # scores[ppt] = {name: np.mean(aucs) for name, aucs in current_aucs.items()} # Average the two attempts
    # gamma = scores.copy()

    # results = all_results['beta']
    # scores = {}
    # for ppt, exps in results.items():
    #     if ppt not in scores.keys():
    #         scores[ppt] = {}
    #     for exp, dates in exps.items():
    #         # if 'gamma' not in scores[ppt]:
    #         #     scores[ppt]['gamma'] = {}
    #         current_aucs = {}
    #         for dates, aucs in dates.items():
    #             for name, auc in aucs.items():
    #                 if name not in current_aucs.keys():
    #                     current_aucs[name] = []
    #                 current_aucs[name] += [auc]
    #         scores[ppt] = {name: aucs[0] for name, aucs in current_aucs.items()}                
    #         # scores[ppt] = {name: np.mean(aucs) for name, aucs in current_aucs.items()}
    # beta = scores.copy()

    # Init grid
    # fig = plt.figure(nrows=2, ncols=len(gamma.keys()), figsize=(16, 8), dpi=300)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), dpi=300) #12,10
    # gs = gridspec.GridSpec(2, 3, wspace=.5)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    # results = {'Beta [12-30] Hz': beta,
    #            'Gamma [55-90] Hz': gamma}
    results = {'Execution': grasp,
               'Imagined': imag}

    significance_value = 0.519

    for i, (band, aucs) in enumerate(results.items()):

        total_bar_width = 0.8
        bar_width = total_bar_width / len(score_order)
        n_bars = np.arange(len(aucs.keys()))
        
        for j, score_name in enumerate(score_order):
            # if score_name == ''
            scores = []
            scores_ci = []
            
            scores = [aucs[ppt][score_name] for ppt in ppt_order]
            scores_ci = [aucs[ppt]['{}_ci'.format(score_name)] for ppt in ppt_order]

            rects = axes[i].bar(n_bars+j*bar_width, scores, bar_width,
                           bottom=0, yerr=scores_ci,
                           color=colors[j], label=score_name_dict[score_name])
            autolabel_strs(axes[i], rects, ['*' if score>significance_value else '' for score in scores], scores_ci)
            # autolabel(axes[i], rects, scores_ci)
            
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].set_yticks(np.linspace(0, 1, 11))
        axes[i].axhline(0.5, color='k', linestyle='dashed', label='Chance')
        axes[i].set_ylim(0, 1)
        axes[i].grid(axis='y', alpha=0.5)
        axes[i].set_xticks(n_bars+(total_bar_width/2)-(total_bar_width/len(score_order))/2)
        axes[i].set_xticklabels([ppt_name_dict[ppt] for ppt in ppt_order], fontsize=16)
        axes[i].tick_params(axis='x',
                            labelrotation=0)
        axes[i].tick_params(axis='y', labelsize=16)
        # if i==0:
        if True:
            axes[i].set_ylabel('ROC AUC', fontsize=20)
        axes[i].set_title(band.capitalize(), fontsize=24, y=1.08)
        if i==1:
            axes[i].legend(
                frameon=False,
                loc='lower center',
                ncol=3,
                prop={'size': 25},
                bbox_to_anchor=(.5, -.35)
            )                
                
                #bbox_to_anchor=(1, 1),
                        #    frameon=False, 
                        #    prop={'size': 18},
                        #    mode='expand')

    plt.tight_layout()
    plt.savefig('{}/overall_result_score_per_ppt.png'.format(savepath))
    # plt.savefig('{}/overall_result_score_per_ppt.svg'.format(savepath))
    # plt.show()
    # print('')

def plot_correlation_matrix(data, savepath):
    import seaborn as sns
    df_d = pd.concat([data, pd.get_dummies(data['loc'])], axis=1)
    df_d = pd.concat([df_d, pd.get_dummies(df_d['score_name'])], axis=1)
    df_d = df_d.drop(['loc', 'score_name'], axis=1)

    plt.figure()
    sns.heatmap(df_d.corr(), xticklabels=True, yticklabels=True, robust=True)
    plt.ylim([0, df_d.shape[1]])
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    return df_d


def pretty_print(results, savepath=None):
    # TODO: Guarantee dates to be ordered?
    print('')
    lines = []
    for current_exp in ['imagine', 'grasp']:
        all_metrics = {}
        for sub, subj_results in results.items():
            for exp, exp_results in subj_results.items():
                if current_exp == exp:
                    # Order all graps and imagines seperately
                    continue
                for i, (date, data) in enumerate(exp_results.items()):
                    n_experiments = len(exp_results.keys())
                    
                    exp_s = '{} {}'.format(exp, i) if n_experiments > 1 else '{}'.format(exp)
                    sub_s = '' if n_experiments > 1 and i > 0 else sub
                    base_s = '{:<5s}\t{:<15s}'.format(sub_s, exp_s)

                    for metric, value in data.items():
                        if 'Links vs rest' in metric or 'Rechts vs rest' in metric:
                            continue
                        if '_ci' in metric:
                            continue
                        if metric not in all_metrics.keys():
                            all_metrics[metric] = []
                        all_metrics[metric] += [value]
                        base_s += '{:s} = {:.2f}'.format(metric, value)
                        base_s += ' \u00B1 {:.2f}\t'.format(data[metric+'_ci'])

                    if n_experiments > 1:
                        # Only add date if multiple experiments
                        base_s += '{:>15s}(date: {:s})'.format('', date)

                    lines += [base_s]
                    print(base_s)
        # Average
        base_s = '{:<5s}\t{:<15s}'.format('AVG', exp_s)
        for metric, values in all_metrics.items():
            values = np.array(values)
            mean = values.mean()
            ci = (1.96*values.std()) / (np.sqrt(values.size))
            base_s += '{:s} = {:.2f}'.format(metric, mean)
            base_s += ' \u00B1 {:.2f}\t'.format(ci)
        if any(all_metrics):
            print(base_s)
            print('')

    if savepath != None:
        fullpath = '{}/results.txt'.format(savepath)
        with open(fullpath, 'w') as file:
            file.writelines(['{}\n'.format(line) for line in lines])

def plot_all(data, results, savepath):
    if not exists(savepath):
        mkdir(savepath)
    figures_path = '{}/figures'.format(savepath) 
    if not exists(figures_path):
        mkdir(figures_path)
    # if not exists('{}/{}'.format(figures_path, data)):
    #     pass
    #     # TODO: Add date
    
    plot_bargrid_score_per_ppt_no_loc(results, savepath)

    # plot_per_electrode(data, figures_path)
    # plot_per_electrode_window(data, figures_path)
    # plot_powerspectrograms_per_electrode(data, figures_path)
    # fig, ax = plot_average_trial(data, figures_path)

    # plot_bargrid(data, results, figures_path)
    # plt.show()
    plt.close('all')

if __name__=='__main__':
    results = {}
    for band in ['beta_gamma']: #'['beta', 'gamma']:
        path = r'C:\Users\p70066129\Projects\Grasp\Figures\{}_1000_100'.format(band)
        with open('{}/{}_results.pkl'.format(path, band), 'rb') as file:
            results[band] = pickle.load(file)
    
    plot_bargrid_score_per_ppt_no_loc(results, r'C:\Users\p70066129\Projects\Grasp\Figures')

