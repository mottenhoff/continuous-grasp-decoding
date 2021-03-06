# Builtin
import sys
import pickle
import copy
from os import mkdir
from os.path import isdir, exists

# 3th party
import numpy as np
import scipy.signal
from sklearn.metrics import roc_auc_score

# Local
paths = [r".\classifiers/"]
for path in paths:
    sys.path.insert(0, path)

from grasp_util import get_filenames
from grasp_util import load_seeg
from grasp_util import load_pickle
from grasp_util import load_location_files
from grasp_util import is_file_to_skip
from grasp_util import save_dict
from grasp_util import remove_data
from grasp_util import clean_data
from grasp_util import apply_reference
from grasp_util import apply_filters
from grasp_util import get_power_spectrum
from grasp_util import get_fractal_component
from grasp_util import generate_complex_morlet_wavelet
from grasp_util import create_windows
from grasp_util import combine_features_with_channels
from grasp_util import split_per_trial
from grasp_util import trial_per_window
from grasp_util import perform_stat_test
from grasp_util import get_train_test
from grasp_util import combine_data_to_x_y

from grasp_plot import pretty_print
from grasp_plot import plot_all
# from grasp_plot import plot_average_trial
from grasp_plot import plot_bargrid
from grasp_plot import plot_bargrid_averaged_score, plot_bargrid_score_per_metric, plot_bargrid_score_per_ppt

# Classifiers
from lda import LDA
from rfc import RFC 


def load_raw_data(filename):
    data = None

    try:
        data = load_seeg(filename)
    except Exception as e:
        print("Failed to load: {}, due to exception: {}". format(str(filename), e))

    return data    

def preprocess_data(data, frequency_bands={}, windows=[],
                    reference=None, power_spectrum=False,
                    savepath=None, save_incl_raw=False):

    # investigate_pink_noise2(data)

    data = remove_data(data)
    data = clean_data(data)

    # Reference electrodes
    if reference != None:
        data = apply_reference(data, reference)
        
    # Filter data
    if any(frequency_bands):
        data = apply_filters(data, frequency_bands)

    # Functions to apply when the data is being windowed
    window_fns = {
        'psd': get_power_spectrum,
        }

    # Create windows
    if any(windows):
        data = create_windows(data, windows,
                              window_fns=window_fns)
    
    # Save
    # if savepath:
    #     if not save_incl_raw:
    #         data.pop('eeg')
    #     save_dict(savepath, data)

    return data

def extract_features(data, features_to_combine=[]):
    ''' frequencybands should also be treated as
    features instead of preprocessing.
    '''

    # Split and combine features per channel to use for learning
    features, feature_names = combine_features_with_channels(data, features_to_combine)
    
    features = trial_per_window(data, features)

    # Add extra info
    features['fs'] = data['fs']
    features['window_size'] = data['window_size']
    features['frameshift'] = data['frameshift']
    features['feature_names'] = feature_names
    features['channel_names'] = data['channel_names']

    if 'channel_locations' in data.keys():
        features['channel_locations'] = data['channel_locations']
    
    # Save in a dictionary
    results = {}
    if data['subject'] not in results.keys():
        results[data['subject']] = {}
    if data['experiment_type'] not in results[data['subject']]:
        results[data['subject']][data['experiment_type']] = {}

    results[data['subject']] \
           [data['experiment_type']] \
           [data['experiment_date']] = features

    return results

def perform_statistics(features):
    # Label permutation test

    #  1 Get labels
    #  2 Randomly permute labels (because H0 == random output from the 'model')
    #  3 Take the folds and calculate auc
    #  4 Repeat 10000 times and get the 95% highest value
    #  5 Compare mean auc with aucs of trained models. If higher, then signifantly above chance.
    #       this is allowed because the base distribution is the same!

    labels = np.unique(labels, return_inverse=True)[1]
    # One hot encode
    n_classes = np.max(labels)+1
    labels = np.eye(n_classes)[labels]

    n_permutations = 10000
    mean_aucs = []
    for _ in range(n_permutations):
        repetitions = 10
        test_size = 1/repetitions
        test_samples = int(test_size*labels.shape[0])
        
        aucs = []
        for rep in range(repetitions):
            test_idc = np.arange(rep*test_samples, rep*test_samples+test_samples)
            
            permuted_idc = test_idc.copy()
            np.random.shuffle(permuted_idc)  # In place, so no need to assign

            fold = labels[test_idc, :].copy()
            permuted_fold = labels[permuted_idc, :].copy()

            try:
                auc = roc_auc_score(fold, permuted_fold, multi_class='ovo')
            except Exception:
                # If a class is not present in a fold: get auc for included folds
                included_classes = ~np.all(fold==fold[0], axis=0)
                auc = roc_auc_score(fold[:, included_classes], permuted_fold[:, included_classes])

            aucs += [auc]

        mean_aucs += [np.mean(aucs)]
        
    # Get the 95% Cutoff (== 95th value of sorted array)
    cutoff_value = np.sort(mean_aucs)[int(len(mean_aucs)*.95)]
    print('Cutoff value: {:.3f} [{:d} permutations]'.format(cutoff_value, n_permutations))

    return features

def train_score_evaluate(data):
    
    x, y = combine_data_to_x_y(data)

    classifier = LDA()
    repetitions = 10
    
    # Learn and score
    scores = []
    for rep in range(repetitions):
        train_x, train_y, test_x, test_y = get_train_test(x, y,
                                                          shuffle=False,
                                                          stratify=False,
                                                          rep=rep,
                                                          repetitions=repetitions,
                                                          print_dist=True,
                                                          random_state=None)
        clf, test_y_hat = classifier.train(train_x, train_y, test_x)
        scores += [classifier.score(clf, train_x, train_y, 
                                      test_x, test_y,
                                      test_y_hat,
                                      protocol='1v1')]
        
        print('.', end='', flush=True)

    # Evaluate
    results = classifier.evaluate(scores)
    return results

def prepare_results_matrix(results, locs, count_locs=False, savepath=None):
    import pandas as pd
    df = pd.DataFrame([pd.Series([ppt, type_, idx_dt, name, score], 
                                  index=['ppt', 'exp', 'n_meas', 'score_name', 'score_value']) \
                           for ppt, data_ppt in results.items() \
                           for type_, data_type in data_ppt.items() \
                           for idx_dt, (_, data_dt) in enumerate(data_type.items()) \
                           for name, score in data_dt.items() \
                           if name in ['AUC [Links vs Rechts]', 'AUC [Links vs Rest]', 'AUC [Rechts vs Rest]', 'AUC [Move vs Rest]']],
                     columns=['ppt', 'exp', 'n_meas', 'score_name', 'score_value'])
    if not count_locs:
        df_locs = pd.DataFrame([pd.Series([ppt, contact_name, location_name], 
                                        index=['ppt', 'contact', 'loc']) \
                                for ppt, locations in locs.items() \
                                for contact_name, location_name in locations.items()])
    else:
        df_locs = pd.DataFrame(columns=['ppt', 'loc', 'loc_count'])
        for ppt, locations in locs.items():
            unique, counts = np.unique(list(locations.values()), return_counts=True)
            for i in range(len(unique)):
                df_locs = df_locs.append(pd.Series([ppt, unique[i], counts[i]], index=['ppt', 'loc', 'loc_count']),
                                        ignore_index=True)

    df = df.merge(df_locs, on='ppt')
    if savepath:
        df.to_excel('{}\performance_locs_{}.xlsx'.format(savepath, 'counted' if count_locs else ''))

    return df


if __name__ == "__main__":
    
    all_freq_bands = [
        # {"beta":  [12, 30]},
        # {"gamma": [55, 90]},
        {"beta":  [12, 30], "gamma": [55, 90]},
        # {"HFO": [90, 200]},
        # {'alpha': [8, 12]},
    ]

    for freq_bands in all_freq_bands:
       
        signal_reference = None  # ['CER', ]
        windows = [1, 0.1] # [window size, frameshift]

        features_to_combine = ['frequency_bands']
        subjects_to_exclude = [] # Complete
        experiment_type_to_exclude = [] # ['imagine', 'grasp']'
        reload_raw = 1
        load_locations = 0
        save_results = 1
        save_raw_data = 0
        power_spectrum = 0 

        path_drive = r'<some_path>' # Local external hard 
        path_local = r'<some_path>'

        # Path raw --> Drive
        # Path processed --> local
        ext = 'xdf' if reload_raw else 'pkl'
        main_path = path_drive if reload_raw else path_local
        filenames = get_filenames(main_path, ext, 
                                keywords=['grasp', 'imagine'],
                                exclude=['speech'])

        electrode_filenames = get_filenames(path_drive, 'csv', keywords=['electrode_locations']) \
                            if load_locations else []

        results = {}
        channel_locs = {}
        for filename in filenames:
            if is_file_to_skip(filename, subjects_to_exclude, experiment_type_to_exclude, []):
                continue

            if reload_raw:
                data = load_raw_data(filename)

                if data == None:
                    continue

                if load_locations and \
                    not is_file_to_skip(filename, subjects_to_exclude,
                                        experiment_type_to_exclude,
                                        electrode_filenames):
                    data = load_location_files(data, electrode_filenames)

                data = preprocess_data(data,
                                    reference=signal_reference,
                                    frequency_bands=freq_bands,
                                    windows=windows,
                                    power_spectrum=power_spectrum,
                                    savepath=path_local,
                                    save_incl_raw=save_raw_data)
            else:
                data = load_pickle(filename)
                if data == None:
                    continue
            
            features = extract_features(data, features_to_combine) 
            
            features = perform_statistics(features)

            # Explore, learn, plot
            for subj, exps in features.items():
                if subj in subjects_to_exclude:
                    continue
                if subj not in results.keys():
                    results[subj] = {}
                for exp, dates in exps.items():
                    if exp in experiment_type_to_exclude:
                        continue
                    path_exp = '{}/{}/{}'.format(path_local, subj, exp)
                    if exp not in results[subj].keys():
                        results[subj][exp] = {}
                    for date, data in dates.items():
                        savepath = '{}/{}/{}/{}'.format(path_local, subj, exp, date)
                        
                        print('\n{} {} {}'.format(subj, exp, date))
                        results[subj][exp][date] = train_score_evaluate(data)
                        
                        channel_locs[subj] = data.get('channel_names', {})
            savepath = '{}/{}'.format(path_local, subj)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

            print('')

        if not freq_bands.keys():
            band = 'raw'
        else:
            band = '_'.join(freq_bands.keys())
        
        savepath = r'.\Figures\{:s}_1000_100'.format(band)
        with open('{}\{}_results.pkl'.format(savepath, band), 'wb') as file:
            pickle.dump(results, file)

        with open('{}\{}_channels.pkl'.format(savepath, band), 'wb') as file:
            pickle.dump(channel_locs, file)
    # pretty_print(results,
    #              savepath=savepath) 
                #  savepath=path_local if save_results else None)
    # plot_bargrid_averaged_score(data, results, channel_locs, savepath,
    #              plot_all=True)
    # plot_bargrid_score_per_metric(data, results, channel_locs, savepath,
    #              plot_all=True)
    # plot_bargrid_score_per_ppt(data, results, [], savepath, plot_all=True)
    print('Done')
