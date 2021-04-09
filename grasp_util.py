# Builtin
import bisect
import pickle
import sys
from datetime import datetime
from os import listdir, mkdir
from os.path import exists, isdir, getctime
from pathlib import Path
import re

# 3th party
import numpy as np
import pandas as pd
import scipy.signal
from mne.filter import filter_data
from scipy import fftpack
from scipy.stats import linregress
from scipy.stats import mode, zscore
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.model_selection import train_test_split
import yasa

# Local
paths = [r".\..\Utility\Helpers\xdf_reader"]
for path in paths:
    sys.path.insert(0, path)

from read_xdf import read_xdf

LINE_NOISE = 50


def locate_pos(available_tss, target_ts):
    # Locate the the closest index within a list of indices
    pos = bisect.bisect_right(available_tss, target_ts)
    if pos == 0:
        return 0
    if pos == len(available_tss):
        return len(available_tss)-1
    if abs(available_tss[pos]-target_ts) < abs(available_tss[pos-1]-target_ts):
        return pos
    else:
        return pos-1

def get_created_date(file, dt_format='%Y%m%d%H%M%S'):
    # Returns the formatted date of creation of a file
    return datetime.fromtimestamp(getctime(file)).strftime(dt_format)

def get_filenames(path_main, extension, keywords=[], exclude=[]):
    ''' Recursively retrieves all files with 'extension', 
    and subsequently filters by given keywords. 
    '''

    if not exists(path_main):
        print("Cannot access path <{}>. Make sure you're on the university network"\
                .format(path_main))
        raise NameError

    keywords = extension if len(keywords)==0 else keywords
    extension = '*.{}'.format(extension)
    files = [path for path in Path(path_main).rglob(extension) \
             if any(kw in path.name for kw in keywords)]

    if any(exclude):
        files = [path for path in files for excl in exclude \
                   if excl not in path.name]
    return files

def get_experiment_data(result):
    # TODO: Offset markers? (see load_grasp_data)
    marker_idx_exp_start = result['GraspMarkerStream']['data'].index(['experimentStarted'])
    marker_idx_exp_end = result['GraspMarkerStream']['data'].index(['experimentEnded'])

    eeg_idx_exp_start = locate_pos(result['Micromed']['ts'], 
                                result['GraspMarkerStream']['ts'][marker_idx_exp_start])
    eeg_idx_exp_end = locate_pos(result['Micromed']['ts'],
                                result['GraspMarkerStream']['ts'][marker_idx_exp_end])

    eeg = result['Micromed']['data'][eeg_idx_exp_start:eeg_idx_exp_end, :]
    eeg_ts = result['Micromed']['ts'][eeg_idx_exp_start:eeg_idx_exp_end]

    marker = result['GraspMarkerStream']['data'][marker_idx_exp_start:marker_idx_exp_end]
    marker_ts = result['GraspMarkerStream']['ts'][marker_idx_exp_start:marker_idx_exp_end]

    return eeg, eeg_ts, marker, marker_ts

def get_trials_info(eeg, eeg_ts, markers, marker_ts):
    # Create a label and trial numbers per timestamp
    # TODO: Change string labels to numerical

    # Find which markers correspond to the start and end of a trial
    trial_start_mask = [marker[0].split(';')[0]=='start' for marker in markers]
    trial_end_mask = [marker[0].split(';')[0]=='end' for marker in markers]

    # Find the indices corresponding to the start and end of the trial
    trial_idc_start = np.array([locate_pos(eeg_ts, trial) for trial in marker_ts[trial_start_mask]])
    trial_idc_end = np.array([locate_pos(eeg_ts, trial) for trial in marker_ts[trial_end_mask]])

    # Retrieve the corresponding labels per trial
    trial_labels = [marker[0].split(';')[1] for marker in markers if marker[0].split(';')[0] == 'start']

    # Map the label and trial number per index.
    trial_seq = [0] * eeg.shape[0] # Trial labels sequential
    trial_nums = [0] * eeg.shape[0]
    for i, idx_start in enumerate(trial_idc_start):
        trial_seq[idx_start:trial_idc_end[i]] = [trial_labels[i]] * (trial_idc_end[i]-idx_start)
        trial_nums[idx_start:trial_idc_end[i]] = [i] * (trial_idc_end[i]-idx_start)

    return np.array(trial_seq), np.array(trial_nums)

def save_dict(path, data):
    try:
        folder_path = '{}/{}'.format(path, data['subject'])
        if not isdir(folder_path):
            mkdir(folder_path)
        filepath = r'{}/seeg_{}_{}.pkl'\
                        .format(folder_path, 
                                data['experiment_type'], 
                                data['experiment_date'])
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print('Failed save retrieved data from {}!'.format(data['subject']))
        print(e)

def is_file_to_skip(filename, 
                    subjects_to_exclude,
                    experiment_type_to_exclude,
                    electrode_filenames=[]):
    if any([subj in str(filename) for subj in subjects_to_exclude]):
        return True
    if any([exp in str(filename) for exp in experiment_type_to_exclude]):
        return True
    
    if len(electrode_filenames) > 0:
        # If there is no electrode_location file with the same KHxx identifier, return True
        pattern = '(?<=sEEG\\\\)\w\w(\d|\d\d)(?=\\\\)' # Finds KHx or KHxx if it follows directly after 'sEEG/' 
                                                    # and is follow by a backslash
        nums = [['{:02d}'.format(int(re.findall(pattern, str(elec_filename))[0]))] for elec_filename in electrode_filenames]                               
        matches = [re.findall(pattern, str(filename)) == num for num in nums]
        if not any(matches):
            print('Skipping {}'.format(str(filename)))
            return True

    return False

def load_seeg(file, savepath=None):
    ''' Loads xdf file and returns a dict with all necessary information'''
    print('Loading file: {}'.format(file))
    
    result, raw_data = read_xdf(str(file))

    # investigate_pink_noise2(result['Micromed']['data'])
    
    eeg, eeg_ts, markers, markers_ts = get_experiment_data(result)
    trials, trial_nums = get_trials_info(eeg, eeg_ts, markers, markers_ts)

    multiple_measurements = 'kh' not in file.parts[-2]

    seeg = {}
    seeg['subject'] = file.parts[-2] if not multiple_measurements else file.parts[-3]
    seeg['experiment_type'] = file.parts[-1].split('.xdf')[0]
    seeg['experiment_date'] = file.parts[-2] if multiple_measurements else get_created_date(file) # Returns created date if no date folder is present
    seeg['channel_names'] = result['Micromed']['channel_names']
    seeg['eeg'] = eeg
    seeg['eeg_ts'] = eeg_ts
    seeg['trial_labels'] = trials
    seeg['trial_numbers'] = trial_nums
    seeg['fs'] = result['Micromed']['fs']
    seeg['dtype'] = result['Micromed']['data_type']
    seeg['first_ts'] = result['Micromed']['first_ts']
    seeg['last_ts'] = result['Micromed']['last_ts']
    seeg['total_stream_time'] = result['Micromed']['total_stream_time']
    seeg['samplecount'] = result['Micromed']['sample_count']
    seeg['features'] = {}

    if savepath != None:
        save_dict(savepath, seeg)

    return seeg

def load_pickle(filename):
    data = None
    
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
         print("Failed to load: {}, due to exception: {}". format(str(filename), e))

    return data

def load_location_files(data, electrode_filenames):
    # Find matching file
    pattern = '(?<=sEEG\\\\)\w\w(\d|\d\d)(?=\\\\)' # Finds KHx or KHxx if it follows directly after 'sEEG/' 
                                                    # and is follow by a backslash
    file = [elec_filename for elec_filename in electrode_filenames \
                if data['subject'][2:] == '{:02d}'.format(int(re.findall(pattern, str(elec_filename))[0]))][0]

    df = pd.read_csv(file)
    data['channel_locations'] = dict(zip(df['electrode_name_1'], df['location']))
    return data

def remove_data(data):
    # Remove channels without information
    #   -> Channels names Exx only contain noise
    #   -> Channels with a + do not hold neural information 

    chs_to_keep = [ch for ch in data['channel_names']
                    if '+' not in ch
                    and 'E' != ch.rstrip('1234567890')]
    chs_to_keep_idc = [data['channel_names'].index(ch)
                        for ch in chs_to_keep]
    data['eeg'] = data['eeg'][:, chs_to_keep_idc]
    data['channel_names'] = chs_to_keep
    return data

#######################

def hilbert3(x):
    return scipy.signal.hilbert(x, 
                fftpack.next_fast_len(len(x)), 
                axis=0)[:len(x)]

def laplacian_reference(seeg, channels):
    seeg_ref = seeg.copy()
    electrodes = np.unique([ch.rstrip('0123456789') for ch in channels
                                                    if '+' not in ch])
    for electrode in electrodes:
        electrode_channels = [ch for ch in channels if electrode in ch] # TODO: doesnt guarantee unique (if channels names include E and LE)
        adjacent_channels = []
        for i, ch in enumerate(electrode_channels):
            current_ch = channels.index(electrode_channels[i])
            if i==0:
                adjacent_channels = [channels.index(electrode_channels[i+1])]
            elif i==(len(electrode_channels)-1):
                adjacent_channels = [channels.index(electrode_channels[i-1])]
            else:
                adjacent_channels = [channels.index(electrode_channels[i-1]),
                                     channels.index(electrode_channels[i+1])]
            ch_average = np.mean([seeg[:, ch] for ch in adjacent_channels])
            seeg_ref[:, current_ch] = seeg[:, current_ch] - ch_average
    return seeg_ref

def clean_data(data, cutoff_l=0.5, cutoff_h=None):
    data['eeg'] = scipy.signal.detrend(data['eeg'], axis=0)

    print('Filtering data from {} to {}'.format(cutoff_l, cutoff_h))
    data['eeg'] = filter_data(data['eeg'].T, data['fs'],
                           cutoff_l, cutoff_h, verbose=0).T
    return data

def common_electrode_reference(seeg, channels):
    seeg_ref = seeg.copy()
    electrodes = np.unique([ch.rstrip('0123456789') for ch in channels])

    for electrode in electrodes:
        electrode_channels = [channels.index(ch) for ch in channels if electrode in ch]
        seeg_ref[:, electrode_channels] = np.subtract(seeg_ref[:, electrode_channels], 
                                                      np.mean(seeg[:, electrode_channels],
                                                              axis=1, keepdims=1))

    return seeg_ref

def common_average_reference(seeg, channels):
    seeg_ref = seeg.copy()
    idc = [channels.index(ch) for ch in channels]
    seeg_ref[:, idc] = seeg_ref[:, idc] - np.mean(seeg_ref[:, idc], axis=1, keepdims=1)
    return seeg_ref

def apply_reference(data, reference_type='cer',
                    exlude_chs=['EKG+', 'MKR2+']):
    ''' 
    Laplacian:
        For each electrode:
            1) Retrieve 2 adjacent electrodes
            2) Remove the average
    '''
    reference_type = reference_type.lower()
    channels = data['channel_names'].copy() # Can't remove chs here, because it will change the indices, specifically for CAR
    # channels = [ch for ch in data['channel_names'] if ch not in exlude_chs]

    if reference_type == 'laplacian':
        seeg = laplacian_reference(data['eeg'], channels)
    elif reference_type == 'cer':
        seeg = common_electrode_reference(data['eeg'], channels)
    elif reference_type == 'car':
        seeg = common_average_reference(data['eeg'], channels)

    data['eeg'] = seeg

    return data

def get_line_noise_filters(band, fs, line_frequency, frequency_offset=2):
    # Finds all possible harmonics and returns all the corresponding 
    # notch filters with frequency_offset.
    line_harmonics = np.array([line_frequency*i for i in range(int(fs/2/line_frequency))])
    harmonics = np.where((line_harmonics>band[0]) & (line_harmonics<band[1]))[0]
    return [[line_frequency*h+frequency_offset, \
             line_frequency*h-frequency_offset] for h in harmonics]

def filter_eeg(data, 
               band,
               line_freq=50):
    eeg = data['eeg'].copy()

    # Filter
    # Check for Nyquist-Shannon Theorem
    if any(f >= (data['fs']/2) for f in band):
        print("Value within band {} doesn't adhere to Nyquist-Shannon theorem with Fs={}. Returning None..." \
                .format(band, data['fs']))
        return
    
    print('Filtering bandpass: {}'.format(band))

    # Filter eeg
    eeg = filter_data(eeg.T, data['fs'],
                      band[0], band[1],
                       method='iir',
                      verbose=0).T
    
    # Filter for line noise and its harmonics if neccesary
    line_noise_filters = get_line_noise_filters(band, data['fs'], line_freq)
    for noise_filter in line_noise_filters:
        # Band-Stop filter. Adjust offset keyword argument of
        # get_line_noise_filters() to set the width of the
        # filter.
        eeg = filter_data(eeg.T, data['fs'],
                          noise_filter[0], noise_filter[1],
                          method='iir',
                          verbose=0).T

    # eeg = np.log(eeg**2)+0.01

    eeg = abs(hilbert3(eeg))

    return eeg

def apply_filters(data, frequency_bands):
    # Handlers for applying different filters 
    # and create different feature bands
    frequency_bands = frequency_bands.copy()

    for k, v in frequency_bands.items():
        frequency_bands[k] = \
            {"band": v,
             "data": filter_eeg(data,
                                band=v,
                                line_freq=LINE_NOISE)}
    # Check for failed filters
    keys_to_delete = [k for k, v in frequency_bands.items()\
                        if v['data'] is None]
    for key in keys_to_delete:
        del frequency_bands[key]

    data['features']['frequency_bands'] = frequency_bands
    return data

def get_power_spectrum(eeg, fs):

    eeg = eeg - eeg.mean()

    ps = np.array(
            [np.abs(np.fft.rfft(eeg[:, ch]))**2
                for ch in range(eeg.shape[1])])

    ps = np.log10(ps)

    fft_freqs = np.fft.rfftfreq(eeg.shape[0], d=1/fs)
    return {'freqs': fft_freqs, 
            'ps': ps}
    # return np.array(ps)

def investigate_pink_noise2(data):
    # https://github.com/raphaelvallat/yasa/blob/master/yasa/spectral.py
    # NOTE: An estimate of the original PSD can be calculated by simply
    #       adding ``psd = psd_aperiodic + psd_oscillatory``.
    import matplotlib.pyplot as plt

    fs = 1024
    data_sh = data[:fs*60]
    freqs, psd_aperiodic, psd_osc, fit = \
        yasa.irasa(data=data_sh.T, 
                    sf=1024,
                    ch_names=None,
                    band=(1, 100))

    plt.plot(freqs, abs(psd_aperiodic[0,:]))
    plt.plot(freqs, abs(psd_osc[0,:]))
    plt.show()
    print('done')

    return 0

def get_fractal_component(eeg, fs):
    '''NOTE: win_sec should be at least 2*1 / band[0], however,
             This is already windowed. 0.1 was the first window
             length for which is would work (which corresponds 
             to a minimum freq of 10 instead of 1.). Result of 
             the function is therefor likely unreliable, but on
             the other hand, the psd angle of 10Hz and higher 
             could still be the same. Maybe check visually. Alpha
             will be lower than its 'true' value then.
    '''
    freqs, psd_aperiodic, psd_osc, fit = \
        yasa.irasa(data=eeg.T, 
                   sf=fs,
                   band=(1, 100),
                   win_sec=0.1
                   )
                
    return fit['Slope']

def generate_complex_morlet_wavelet(t, f, n):
    # n = number of periods
    # f = frequency
    # t = time

    s = n / (2*np.pi*f)

    A = 1/((s*np.sqrt(np.pi))**(1/2))
    real = (-t**2)/(2*s**2)
    imag = (2*np.pi*f*t)
    cmw = A*np.exp(real + 1j*imag)
    return cmw

def calculate_windows(data, fs, window_length, frameshift, aggr='mean',
                      apply_per_window={}):
    ''' Slice the data into windows of size window_size and
    increment with frameshift. If frameshift < windows_size, 
    the windows will overlap. 
    The implemented window function is inherently also a
    resampling of the data, because a single value is taken
    for the whole window using the function aggr.
    
    data = timeseries to be slices
    fs   = sampling frequency
    windowlength = length of window in seconds
    frameshift   = length of each increment in seconds

    Note that a bit of data is lost by int(np)
    creating windows (size < window_length).
    '''

    # Convert to n_samples 
    window_length = window_length*fs
    frameshift = frameshift*fs

    # Add dimension to singleton array
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1)
    
    n_windows = int((data.shape[0]-window_length) / (frameshift)) # int() always floors
 
    # Determine how to aggregate values within a window
    fn_agg = mode if aggr=='mode' else np.mean

    # TODO: Make vectorized solution
    # windows = np.zeros((n_windows, data.shape[1]))
    windows = np.empty((n_windows, data.shape[1]), dtype=data.dtype)

    # Create matrices for variable amount of functions
    fns_output = {name: [] for name in apply_per_window.keys()}

    for current_window in range(n_windows):
        window_start_index = int(np.round(current_window*frameshift))
        window_end_index = int(np.round(window_start_index+window_length))
        if aggr=='mode':
            windows[current_window, :] = mode(data[window_start_index:window_end_index, :])[0][0]
        else:
            windows[current_window, :] = np.mean(data[window_start_index:window_end_index, :],
                                                 axis=0)
            for name, fn in apply_per_window.items():
                fns_output[name] += [fn(data[window_start_index:window_end_index, :],
                                        fs)]

    windows = np.squeeze(windows)
    fns_output = {name: np.stack(array) for name, array in fns_output.items()}

    if any(fns_output):
        return windows, fns_output
    else:
        return windows

def create_windows(data, windows, window_fns):
    # Handler for windowing of the data.
    # TODO: featurestacking? See process_grasp_data
    # TODO: Change windows for spectrogram

    data['window_size'] = windows[0] # Ms
    data['frameshift'] = windows[1]

    # Create windows for eeg
    if 'frequency_bands' in data['features'].keys():
        for band_name, values in data['features']['frequency_bands'].items():
            data['features']['frequency_bands'][band_name]['data'] = \
                calculate_windows(values['data'], 
                               data['fs'],
                               window_length=windows[0],
                               frameshift=windows[1])
    
    result, fns_output = calculate_windows(
                                    data['eeg'], 
                                    data['fs'],
                                    window_length=windows[0],
                                    frameshift=windows[1],
                                    apply_per_window=window_fns)
    data['eeg'] = result # Note that this is effectively a low-pass
                         # filter by .5*fs/window_length
    data['features'].update(fns_output) # Merges the two dictionaries in place

    # Create windows for labels and trial numbers
    data['trial_labels'] = calculate_windows(data['trial_labels'],
                                             data['fs'],
                                             window_length=windows[0],
                                             frameshift=windows[1],
                                             aggr='mode')
    data['trial_numbers'] = calculate_windows(data['trial_numbers'],
                                             data['fs'],
                                             window_length=windows[0],
                                             frameshift=windows[1],
                                             aggr='mode')                                             

    return data

    #################

def combine_features_with_channels(data, features_to_combine, exclude=[]):
    ''' Combines all features into a single matrix
    
    input  = data: [samples x ch] * n_features
    output = data: [samples x ch*features] '''
    
    # TODO: add exclude
    # ADD chxfeatures to preprocessed eeg

    # ata['eeg']
    # feature_names = data['channel_names']
    
    
    # feature_names = np.array([])
    # features = np.array([])

    feature_names = []
    features = []

    if any(features_to_combine):
        for feature in features_to_combine:
            for name, values in data['features'][feature].items():
                feature_names += [name]
                features += [values['data']]
        
        features = np.concatenate(features, axis=1)
        feature_names = ['{}_{}'.format(band, ch) \
                            for band in feature_names \
                            for ch in data['channel_names']]

    if 'eeg' in data.keys():
        if any(features_to_combine):
            pass
            # features = np.concatenate([data['eeg'], features], axis=1)
            # feature_names = data['channel_names'] + feature_names
        else:
            features = data['eeg']
            feature_names = data['channel_names']
            
    return features, feature_names

def get_mask_equal_trial_size(trials, trial_length, rest_trial_number=0):
    ''' Finds if there is a difference in trial length. Trials are
    extended or truncated on a mask, such that the changes on the
    actual data can be done vectorized. 
    NOTE: The current implementation re-uses data from the start
          of the subsequent rest trial if the trial is too short. 
          If the trial is too long, the surplus of trials is 
          removed, potentially removing some information
          This behaviour might be changed to extend to the 
          longest present trial, instead of the pre-set trial
          length.

    # Input:
        trials = numbered trials for each index
                    e.g. [1,1,1, 0,0,0, 2,2,2, 0,0,0, ...]
        trial_length = Length to shape all trials to
        rest_trial_number = numbers to remove.
    '''
    
    # TODO: Ommit trial when size is x% different than intented length

    # Find the length per trial
    unique, counts = np.unique(trials, return_counts=True)
    unique = dict(zip(unique, counts))
    if rest_trial_number in unique: 
        del unique[rest_trial_number]

    mask = trials != 0

    n_trials = len(unique.keys())
    for trial_number, sample_count in unique.items():
        if sample_count != trial_length:
            # Find index of last sample in trial
            idx = max(np.where(trials==trial_number)[0])

            # Determine the discrepancy in samples
            n_missing_samples = trial_length - sample_count

            # Change the next n_missing_samples to True or False,
            # depending on if the sample is too short or too long.
            if n_missing_samples > 0:
                if idx+n_missing_samples+1 > len(mask):
                    print('Not enough data available for the last trial. Omitting that trial...')
                    mask[-sample_count:] = False
                    n_trials = n_trials - 1
                mask[idx+1:idx+n_missing_samples+1] = True
            else:
                mask[idx+n_missing_samples+1:idx+1] = False
                
    assert mask.sum() == n_trials * trial_length, 'Trials in updated mask are not equal!'

    return mask, n_trials

def get_label_rest_trials(trials_labeled, rest_label):
    ''' Labels the unlabeled rest trials.

    Order of operations
        - Retrieve all trial indices from the rest trials
        - Calculate the difference to determine where the 
            trials start and end
        - Each difference that is larger than 1 means that
            that is the start of a rest trial. To determine
            the length of each trials, calculated the 
            difference of each difference > 1.
        - Select indices at diff > 1 to retrieve the indices
            at the start of each rest trial in the original
            array (trials_labeled)
        - Add the lenght of each trial to retrieve the 
            end of each trial
        - Create a labeled array with a number for each trial  
    '''

    trials_rest = np.where(trials_labeled==rest_label)[0]
    index_diff = np.diff(np.r_[0, trials_rest])

    # Get length, start and end of trials
    trial_rest_lengths = np.diff(np.r_[np.where(index_diff>1)[0], len(index_diff)]) 
    trial_rest_start_idc = trials_rest[np.where(index_diff>1)] 
    trial_rest_end_idc = trial_rest_start_idc + trial_rest_lengths
        
    # Create labeled array
    rest_labeled = np.zeros(trials_labeled.shape[0])
    for i in range(trial_rest_start_idc.shape[0]):
        rest_labeled[trial_rest_start_idc[i]:trial_rest_end_idc[i]] = i+1

    return rest_labeled

def split_per_trial(data, features):
    ''' Split the 2d data per trial for all labels
    included

    input  = data:   [samples x ch*features]
    output = data:   [samples x ch*features x trials]
             labels: [labels]  

    TODO: Make this more automated
    '''
    # trial_length = sum(data['trial_numbers']==1)  # Get the length of the first trial #TODO Assumes first trial is correct...
    # trial_rest_length = \
    #      np.min(np.where(data['trial_numbers']==2)) - \
    #     (np.max(np.where(data['trial_numbers']==1))+1)  # Length between trial 1 and 2
    # trial_size = 40
    # trial_size_rest = 20

    # First trial is labeled zero, which is the same label as
    # rest, so this fix adds 1 to the trial numbers to make it
    # easier to retrieve the right trials.
    data['trial_numbers'] = np.where(data['trial_labels'] != '0',\
                                     data['trial_numbers'] + 1,\
                                     data['trial_numbers'])

    # TODO: Add window time to LSL stream information.
    # TODO: NOTE: This code assumes that the first trial is a move trial.
    trial_sizes = np.diff(np.nonzero(np.diff(data['trial_numbers']))).squeeze()
    trial_0_size = trial_sizes[np.arange(0, 10, 2)].mean(dtype=np.int32)
    trial_1_size = trial_sizes[np.arange(1, 10, 2)].mean(dtype=np.int32)
    trial_size = trial_0_size if data['trial_labels'][0] != '0' else trial_1_size
    trial_size_rest = trial_1_size if data['trial_labels'][0] != '0' else trial_0_size
    print('Slicing trials to size: {}, and rest to size {}'.format(trial_size, trial_size_rest))


    # Non rest trials
    mask, n_trials = get_mask_equal_trial_size(data['trial_numbers'], trial_size)

    if len(features.shape) == 3:
        dims = (n_trials, trial_size, features.shape[1], features.shape[2])
    else:
        dims = (n_trials, trial_size, -1)
    trials_non_rest = np.reshape(features[mask, :], dims)

    # Rest trials:
    rest_trials_labeled = get_label_rest_trials(data['trial_labels'], rest_label='0')
    mask, n_trials = get_mask_equal_trial_size(rest_trials_labeled, trial_size_rest)
    
    if len(features.shape) == 3:
        dims = (n_trials, trial_size_rest, features.shape[1], features.shape[2])
    else:
        dims = (n_trials, trial_size_rest, -1)
    trials_rest = np.reshape(features[mask, :], dims)
    
    # Retrieve trial labels
    # Find indices of non rest numbers and grab the first indices
    # of the non_rest number and extract the corresponding label
    non_rest_idc = (np.where(data['trial_numbers']!=0))[0] 
    non_rest_labels = data['trial_labels'][non_rest_idc[np.diff(np.r_[-2, non_rest_idc])>1]]

    result = {
        'rest': trials_rest,
        'non_rest': trials_non_rest,
        'non_rest_labels': non_rest_labels}

    # result = {'non_rest': {'data': trials_non_rest,
    #                        'labels': non_rest_labels},
    #           'rest': {'data': trials_rest,
    #                    'labels': []}}
    return result

def trial_per_window(data, features):
    trial_labels = data['trial_labels']
    trial_labels = np.where(trial_labels=='0', 'Rest', trial_labels)

    # Create datasets per trial label
    unique_labels = np.unique(trial_labels)
    results = {label: np.squeeze(features[np.where(trial_labels==label), :])\
               for label in unique_labels}

    # Add the original data to the datasets per trial dictionary
    results['Original'] = {'data': features,
                           'label': trial_labels}    

    return results

def get_statistics(a, b, test,
                   significance_level=0.05, 
                   bonferonni=True, n_tests=None):
    if test=='ttest':
        t, p = ttest_ind(a, b, axis=0)
    if bonferonni:
        # TODO: check if this is the right amount of correction.
        if n_tests == None:
            n_tests = a.shape[1]
        significance_level /= n_tests
    
    is_significant = np.where(p<significance_level, 1, 0)
    return is_significant, significance_level

def perform_stat_test(group_a, group_b, 
                      significance_level=0.05, 
                      bonferonni=True):
    '''
    data = data
    metric = [erp, band, anything else]

    Check the type of difference, Monte Carlo? Or simple t-test/permutation?
    
    '''
    # group_a = features[group_a_label]
    # group_b = features[group_b_label]

    n_features = group_a.shape[-1]
    max_len = min(group_a.shape[1], group_b.shape[1])
    significance_matrix = np.zeros((max_len, n_features))

    for idx in range(n_features):
        

        is_significant, level = get_statistics(group_a[:, :max_len, idx], 
                                            group_b[:, :max_len, idx],    
                                            test='ttest',
                                            bonferonni=True,
                                            significance_level=0.05)

        significance_matrix[:, idx] = is_significant

    # features['{}_vs_{}'.format(group_a_label, group_b_label)] = significance_matrix
   
    return significance_matrix

######################
    
def get_train_test(x, y, test_size=0.2,
                   stratify=False, shuffle=True,
                   rep=0, repetitions=10,
                   print_dist=False,
                   random_state=None):

    if len(x.shape)==3:
        # Reshape to 2d problem if x is supplies in 3D.
        # Assumes trials are in the first dimension
        x = np.reshape(x, (-1, x.shape[1]*x.shape[2]))

    if shuffle==False:
        test_size = 1/repetitions
        test_samples = int(test_size*y.shape[0])
        test_idc = np.arange(rep*test_samples, rep*test_samples+test_samples)

        test_x = x[test_idc, :].copy()
        test_y = y[test_idc].copy()
        # print(x.shape, min(test_idc), max(test_idc))
        train_x = np.delete(x, test_idc, axis=0) # Delete returns a copy
        train_y = np.delete(y, test_idc)
    else:
        stratify = y if stratify else None
        train_x, test_x, train_y, test_y = train_test_split(x, y,
                                                        test_size=test_size,
                                                        stratify=stratify,
                                                        shuffle=shuffle,
                                                        random_state=random_state)

    if print_dist and rep==0:
        print('Label distributions:')
        for name, value in {'train': train_y, 'test': test_y}.items():
            values, counts = np.unique(value, return_counts=True)
            print('{}: {}'.format(name, dict(zip(values, counts))))
    return train_x, train_y, test_x, test_y

def combine_data_to_x_y(data):

    # x = np.concatenate((data['Links'], data['Rechts'], data['Rest']), axis=0) 
    # y = np.array(['Links'] * data['Links'].shape[0] \
    #            + ['Rechts'] * data['Rechts'].shape[0] \
    #            + ['Rest'] * data['Rest'].shape[0])
    x = data['Original']['data']
    y = data['Original']['label']

    return x, y

def get_rotb_frequencies():
    ''' Supposedly the independent oscillators,
    the averages of these oscilators allign perfectly
    on a natural logarithmic scale (see p114)
    
    Returned bands in order:
    slow 4      [15, 40] s
    slow 3      [5, 15] s
    slow 2      [2, 5] s
    slow 1      [0.7, 2] s
    delta       [1.5 - 4] Hz
    theta       [4, 10] Hz
    beta        [10, 30] Hz
    gamma       [30, 80] Hz
    fast        [80, 200] Hz
    ultra fast  [200, 600] Hz
    '''
    start = 0.05
    end = 141.48
    freqs = np.linspace(np.log(start), np.log(end), 9)
    diff_freqs = np.diff(freqs)[0]
    freqs = np.append(freqs, freqs[-1]+diff_freqs)
    
    return freqs
