'''
Christopher W Johnson
Los Alamos National Laboratory

Copyright 2024

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software. THE SOFTWARE IS 
PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
'''


import obspy
import h5py
import joblib
import matplotlib.pyplot as plt
import os,sys
import numpy as np
import scipy.signal
import scipy.stats

feat = [
        'cross_rate',
        'q40_60',
        'rms',
        'thresh_0.01',
        'thresh_0.001',
        ]

no_bp = ['fft_center',
         'freq_E_N',
         'freq_E_Z',
         'freq_N_Z']

no_bp_chan = ['freq_E_N',
             'freq_E_Z',
             'freq_N_Z']

stat =[['AHUD', ['EHE', 'EHN', 'EHZ']],
       ['BYL' , ['HHE', 'HHN', 'HHZ']],
       ['RIMD', ['HHE', 'HHN', 'HHZ']],
       ['UWE' , ['HHE', 'HHN', 'HHZ']]
       ]

bp = [
      [0.1,  49.0],
      [0.1,  1.0],
      [1.0,   4.0],
      [1.0,  10.0],
      [1.0,  49.0],
      [10.0, 20.0],
      [10.0, 49.0],
     ]

idx_use = int(sys.argv[1])
stat_use = stat[idx_use][0]
version = sys.argv[2]

n_features = 0
with open(f'{stat_use}_features_{version}.txt', 'w') as f:
    for feat_ in feat:
        if not feat_ in no_bp:
            for bp0, bp1 in bp:
                stat_, chan = stat[idx_use]
                for chan_ in chan:
                    n_features += 1
                    print(f'{feat_}_{bp0:3.1f}_{bp1:3.1f}_{stat_}_{chan_}', file=f)
        else:
            bp0 = 1.0
            bp1 = 40.0
            if feat_ in no_bp_chan:
                chan_ = 'None'
                n_features += 1
                print(f'{feat_}_{bp0:3.1f}_{bp1:3.1f}_{stat_}_{chan_}', file=f)
            else:
                stat_, chan = stat[idx_use]
                for chan_ in chan:
                    n_features += 1
                    print(f'{feat_}_{bp0:3.1f}_{bp1:3.1f}_{stat_}_{chan_}', file=f)

###########  Helper Funcs ###########
def do_trace_calcs(seis_in):
    seis_ts = seis_in.T
    shape0 = seis_ts.shape[0]
    shape1 = seis_ts.shape[1]

    # Time domain features
    # Filter 
    Hz = 100.
    df = 1./Hz

    # The bp_buterworth routine is vectorized
    p     = 2
    order = 4

    tr_filt = np.zeros((shape0*(len(bp)), shape1))
    tr_filt[:shape0] = seis_ts
    for j, (bp0, bp1) in enumerate(bp[1:]):
        i0 = (j+1)*shape0
        i1 = i0+shape0
        tr_filt[i0:i1] = bp_butterworth(seis_ts, Hz, p, order, bp0, bp1)

    # Calculations on 2D matrix
    zeroCrossRate = np.nansum(np.diff(np.sign(tr_filt), axis=1) == 0, axis=1).astype(np.float32)
    q40, q60 = np.nanquantile(tr_filt, (0.40, 0.60), axis=1).astype(np.float32)
    q40_60   = q60-q40
    t_rms = rms(tr_filt).astype(np.float32)

    # Thresholds
    v_thresh_arr = np.array([0.01, 0.001])
    thresh = []
    for v_thresh_arr_ in v_thresh_arr:
        thresh.extend(((tr_filt - v_thresh_arr_) > 0).sum(axis=1)/tr_filt.shape[1])
    thresh = np.array(thresh).astype(np.float32)

    features_out = np.hstack([
                            zeroCrossRate,
                            q40_60,
                            t_rms,
                            thresh,
                            ])

    # Basic cleanup for NaN/Inf
    features_out[np.isnan(features_out)] = 0.0
    features_out[np.isinf(features_out)] = 0.0

    return features_out


def bp_butterworth(trace, sps, p, order, cnr1, cnr2):
    f_Ny = sps / 2.
    sos = scipy.signal.butter(order, [cnr1/f_Ny, cnr2/f_Ny], btype='bandpass', output='sos')
    if p == 2:
        firstpass = scipy.signal.sosfilt(sos, trace)
        return scipy.signal.sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return scipy.signal.sosfilt(sos, trace)

def rms(tr):
    '''
    Root mean square
    '''
    return np.sqrt(np.sum(tr**2, axis=1)/float(len(tr)))


hdf5_file = f'../../Data/All_30s_data_{stat_use}_ps_idx_3d_block_v6_ENZ_smoothGPS_REV_GRL.h5'
with h5py.File(hdf5_file, 'r') as data:
    X_train = data['X_train'][:]
    X_test  = data['X_test'][:]

    ttf_train = data['ttf_train'][:]
    ttf_test  = data['ttf_test'][:]

    disp_train = data['disp_train'][:]
    disp_test  = data['disp_test'][:]

    idx_train = data['idx_train'][:]
    idx_test  = data['idx_test'][:]

    t_train = data['t_train'][:]
    t_test  = data['t_test'][:]


X_train = X_train[idx_train]
X_test  = X_test[idx_test]

# Standardize
X_train = X_train / X_train.std()
X_test  = X_test  / X_test.std()

ttf_train = ttf_train[idx_train]
ttf_test  = ttf_test[idx_test]

disp_train = disp_train[idx_train]
disp_test  = disp_test[idx_test]

t_train = t_train[idx_train]
t_test  = t_test[idx_test]

X_train_features = np.zeros([X_train.shape[0], n_features])
X_test_features  = np.zeros([X_test.shape[0],  n_features])

for j, X_train_ in enumerate(X_train):
    if np.mod(j,1000) == 0:
        print(f'Train :: {j} of {len(X_train)}')
    X_train_features[j] = do_trace_calcs(X_train_)

for j, X_test_ in enumerate(X_test):
    if np.mod(j,1000) == 0:
        print(f'Test :: {j} of {len(X_test)}')
    X_test_features[j] = do_trace_calcs(X_test_)


dat_out = f'{stat_use}_30s_data_features_{version}.h5'
with h5py.File(dat_out, 'w') as fout:
    fout.create_dataset('X_train', data=X_train_features)
    fout.create_dataset('X_test',  data=X_test_features)

    fout.create_dataset('ttf_train', data=ttf_train)
    fout.create_dataset('ttf_test',  data=ttf_test)

    fout.create_dataset('disp_train', data=disp_train)
    fout.create_dataset('disp_test',  data=disp_test)

    fout.create_dataset('t_train', data=t_train)
    fout.create_dataset('t_test',  data=t_test)
