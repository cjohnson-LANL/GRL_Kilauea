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
import os, sys
import numpy as np
import scipy.signal
import pandas as pd 
#import tensorflow as tf

def does_dir_exist(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def smooth(x_in, window_len=5, window='hanning'):
    """
    _smooth the time series for a given window length and window taper function
    
    window   : flat - boxcar or rectangle
               hanning - weighted cosine with tails to zero. Better at reducing high freq beyond the first lobe.
               hamming  - weighted cosine with heavy tails
               bartlett - Close representation of a triangle window
               blackman - a taper formed by using the first three terms of a summation of cosines
    
    scale    : [True,False] Window scale to get the new 1-sigma values after smoothing

    Usage : x (must be 1-dim, will convert list to np.array), window_len=11, window='flat'
    Return: smoothed np.array of equal length, scale, window_scale (optional)
    """
    x = np.array(x_in)
    if x.ndim != 1:
        raise ValueError("_smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[2.0*x[0]-x[int(window_len-1)::-1], x, 2.0*x[-1]-x[-1:int(-window_len):-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:  
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[int(window_len):int(-window_len+1)]


def get_collapse_time():
    # Use stations CALS and NPIT to get the collapse timing
    # These are both incomplete with a 2 day gap between them
    # Three events are manually added to fill gap

    f_GPS_CALS = '../GPS_highrate/CALS_HI.pkl'
    df_CALS   = joblib.load(f_GPS_CALS)
    df_CALS   = df_CALS[~df_CALS.index.duplicated()]
    df_CALS   = df_CALS.resample('5S').bfill()
    t_df_CALS = np.array([t.timestamp() for t in df_CALS.index]) 
    hor_mag_CALS  = np.array(df_CALS['hor_mag_detrend'])/1000.0
    # Colapse times from horizontal
    idx_collapse_CALS = np.where(np.diff(hor_mag_CALS) > 0.05)[0]

    f_GPS_NPIT = '../GPS_highrate/NPIT_HI.pkl'
    df_NPIT   = joblib.load(f_GPS_NPIT)
    df_NPIT   = df_NPIT[~df_NPIT.index.duplicated()]
    df_NPIT   = df_NPIT.resample('5S').bfill()
    t_df_NPIT = np.array([t.timestamp() for t in df_NPIT.index]) 
    hor_mag_NPIT  = np.array(df_NPIT['hor_mag_detrend'])/1000.0
    # Colapse times from horizontal
    idx_collapse_NPIT = np.where(np.diff(hor_mag_NPIT) > 0.05)[0]

    t_collapse0 = np.array([t_df_NPIT[t_] for t_ in idx_collapse_NPIT])
    t_collapse1 = np.array([t_df_CALS[t_] for t_ in idx_collapse_CALS])

    # Manually add 3 events in gap between the missing days in CALS and NPIT
    t_collapse = np.r_[t_collapse0,[1529338351, 1529420728, 1529504523],t_collapse1]
    return t_collapse

t_collapse = get_collapse_time()

idx_keep_collapse = np.r_[True, np.diff(t_collapse) > 3600]
t_collapse = t_collapse[idx_keep_collapse]

idx_use = int(sys.argv[1])
secs_per_label = 30 # 0.5 minute resample

# GPS
# Open and build dictionary with GPS displacements
gps_dir = '../GPS_highrate'
gps_stats = ['AHUP', 'BYRL', 'CRIM', 'UWEV']

gps_sps = 1./5.
gps_resample = int(gps_sps*secs_per_label)

gps_stats_ = gps_stats[idx_use]
df = joblib.load(os.path.join(gps_dir, f'{gps_stats_}_HI.pkl'))
df = df[~df.index.duplicated()]
df = df.resample('5S').bfill()
t_df = np.array([t.timestamp() for t in df.index])
# Horizontal magnitude
disp = np.array(df['hor_mag_detrend']) #/1000.0 # millimeter to meter

t_collapse = np.r_[t_df[0], t_collapse, t_df[-1]]
t_interval = np.vstack([t_collapse[:-1], t_collapse[1:]]).T

disp_smooth = np.zeros_like(disp)
for t0,t1 in t_interval:
    # print(t0,t1)
    idx0 = np.argmin(np.abs(t_df - t0))
    idx1 = np.argmin(np.abs(t_df - t1))
    disp_smooth[idx0:idx1] = smooth(disp[idx0:idx1], 120)

disp = disp_smooth.copy()

disp = disp.reshape(-1, gps_resample).mean(axis=1)
# gps[gps_stats_] = {'disp':hor_mag, 't':t_df}
t_disp = t_df.reshape(-1, gps_resample)[:,0]

# two_day = int(48*3600 / 30)
three_day = int(72*3600 / secs_per_label)

t_interval    = t_disp.reshape(-1, three_day)
disp_interval = disp.reshape(-1, three_day)
idx_interval  = np.arange(disp.size).reshape(-1, three_day).astype(int)

# Seismic
seis_dir = '../Seismic'
seis_stats = [['AHUD', ['EHE', 'EHN', 'EHZ']],
              ['BYL' , ['HHE', 'HHN', 'HHZ']],
              ['RIMD', ['HHE', 'HHN', 'HHZ']],
              ['UWE' , ['HHE', 'HHN', 'HHZ']],
             ]
seis_sps = 100
N = 570240000

# Three day
idx0 = idx_interval[np.array([0,     4,        8,         12,          16])].flatten()
idx1 = idx_interval[np.array([ 1,2,3,  5, 6, 7,  9, 10, 11,  13, 14, 15,  17, 18, 19, 20])].flatten()


print(f'Train N {idx0.size}')
# print(f'Val   N {idx1.size}')
print(f'Test  N {idx1.size}')

# seis = {}
# for stat, chan in seis_stats:
stat, chan = seis_stats[idx_use]
fin = os.path.join(seis_dir, stat+'_01_50.h5')
with h5py.File(fin, 'r') as f_seis:
    starttime = obspy.UTCDateTime(f_seis.attrs['starttime']).timestamp
    endtime = obspy.UTCDateTime(f_seis.attrs['endtime']).timestamp
    sampling_rate = float(f_seis.attrs['sampling_rate'])
    seis_e = f_seis[chan[0]][:N].astype(np.float32)
    seis_n = f_seis[chan[1]][:N].astype(np.float32)
    seis_z = f_seis[chan[2]][:N].astype(np.float32)
    npts = seis_z.size

t_stamp = np.arange(npts)/sampling_rate + starttime - 0.0049999

# Get the phase picks for the p waves
fdir = f"../Seismic/PhasePicks/detection_results_{stat}/{stat}_outputs"
df = pd.read_csv(os.path.join(fdir, "X_prediction_results.csv"))
pwave = []
for p in df['p_arrival_time']:
    if type(p) is str:
        pwave.append(obspy.UTCDateTime(p).timestamp)
pwave = np.array(pwave)

swave = []
for s in df['s_arrival_time']:
    if type(s) is str:
        swave.append(obspy.UTCDateTime(s).timestamp)
swave = np.array(swave)

# reshape the seismic data
seis_e_rs =  seis_e.reshape(-1,seis_sps*secs_per_label)
seis_n_rs =  seis_n.reshape(-1,seis_sps*secs_per_label)
seis_z_rs =  seis_z.reshape(-1,seis_sps*secs_per_label)

# 30 s sampling
t_stamp_rs  = t_stamp.reshape(-1,seis_sps*secs_per_label)[:,0]
t_stamp_rs_ = t_stamp.reshape(-1,seis_sps*secs_per_label)

# Build a mask to remove time segments with a p wave arrival
# All index are True -- Keep everything
# If waveform segment contains p or s wave set to False 0
p_idx = np.ones_like(t_stamp_rs)
for k, t in enumerate(t_stamp_rs_):
    if np.any((pwave >= t[0]) & (pwave < t[-1])):
        p_idx[k] = 0

s_idx = np.ones_like(t_stamp_rs)
for k, t in enumerate(t_stamp_rs_):
    if np.any((swave >= t[0]) & (swave < t[-1])):
        s_idx[k] = 0

# Both p and s are positive meaning there is no phase detected
ps_idx = (p_idx + s_idx) == 2

seis_enz_rs = np.zeros((*seis_e_rs.shape,3), dtype=np.float32)
seis_enz_rs[:,:,0] = seis_e_rs
seis_enz_rs[:,:,1] = seis_n_rs
seis_enz_rs[:,:,2] = seis_z_rs


# Time to failure
ttf = np.zeros(disp.size)
t_c_idx_last = 0
collapse_time = get_collapse_time()
for t_collapse_ in collapse_time:
    t_c_idx = np.argmin(np.abs(t_stamp_rs - t_collapse_))
    # print(t_c_idx)
    t_c = secs_per_label/3600 * np.arange(t_c_idx - t_c_idx_last)[::-1] #hours
    if t_c_idx - t_c_idx_last < 240: continue

    ttf[t_c_idx_last:t_c_idx] = t_c
    t_c_idx_last = t_c_idx
    
stat_use = seis_stats[idx_use][0]

with open(f'All_30s_data_{stat_use}_ps_idx_3d_block_v6_ENZ_smoothGPS_REV_GRL.txt','w') as ftxt:
    print(f'P  % keep {p_idx.sum()/p_idx.size}', file=ftxt)
    print(f'S  % keep {s_idx.sum()/s_idx.size}', file=ftxt)
    print(f'PS % keep {ps_idx.sum()/ps_idx.size}', file=ftxt)


dat_out = f'All_30s_data_{stat_use}_ps_idx_3d_block_v6_ENZ_smoothGPS_REV_GRL.h5'
with h5py.File(dat_out, 'w') as fout:

    fout.create_dataset('X_train', data=seis_enz_rs[idx0])
    fout.create_dataset('X_test',  data=seis_enz_rs[idx1])

    fout.create_dataset('disp_train', data=disp[idx0])
    fout.create_dataset('disp_test',   data=disp[idx1])

    fout.create_dataset('ttf_train', data=ttf[idx0])
    fout.create_dataset('ttf_test',   data=ttf[idx1])

    fout.create_dataset('idx_train', data=ps_idx[idx0])
    fout.create_dataset('idx_test',   data=ps_idx[idx1])

    fout.create_dataset('t_train', data=t_stamp_rs[idx0])
    fout.create_dataset('t_test',   data=t_stamp_rs[idx1])


