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

import pandas as pd 
import glob
import joblib
import numpy as np
import datetime 
import matplotlib.pyplot as plt

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

stats = ['AHUP',
         'BYRL',
         'CALS',
         'CNPK',
         'CRIM',
         'NPIT',
         'OUTL',
         'UWEV',
         'VO46']

save_fig = True
stats = [stats[4]]
for stat in stats:
    df = pd.DataFrame()
    for f in sorted(glob.glob(f'HI_{stat}/*.csv')):
        print(f)
        df = df.append(pd.read_csv(f))

    t_key = np.array(df['Time(UTC)'])
    t_idx = [datetime.datetime.strptime(t,'%Y-%m-%dT%H:%M:%SZ') for t in t_key]
    df.index = t_idx
    del df['Time(UTC)']

    new_keys = {}
    for key in df.keys():
        new_keys[key] = key.strip()

    df = df.rename(columns=new_keys)

    hor_mag = np.array(np.sqrt(df['East']**2 + df['North']**2))
    # 10 day moving window
    hor_long = smooth(hor_mag, window_len=int(17280*10))
    hor_mag_detrend = hor_mag - hor_long

    df['hor_mag']         = hor_mag
    df['hor_long']        = hor_long
    df['hor_mag_detrend'] = hor_mag_detrend

    # 300s (5 min) smoothing
    window_len = 60
    df['hor_mag_detrend_smth'] = smooth(hor_mag_detrend, window_len=window_len)

    enu = {}
    comps = ['East', 'North', 'Up']
    for comp in comps:
        enu[comp] = smooth(df[comp], window_len=window_len)



    ndx = df.index > datetime.datetime(2018, 6, 1)
    df = df[ndx]
    joblib.dump(df, f'{stat}_HI.pkl')

    if save_fig:
        figout = f'{stat}_HI.png'
        plt.close('all')
        fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(10,8))
        lw = 0.2
        for k, comp in enumerate(comps):
            ax[k].plot(df.index, df[comp],  linewidth=lw, color='black')
            ax[k].plot(df.index, enu[comp][ndx], linewidth=lw, color='red')
            ax[k].set_ylabel(comp + ' [mm]')
        ax[3].plot(df.index, df['hor_mag_detrend'], linewidth=lw, color='red')
        ax[0].set_xlim([df.index[0], df.index[-1]])
        ax[0].set_title(stat, fontsize=16)
        fig.savefig(figout, dpi=300, bbox_inches='tight')
