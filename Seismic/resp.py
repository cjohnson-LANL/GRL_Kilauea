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
import glob
import h5py
import os,sys
import datetime

stations = ['AHUD',
            'BYL',
            'RIMD',
            'UWE']

stat = stations[int(sys.argv[1])]

tstart = obspy.UTCDateTime(datetime.datetime(2018,6,1))
tend   = obspy.UTCDateTime(datetime.datetime(2018,8,6))

#for stat in stations:
fpath = os.path.join(stat, '*.mseed')
# print(fpath)
dat = obspy.read(fpath)
print(dat)
stationxml = glob.glob(os.path.join(stat, 'resp','*'))[0]
net = dat[0].stats['network']
sta = dat[0].stats['station']
loc = dat[0].stats['location']

pre_filt = (0.05, 0.01, 49.0, 50.0)
inv  = obspy.read_inventory(stationxml, format='STATIONXML')

for dat_ in dat:
    chan = dat_.stats['channel']
    t0   = dat_.stats['starttime']
    str_in = f'{net.upper()}.{sta.upper()}.{loc.upper()}.{chan.upper()}'
    # print(str_in)
    resp = inv.get_response(str_in, datetime=t0)
    paz   = resp.get_paz()
    poles = paz.poles
    zeros = paz.zeros
    A0    = paz.normalization_factor*resp.instrument_sensitivity.value 
    pz    = {'poles': poles,
             'zeros': zeros,
             'gain' : A0,
             'sensitivity' : resp.instrument_sensitivity.value}
    
    dat_.simulate(paz_remove=pz,
                 pre_filt=pre_filt,
                 taper=0.0,
                 simulate_sensitivity=False,
                 remove_sensitivity=False)
    dat_.detrend('linear')


dat.merge(fill_value=0.0)
dat.trim(starttime=tstart, endtime=tend)
print(f'Write {stat}.h5')
with h5py.File(f'{stat}_01_50.h5','w') as f:
    for k in range(len(dat)):
        chan = dat[k].stats['channel']
        f.create_dataset(chan, data=dat[k].data)
    f.attrs['starttime']     = str(dat[0].stats['starttime'])
    f.attrs['endtime']       = str(dat[0].stats['endtime'])
    f.attrs['npts']          = str(dat[0].stats['npts'])
    f.attrs['sampling_rate'] = str(dat[0].stats['sampling_rate'])
