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

import os, copy
import numpy as np
import h5py
import time, datetime
import joblib
import itertools
import glob


def sort_list(l, index=0):
    from operator import itemgetter
    # Sort a list of list by the element in index position 
    return sorted(l, key=itemgetter(index))

n_best = 45

# set params
stats    = ['UWE', 'RIMD', 'BYL', 'AHUD']
labels   = ['DISP', 'TTF']
version  = 'v1_30s'

for stat in stats:
    for label in labels:
        wd        = os.getcwd()
        model_dir = f'model_REV_GRL_{stat}'
        opt_label = f'optimizer_{version}_{label}_v_REV_GRL'
        opt_dir   = os.path.join(wd, model_dir, opt_label)
        pkl_list  = glob.glob(os.path.join(opt_dir, '*.pkl'))

        opt_results = []
        for pkl_list_ in pkl_list:
            fin = pkl_list_.split('/')[-1]
            fin_split = fin.split('_')
            count = int(fin_split[1])
            itera = int(fin_split[2])
            score = float(fin_split[3].replace('.pkl',''))
            opt_results.append([fin, count, itera, score])

        opt_results = sort_list(opt_results, index=3)
        x, x_val, y, y_val, y_fit_train, y_fit_test = joblib.load(os.path.join(opt_dir, opt_results[0][0]))
        y_fit_test_avg = np.zeros((n_best, y_fit_test.size))
        for k, opt_results_ in enumerate(opt_results[:n_best]):
            _, _, _, _, _, y_fit_test_avg[k] = joblib.load(os.path.join(opt_dir, opt_results_[0]))

        y_fit_test_mean = y_fit_test_avg.mean(axis=0)
        y_fit_test_std  = y_fit_test_avg.std(axis=0)
        y_fit_test_max  = y_fit_test_mean + y_fit_test_std
        y_fit_test_min  = y_fit_test_mean - y_fit_test_std


        # Load data
        hdf5_file=f'data/{stat}_30s_data_features_v_REV_GRL.h5'
        with h5py.File(hdf5_file, 'r') as data:
            x_train = data['X_train'][:]
            x_test  = data['X_test'][:]
            t_train = data['t_train'][:]
            t_test  = data['t_test'][:]
            y_train = data['disp_train'][:]
            y_test  = data['disp_test'][:]

        var_out = np.array([t_test, y_test, y_fit_test_mean, y_fit_test_std])
        np.save(stat+'_'+label, var_out)
