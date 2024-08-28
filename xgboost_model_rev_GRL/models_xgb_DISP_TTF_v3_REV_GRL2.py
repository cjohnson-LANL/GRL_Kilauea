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

import matplotlib.pyplot as plt
#matplotlib.use('Agg')
import os, copy
import numpy as np
import h5py
import time, datetime
import joblib
import itertools
import sklearn
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import sklearn.preprocessing
from sklearn.model_selection import KFold

try:
    import shap
    import shap.plots
except(ModuleNotFoundError):
    print('SHAP not loaded. Install for this env')

gpu_id           = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

# stats = ['AHUD', 'BYL', 'RIMD', 'UWE']
stats = ['RIMD', 'UWE']
label = ['DISP', 'TTF']
N     = 250

for label_ in label:
    for stat in stats:

        version     = f'v1_30s_{label_}'
        n_splits    = 3
        wd          = ''
        model_dir   = f'model_REV_GRL_{stat}'
        opt_pkl     = 'optimizer_{0}.pkl'.format(version)
        opt_pkl_dir = os.path.join(wd, model_dir, opt_pkl.replace('.pkl','_v_REV_GRL'))

        if not os.path.isdir(opt_pkl_dir):
            os.makedirs(opt_pkl_dir)

        # Load training data
        hdf5_file=f'data/{stat}_30s_data_features_v_REV_GRL.h5'
        feat_file=f'data/{stat}_features_v_REV_GRL.txt'
        with h5py.File(hdf5_file, 'r') as data:
            x_train  = data['X_train'][:]
            x_test   = data['X_test'][:]

            if label_ == label[0]:
                y_train = data['disp_train'][:]
                y_test  = data['disp_test'][:]
            
            elif label_ == label[1]:
                y_train = data['ttf_train'][:]
                y_test  = data['ttf_test'][:]

        # Fit model
        for k in range(N):
            print(f'\nModel iteration {k} :: Stat {stat} :: Label {label_}')

            kfold    = KFold(n_splits=n_splits, shuffle=True)
            idx_train, idx_eval = next(kfold.split(y_train))

            x = x_train[idx_train]
            y = y_train[idx_train]

            x_val = x_train[idx_eval]
            y_val = y_train[idx_eval]

            # Eval set for early stopping
            eval_set = []
            for j, X_val_ in enumerate(x_val):
                eval_set.append((np.expand_dims(X_val_, 0), y_val[j]))

            params = {}
            #params['objective']            = 'reg:squarederror'
            params['objective']             = 'reg:absoluteerror'
            params['eval_metric']           = 'mae'
            params['early_stopping_rounds'] = 5
            params['booster']               = 'gbtree'
            #params['tree_method']          = 'hist'
            params['tree_method']           = 'gpu_hist'
            #params['sampling_method']      = 'gradient_based'
            params['sampling_method']       = 'uniform'
            params['max_depth']             = 4
            params['n_estimators']          = 100
            params['learning_rate']         = 0.01
            params['gamma']                 = 2.0
            params['subsample']             = 0.333
            params['colsample_bytree']      = 0.333
            #params['reg_lambda']           = 1.0
            #params['reg_alpha']            = 0.0
            params['random_state']          = np.random.randint(0, 1000000)

            # XGB Model fir with early stopping
            model = XGBRegressor()
            model.set_params(**params)
            model.fit(x, y, eval_set=eval_set, verbose=False)

            print(f'best_iteration :: {model.best_iteration:02d}')
            print(f'best_score     :: {model.best_score:6.4f}')
            print('Save')
            y_fit_train = model.predict(x_train, iteration_range=(0, model.best_iteration+1))
            y_fit_test  = model.predict(x_test,  iteration_range=(0, model.best_iteration+1))

            scores = f'{k:03d}_{model.best_iteration:02d}_{model.best_score:6.4f}'
            model_str = os.path.join(opt_pkl_dir, f'model_{scores}.json')
            model.save_model(model_str)

            dat = [x, x_val, y, y_val, y_fit_train, y_fit_test ]
            joblib.dump(dat, os.path.join(opt_pkl_dir, f'dat_{scores}.pkl'))

