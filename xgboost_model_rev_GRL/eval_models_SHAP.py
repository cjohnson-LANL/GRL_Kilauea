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
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


import matplotlib.pyplot as plt
#matplotlib.use('Agg')
import os, copy
import numpy as np
import h5py
import time, datetime
import joblib
import itertools
import sklearn
import glob
import obspy
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

def sort_list(l, index=0):
    from operator import itemgetter
    # Sort a list of list by the element in index position 
    return sorted(l, key=itemgetter(index))

###########################################################################

# set params
stat    ='UWE'
label   = 'DISP'
version = 'v1_30s'
n_best  = 45

gps_label = 'UWEV'

####
wd        = os.getcwd()
model_dir = f'model_REV_GRL_{stat}'
opt_label = f'optimizer_{version}_{label}_v_REV_GRL'
opt_dir   = os.path.join(wd, model_dir, opt_label)
pkl_list  = glob.glob(os.path.join(opt_dir, '*.json'))

opt_results = []
for pkl_list_ in pkl_list:
    fin = pkl_list_.split('/')[-1]
    fin_split = fin.split('_')
    count = int(fin_split[1])
    itera = int(fin_split[2])
    score = float(fin_split[3].replace('.json',''))
    opt_results.append([fin, count, itera, score])

opt_results = sort_list(opt_results, index=3)

model_idx = 0
model = XGBRegressor()
model.load_model(os.path.join(opt_dir, opt_results[model_idx][0]))

x, x_val, y, y_val, y_fit_train, y_fit_test = joblib.load(os.path.join(opt_dir, opt_results[model_idx][0].replace('model', 'dat').replace('json', 'pkl')))

models_bs = []
for res in opt_results[:n_best]:
    model_tmp = XGBRegressor()
    model_tmp.load_model(os.path.join(opt_dir, res[0]))
    models_bs.append(copy.deepcopy(model_tmp))

# SHAP
    feat_file=f'data/UWE_features_v_REV_GRLclean.txt'
feature_list = []
with open(feat_file, 'r') as ffeat:
    for k, line in enumerate(ffeat):
        feature_list.append(line.strip())

# X100 = shap.utils.sample(x, 100) # 100 instances for use as the background distribution
explainer_xgb = shap.Explainer(model, feature_names=feature_list)

# explainer_xgb = shap.explainers.Tree(model, x_val, feature_names=feature_list)
shap_values_xgb = explainer_xgb.shap_values(x_val, y=y_val, tree_limit=model.best_iteration+1)

shap_values_xgb_bs = []
for models in models_bs:
    explainer_xgb_bs = shap.Explainer(models, feature_names=feature_list)
    shap_values_xgb_ = explainer_xgb_bs.shap_values(x_val, y=y_val, tree_limit=models.best_iteration+1)
    shap_values_xgb_bs.append(shap_values_xgb_)
shap_values_xgb_bs = np.array(shap_values_xgb_bs)
# shap.plots.violin(shap_values_xgb, features=x_val, feature_names=feature_list, max_display=10, plot_type="layered_violin")

shap_values_xgb = shap_values_xgb_bs.mean(axis=0)

plt.close('all')
shap.plots.violin(shap_values_xgb, features=x_val, feature_names=feature_list, max_display=105, 
    plot_type="violin", color_bar=True, plot_size=(10,17), show=False)
plt.savefig(os.path.join(wd, f'{stat}_all_SHAP_v3.png'), dpi=300, bbox_inches='tight')

plt.close('all')
shap.plots.violin(shap_values_xgb, features=x_val, feature_names=feature_list, max_display=25, 
    plot_type="violin", color_bar=True, plot_size=(10,17), show=False)
plt.savefig(os.path.join(wd, f'{stat}_25_shap_v2.png'), dpi=300, bbox_inches='tight')

plt.close('all')
shap.plots.violin(shap_values_xgb, features=x_val, feature_names=feature_list, max_display=15, 
    plot_type="violin", color_bar=True, plot_size=(10,12), show=False)
plt.savefig(os.path.join(wd, f'{stat}_15_shap_v2.png'), dpi=300, bbox_inches='tight')


######
feature_order = np.argsort(np.sum(np.abs(shap_values_xgb), axis=0))[::-1]
explainer_xgb2 = shap.Explainer(model, feature_names=feature_list)
shap_values2 = explainer_xgb2(x_val)

plt.close('all')
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12,12))
k = 0
feat_name = feature_list[feature_order[k]]
shap.plots.scatter(shap_values2[:,feat_name], ax=ax[0,0], show=False)
k = 1
feat_name = feature_list[feature_order[k]]
shap.plots.scatter(shap_values2[:,feat_name], ax=ax[0,1], show=False)
k = 2
feat_name = feature_list[feature_order[k]]
shap.plots.scatter(shap_values2[:,feat_name], ax=ax[1,0], show=False)
k = 3
feat_name = feature_list[feature_order[k]]
shap.plots.scatter(shap_values2[:,feat_name], ax=ax[1,1], show=False)
k = 4
feat_name = feature_list[feature_order[k]]
shap.plots.scatter(shap_values2[:,feat_name], ax=ax[2,0], show=False)
k = 5
feat_name = feature_list[feature_order[k]]
shap.plots.scatter(shap_values2[:,feat_name], ax=ax[2,1], show=False)
plt.savefig(os.path.join(wd, f'{stat}_SHAP_scatter.png'), dpi=300, bbox_inches='tight')

