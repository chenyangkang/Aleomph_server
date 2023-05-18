
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import arviz as az
import pymc as pm
import pickle
import seaborn as sns
from tqdm import tqdm
import time
from warnings import filterwarnings
import contextily as cx
import pickle
import sys
import os
filterwarnings('ignore')
pd.set_option('display.max_row', 100)


def parse_wintering_results(city_index, idata, df_ori, ess_, rhat_, bmfi_, train_roc_auc, test_roc_auc,
                          X_train_Cropland_std, X_train_Built_up_std):
    
    spring_departure_urban_effect = np.concatenate(idata.posterior['spring_departure_urban_effect'].values, axis=0)
    spring_departure_cropland_effect = np.concatenate(idata.posterior['spring_departure_Cropland_effect'].values, axis=0)
    fall_arrival_urban_effect = np.concatenate(idata.posterior['fall_arrival_urban_effect'].values, axis=0)
    fall_arrival_cropland_effect = np.concatenate(idata.posterior['fall_arrival_Cropland_effect'].values, axis=0)

    out_df = []
    for index, line in df_ori[['sp','year','Order','migration_type','year_index','sp_index']].drop_duplicates().iterrows():
        out_df.append({
            'city_index':city_index,
            'sp':line['sp'],
            'year':line['year'],
            'Order':line['Order'],
            'migration_type':line['migration_type'],
            'spring_departure_urban_effect_mean':spring_departure_urban_effect[...,line['year_index'],line['sp_index']].mean(axis=0)/X_train_Built_up_std,
            'spring_departure_urban_effect_std':spring_departure_urban_effect[...,line['year_index'],line['sp_index']].std(axis=0)/X_train_Built_up_std,
            'fall_arrival_urban_effect_mean':fall_arrival_urban_effect[...,line['year_index'],line['sp_index']].mean(axis=0)/X_train_Built_up_std,
            'fall_arrival_urban_effect_std':fall_arrival_urban_effect[...,line['year_index'],line['sp_index']].std(axis=0)/X_train_Built_up_std,
            'spring_departure_cropland_effect_mean':spring_departure_cropland_effect[...,line['year_index'],line['sp_index']].mean(axis=0)/X_train_Cropland_std,
            'spring_departure_cropland_effect_std':spring_departure_cropland_effect[...,line['year_index'],line['sp_index']].std(axis=0)/X_train_Cropland_std,
            'fall_arrival_cropland_effect_mean':fall_arrival_cropland_effect[...,line['year_index'],line['sp_index']].mean(axis=0)/X_train_Cropland_std,
            'fall_arrival_cropland_effect_std':fall_arrival_cropland_effect[...,line['year_index'],line['sp_index']].std(axis=0)/X_train_Cropland_std,

            'ess_spring_departure_urban_effect':ess_['spring_departure_urban_effect'].values[line['year_index'],line['sp_index']],
            'rhat_spring_departure_urban_effect':rhat_['spring_departure_urban_effect'].values[line['year_index'],line['sp_index']],

            'ess_spring_departure_cropland_effect':ess_['spring_departure_Cropland_effect'].values[line['year_index'],line['sp_index']],
            'rhat_spring_departure_cropland_effect':rhat_['spring_departure_Cropland_effect'].values[line['year_index'],line['sp_index']],

            'ess_fall_arrival_urban_effect':ess_['fall_arrival_urban_effect'].values[line['year_index'],line['sp_index']],
            'rhat_fall_arrival_urban_effect':rhat_['fall_arrival_urban_effect'].values[line['year_index'],line['sp_index']],

            'ess_fall_arrival_cropland_effect':ess_['fall_arrival_Cropland_effect'].values[line['year_index'],line['sp_index']],
            'rhat_fall_arrival_cropland_effect':rhat_['fall_arrival_Cropland_effect'].values[line['year_index'],line['sp_index']],

            'mean_bmfi':np.mean(bmfi_),
            
            'train_roc_auc':train_roc_auc,
            'test_roc_auc':test_roc_auc

        })



    out_df = pd.DataFrame(out_df)
    return out_df


    
    


def parse_breeding_results(city_index, idata, df_ori, ess_, rhat_, bmfi_, train_roc_auc, test_roc_auc,
                          X_train_Cropland_std, X_train_Built_up_std):
    
    spring_arrival_urban_effect = np.concatenate(idata.posterior['spring_arrival_urban_effect'].values, axis=0)
    spring_arrival_cropland_effect = np.concatenate(idata.posterior['spring_arrival_Cropland_effect'].values, axis=0)
    fall_departure_urban_effect = np.concatenate(idata.posterior['fall_departure_urban_effect'].values, axis=0)
    fall_departure_cropland_effect = np.concatenate(idata.posterior['fall_departure_Cropland_effect'].values, axis=0)

    out_df = []
    for index, line in df_ori[['sp','year','Order','migration_type','year_index','sp_index']].drop_duplicates().iterrows():
        out_df.append({
            'city_index':city_index,
            'sp':line['sp'],
            'year':line['year'],
            'Order':line['Order'],
            'migration_type':line['migration_type'],
            'spring_arrival_urban_effect_mean':spring_arrival_urban_effect[...,line['year_index'],line['sp_index']].mean(axis=0)/X_train_Built_up_std,
            'spring_arrival_urban_effect_std':spring_arrival_urban_effect[...,line['year_index'],line['sp_index']].std(axis=0)/X_train_Built_up_std,
            'fall_departure_urban_effect_mean':fall_departure_urban_effect[...,line['year_index'],line['sp_index']].mean(axis=0)/X_train_Built_up_std,
            'fall_departure_urban_effect_std':fall_departure_urban_effect[...,line['year_index'],line['sp_index']].std(axis=0)/X_train_Built_up_std,
            'spring_arrival_cropland_effect_mean':spring_arrival_cropland_effect[...,line['year_index'],line['sp_index']].mean(axis=0)/X_train_Cropland_std,
            'spring_arrival_cropland_effect_std':spring_arrival_cropland_effect[...,line['year_index'],line['sp_index']].std(axis=0)/X_train_Cropland_std,
            'fall_departure_cropland_effect_mean':fall_departure_cropland_effect[...,line['year_index'],line['sp_index']].mean(axis=0)/X_train_Cropland_std,
            'fall_departure_cropland_effect_std':fall_departure_cropland_effect[...,line['year_index'],line['sp_index']].std(axis=0)/X_train_Cropland_std,

            'ess_spring_arrival_urban_effect':ess_['spring_arrival_urban_effect'].values[line['year_index'],line['sp_index']],
            'rhat_spring_arrival_urban_effect':rhat_['spring_arrival_urban_effect'].values[line['year_index'],line['sp_index']],

            'ess_spring_arrival_cropland_effect':ess_['spring_arrival_Cropland_effect'].values[line['year_index'],line['sp_index']],
            'rhat_spring_arrival_cropland_effect':rhat_['spring_arrival_Cropland_effect'].values[line['year_index'],line['sp_index']],

            'ess_fall_departure_urban_effect':ess_['fall_departure_urban_effect'].values[line['year_index'],line['sp_index']],
            'rhat_fall_departure_urban_effect':rhat_['fall_departure_urban_effect'].values[line['year_index'],line['sp_index']],

            'ess_fall_departure_cropland_effect':ess_['fall_departure_Cropland_effect'].values[line['year_index'],line['sp_index']],
            'rhat_fall_departure_cropland_effect':rhat_['fall_departure_Cropland_effect'].values[line['year_index'],line['sp_index']],

            'mean_bmfi':np.mean(bmfi_),
            
            'train_roc_auc':train_roc_auc,
            'test_roc_auc':test_roc_auc

        })



    out_df = pd.DataFrame(out_df)
    return out_df


    