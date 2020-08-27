
import numpy as np
import os
import pandas as pd
import csv

NEW_DATA_DIR = 'train_data'

numeric_columns = ['Longitude', 'Latitude']

category_columns = ['Severity', 'Number_of_Vehicles',
                    'Number_of_Casualties', 'Date', 'Day_of_Week',
                    'Time', 'Speed_limit', 'Light_Conditions',
                    'Road_Surface_Conditions', 'Urban_or_Rural_Area',
                    'Did_Police_Officer_Attend_Scene_of_Accident']

cols_to_load = numeric_columns + category_columns[1:] + ['Accident_Severity']

metrics = ['accuracy', 'precision', 'recall', 'auc']


def create_logfile(log_filename):
    if not os.path.exists(log_filename):
        with open(log_filename, 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(['model_name'] + metrics)


def load_data():

    train_filename = f'{NEW_DATA_DIR}/train.pkl'
    val_filename = f'{NEW_DATA_DIR}/val.pkl'
    test_filename = f'{NEW_DATA_DIR}/test.pkl'

    train_data = pd.read_pickle(train_filename)
    val_data = pd.read_pickle(val_filename)
    test_data = pd.read_pickle(test_filename)

    y_train = train_data['Severity'].copy()
    X_train = train_data.drop(['Severity'], axis=1).copy()

    y_val = val_data['Severity'].copy()
    X_val = val_data.drop(['Severity'], axis=1).copy()

    y_test = test_data['Severity'].copy()
    X_test = test_data.drop(['Severity'], axis=1).copy()

    del train_data, val_data, test_data

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Source for func.:  https://www.kaggle.com/kyakovlev/ashrae-data-minification
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df
