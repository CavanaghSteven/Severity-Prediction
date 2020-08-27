
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from util import create_dir, NEW_DATA_DIR, reduce_mem_usage
from util import numeric_columns, category_columns, cols_to_load
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
from glob import glob


def load_data(filename, cols_to_keep):
    data = pd.read_csv(filename)
    cols = [col for col in data.columns if col not in cols_to_keep]
    data.drop(cols, axis=1, inplace=True)
    data.rename(columns={'Accident_Severity': 'Severity'}, inplace=True)
    return data


def _split_date(date_string):  # Format dd/mm/yyyy
    tmp = date_string.split('/')
    return tmp[1], tmp[2]  # Month, Year


def _impute_numerical(df, column):
    # df[column] = df[column].astype(float64)
    df[column] = df[column].fillna(df[column].mean())
    return df


def _impute_categorical(df, column):
    # df[column] = df[column].astype(str)
    df[column] = df[column].fillna(df[column].value_counts().index[0])
    return df


def _time_string_to_num(time_string):
    # Input of type 17:42
    # output in minutes like (17 * 60) + 42

    time_string = str(time_string)
    tmp = time_string.split(':')
    result = int(tmp[0]) * 60
    result += int(tmp[1])
    return result


def sub_sample_df(df, AMT_TO_SAMPLE):
    return df.sample(n=AMT_TO_SAMPLE, axis=0, random_state=42).reset_index(drop=True)


def create_dummies(df, columns):
    cat_df = df[columns]
    cat_df = pd.get_dummies(cat_df, sparse=True)
    df.drop(columns, axis=1, inplace=True)
    df = df.merge(cat_df, left_index=True, sort=False, right_index=True)
    return df.reset_index(drop=True)


def do_concat(df, df2):
    return pd.concat([df, df2], sort=False, axis=0, ignore_index=True)


def impute_df(data):

    for col in numeric_columns:
        data = _impute_numerical(data, col)

    for col in category_columns:
        data = _impute_categorical(data, col)

    data['Time'] = data['Time'].map(_time_string_to_num).astype(int)
    data['Month'], data['Year'] = zip(*data['Date'].map(_split_date))
    data.drop(['Date'], axis=1, inplace=True)

    data.rename(columns={'Road_Surface_Conditions': 'RSC'}, inplace=True)
    data.rename(columns={'Light_Conditions': 'LC'}, inplace=True)
    data.rename(columns={'Did_Police_Officer_Attend_Scene_of_Accident': 'Officer'}, inplace=True)

    one_hot_columns = ['LC', 'RSC', 'Officer']

    data = create_dummies(data, one_hot_columns)

    return data


def load_dataset(filename, AMT_TO_SAMPLE):

    data = load_data(filename, cols_to_load)

    # data = impute_df(data)

    if AMT_TO_SAMPLE is not None:
        data = sub_sample_df(data, AMT_TO_SAMPLE)

    return data


def transform_data(df, scaler):
    labels = df['Severity'].copy()
    df = df.drop(['Severity'], axis=1).copy()
    columns = df.columns
    df = pd.DataFrame(scaler.transform(df), columns=columns)
    df['Severity'] = labels - 1  # Transforms labels from [1, 2, 3] to [0, 1, 2]
    return df


def train_scaler(df, scaler):
    train_df = df.drop(['Severity'], axis=1).copy()
    scaler.fit(train_df)
    return scaler


def save_dataset(df):
    # Split data into train, val, and test
    temp, test = train_test_split(df, stratify=df['Severity'], test_size=0.1, random_state=42)
    train, val = train_test_split(temp, stratify=temp['Severity'], test_size=0.1, random_state=42)
    del temp

    # Impute missing values
    train = impute_df(train)
    val = impute_df(val)
    test = impute_df(test)

    # Scaling variables to known range
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = train_scaler(train, scaler)

    train = transform_data(train, scaler)
    val = transform_data(val, scaler)
    test = transform_data(test, scaler)

    df = reduce_mem_usage(df, verbose=False)
    train = reduce_mem_usage(train, verbose=False)
    val = reduce_mem_usage(val, verbose=False)
    test = reduce_mem_usage(test, verbose=False)

    df.to_pickle(f'{NEW_DATA_DIR}/full_df.pkl')

    train.to_pickle(f'{NEW_DATA_DIR}/train.pkl')
    val.to_pickle(f'{NEW_DATA_DIR}/val.pkl')
    test.to_pickle(f'{NEW_DATA_DIR}/test.pkl')

    # train.to_csv(f'{NEW_DATA_DIR}/train.csv')
    # val.to_csv(f'{NEW_DATA_DIR}/val.csv')
    # test.to_csv(f'{NEW_DATA_DIR}/test.csv')

    del train, test, val


if __name__ == "__main__":

    create_dir(NEW_DATA_DIR)

    first_file = True  # Multiple files
    data = None
    data_filename = ''
    num_sample = None

    filenames = glob(f'data/accidents_*.csv')
    print('Number of files found', len(filenames))

    # multiple files, so they need to be handled separately
    for i, f in enumerate(filenames):
        print(f, i)

        if first_file:
            data = load_dataset(f, num_sample)
            first_UK_file = False
        else:
            new_data = load_dataset(f, num_sample)
            data = do_concat(data, new_data)
            data = reduce_mem_usage(data, verbose=False)

    if data is not None:
        save_dataset(data)

