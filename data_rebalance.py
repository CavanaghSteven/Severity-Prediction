
import csv
from vanilla_ann import Prediction_Model, DataGenerator, train_model
import numpy as np
from util import load_data, create_dir, create_logfile
import keras
from imblearn.keras import BalancedBatchGenerator
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from glob import glob
from GAN import CreateGANModel
import os


def train(
        train_gen,
        X_test_func, y_test_func,
        X_val_func, y_val_func,
        model_name,
        batch_size,
        log_filename
        ):

    num_features = len(X_test_func.columns)
    layers = [128, 128, 128, 128, 128, 128]
    num_outputs = len(np.unique(y_test_func.astype(int)))
    num_epochs = 30

    model = Prediction_Model((num_features, layers, num_outputs, model_name))

    # Parameters
    params = {'num_features': num_features,
              'batch_size': batch_size,
              'shuffle': True,
              'num_outputs': num_outputs}

    validation_generator = DataGenerator(X_test_func, y_test_func, **params)
    test_generator = DataGenerator(X_val_func, y_val_func, **params)

    train_model(
        model,
        train_gen,
        validation_generator,
        model_name,
        num_epochs=num_epochs
    )

    model.load_weights(f'log/{model_name}/weights.hdf5')

    create_logfile(log_fname)

    scores = model.evaluate_generator(test_generator)

    with open(log_fname, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([model_name] + scores)

    # print('test loss', model_name, scores)


if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test), (X_val, y_val) = load_data()
    y_train_onehot = keras.utils.to_categorical(y_train, len(np.unique(y_train.values.astype('float32'))))
    batchsize = 128

    create_dir('log')
    log_fname = 'log/balanced_log.csv'

    if not os.path.exists(log_fname):
        with open(log_fname, 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(['model_name', 'loss'])

    # upsample minority classes  ######################################
    model_name = 'upsample_random'
    create_dir(f'log/{model_name}')
    sampler = RandomOverSampler(random_state=42)
    training_generator = BalancedBatchGenerator(
        X_train, y_train_onehot, sampler=sampler, batch_size=batchsize, random_state=42)
    train(training_generator, X_test, y_test, X_val, y_val, model_name, batchsize, log_fname)

    # downsample majority classes #####################################
    model_name = 'downsample-random'
    create_dir(f'log/{model_name}')
    sampler = RandomUnderSampler(random_state=42)
    training_generator = BalancedBatchGenerator(
        X_train, y_train_onehot, sampler=sampler, batch_size=batchsize, random_state=42)
    train(training_generator, X_test, y_test, X_val, y_val, model_name, batchsize, log_fname)

    # Create Synthetic Samples  #######################################

    # SMOTE ###########################################################
    model_name = 'upsample-smote'
    create_dir(f'log/{model_name}')
    X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train_onehot)
    training_generator = BalancedBatchGenerator(
        X_train_res, y_train_res, sampler=None, batch_size=batchsize, random_state=42)
    train(training_generator, X_test, y_test, X_val, y_val, model_name, batchsize, log_fname)

    del X_train_res, y_train_res

    # GAN ###########################################################
    val_counts = y_train.value_counts().sort_index()

    gen_filename = sorted(glob(f'log/GAN/gen_*.h5'))[-1]

    num_features = len(X_train.columns)
    latent_dim = 64
    num_outputs = len(np.unique(y_train))

    # Create models
    gan_obj = CreateGANModel(num_features, latent_dim, num_outputs)

    model = gan_obj.gen_nn
    model.load_weights(gen_filename)
    X_train = X_train.values

    max_value = val_counts[val_counts.idxmax()]
    for Severity, val in enumerate(val_counts):
        difference = max_value - val
        print(Severity, val, difference)
        if not difference > 0:
            continue

        if difference > 500000:
            difference = 500000

        latent_vector = np.random.normal(size=(difference, latent_dim))
        severity_vector = np.zeros(shape=(difference, num_outputs))
        severity_vector[:, Severity] = np.ones(shape=(difference))

        new_data_X = model.predict([latent_vector, severity_vector])
        new_data_y = severity_vector

        X_train = np.concatenate([X_train, new_data_X], axis=0)
        y_train_onehot = np.concatenate([y_train_onehot, new_data_y], axis=0)

    del new_data_X, new_data_y, latent_vector, severity_vector

    model_name = 'upsample-GAN'
    create_dir(f'log/{model_name}')
    training_generator = BalancedBatchGenerator(
        X_train, y_train_onehot, sampler=None, batch_size=batchsize, random_state=42)
    train(training_generator, X_test, y_test, X_val, y_val, model_name, batchsize, log_fname)
