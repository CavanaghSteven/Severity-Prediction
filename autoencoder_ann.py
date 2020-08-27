
import numpy as np
from util import load_data, create_dir
from util import create_logfile
import csv
from vanilla_ann import DataGenerator, Prediction_Model, train_model
from autoencoder import DR_Model
import keras
from keras.layers import Input
from keras.models import Model
from imblearn.keras import BalancedBatchGenerator
from imblearn.under_sampling import RandomUnderSampler


def auto_ann_model(args):

    num_features, layers, latentDim, num_outputs, model_name = args

    autoencoder, encoder = DR_Model((num_features, layers, latentDim))
    autoencoder.load_weights(f'log/autoencoder/weights.hdf5')
    encoder.trainable = False
    prediction_model = Prediction_Model((latentDim, layers, num_outputs, model_name))

    input_dim = Input(shape=(num_features,))
    encoder_out = encoder(input_dim)
    prediction_out = prediction_model(encoder_out)
    model = Model(input_dim, prediction_out, name=model_name)
    model.compile(optimizer='adam',
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[
                      keras.metrics.CategoricalAccuracy(),
                      keras.metrics.Precision(),
                      keras.metrics.Recall(),
                      keras.metrics.AUC()
                  ]
                  )

    return model


if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test), (X_val, y_val) = load_data()
    y_train_onehot = keras.utils.to_categorical(y_train, len(np.unique(y_train.values.astype('float32'))))

    num_features = len(X_train.columns)
    latentDim = 6
    layers = [64, 64, 64]
    model_name = 'ann-autoencoder'
    num_outputs = len(np.unique(y_train.astype(int)))
    batch_size = 128

    create_dir('log')
    create_dir(f'log/{model_name}')

    model = auto_ann_model((num_features, layers, latentDim, num_outputs, model_name))

    # Parameters
    params = {'num_features': num_features,
              'batch_size': batch_size,
              'shuffle': True,
              'num_outputs': num_outputs}

    # training_generator = DataGenerator(X_train, y_train, **params)
    sampler = RandomUnderSampler(random_state=42)
    training_generator = BalancedBatchGenerator(
        X_train, y_train_onehot, sampler=sampler, batch_size=batch_size, random_state=42)
    validation_generator = DataGenerator(X_val, y_val, **params)
    test_generator = DataGenerator(X_test, y_test, **params)

    train_model(
        model,
        training_generator,
        validation_generator,
        model_name,
        num_epochs=30
    )

    model.load_weights(f'log/{model_name}/weights.hdf5')

    log_fname = f'log/repres_losses.csv'

    create_logfile(log_fname)

    scores = model.evaluate_generator(test_generator)

    with open(log_fname, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([model_name] + scores)

    # print('test loss', model_name, scores)

