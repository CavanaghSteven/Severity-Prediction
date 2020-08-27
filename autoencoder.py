
import keras
from keras.layers import Dense, Input
from keras.models import Model
import numpy as np
from util import load_data, create_dir


def DR_Model(args):
    num_features, layers, encoding_dim = args

    # this is our input placeholder
    input_dim = Input(shape=(num_features,))
    encoded = Dense(layers[0], activation='relu')(input_dim)
    encoded = Dense(layers[1], activation='relu')(encoded)
    encoded = Dense(layers[2], activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    decoded = Dense(layers[2], activation='relu')(encoded)
    decoded = Dense(layers[1], activation='relu')(decoded)
    decoded = Dense(layers[0], activation='relu')(decoded)
    decoded = Dense(num_features, activation='sigmoid')(decoded)

    autoencoder = Model(input_dim, decoded, name='autoencoder')

    encoder = Model(input_dim, encoded, name='encoder')

    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder


class DataGenerator(keras.utils.Sequence):
    def __init__(self, X, num_features, batch_size=32, shuffle=True):
        'Initialization'
        self.num_features = num_features
        self.batch_size = batch_size
        self.X = X
        self.indexes = np.arange(len(self.X))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        X_batch = self.X.loc[indexes, :]

        X_batch = np.reshape(X_batch.values, (-1, self.num_features))

        return X_batch, X_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def train_model(model, m_name, num_epochs):

    tensorboard = keras.callbacks.TensorBoard(
        # log_dir=f'log/{m_name}/',
        log_dir=f'log\\{m_name}\\',

        histogram_freq=0,
        write_graph=True,
        write_images=True)

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss",
                                               mode="min",
                                               patience=11,
                                               restore_best_weights=True)

    # filepath = f'log/{m_name}/weights.hdf5'
    filepath = f'log\\{m_name}\\weights.hdf5'
    checkpoint = keras.callbacks.ModelCheckpoint(
        '{}'.format(filepath),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        save_weights_only=True)

    csvlog = keras.callbacks.CSVLogger(
        f'log\\{m_name}\\train_history.csv',
        separator=",",
        append=False)

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=6,
                        epochs=num_epochs,
                        callbacks=[tensorboard,
                                   checkpoint,
                                   early_stop,
                                   csvlog
                                   ])


if __name__ == '__main__':

    (X_train, _), _, (X_val, _) = load_data()

    num_features = len(X_train.columns)
    layers = [64, 64, 64]
    latentDim = 6
    model_name = 'autoencoder'

    create_dir('log')
    create_dir(f'log/{model_name}')

    autoencoder, _, = DR_Model((num_features, layers, latentDim))

    # Parameters
    params = {'num_features': num_features,
              'batch_size': 128,
              'shuffle': True}

    training_generator = DataGenerator(X_train, **params)
    validation_generator = DataGenerator(X_val, **params)

    train_model(autoencoder,
                model_name,
                num_epochs=30)
