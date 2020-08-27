
import csv
import keras
from keras.layers import Dense, Input
from keras.models import Model
import numpy as np
from util import load_data, create_dir
from util import create_logfile

def Prediction_Model(args):
    num_features, layers, num_outputs, model_name = args

    input_dim = Input(shape=(num_features,))
    x = input_dim

    for i in range(len(layers)):
        x = Dense(layers[i], activation='relu')(x)

    output = Dense(num_outputs, activation='sigmoid')(x)

    model = Model(input_dim, output, name=model_name)
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


class DataGenerator(keras.utils.Sequence):
    def __init__(self, X, y, num_features, num_outputs, batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.num_features = num_features
        self.indexes = np.arange(len(self.X))
        self.shuffle = shuffle
        self.num_outputs = num_outputs
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        X_batch = self.X.loc[indexes, :]
        y_batch = self.y.iloc[indexes]

        X_batch = np.reshape(X_batch.values, (-1, self.num_features))
        y_batch = np.reshape(y_batch.values, (-1, 1)).astype('int32')
        y_batch = keras.backend.one_hot(y_batch, self.num_outputs)
        y_batch = np.reshape(y_batch, (-1, self.num_outputs))
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def train_model(model, train_gen, val_gen, m_name, num_epochs):
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

    checkpoint = keras.callbacks.ModelCheckpoint(
        # f'log/{m_name}/weights.hdf5',
        f'log\\{m_name}\\weights.hdf5',

        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        save_weights_only=True)

    csvlog = keras.callbacks.CSVLogger(
        f'log\\{m_name}\\train_history.csv',
        separator=",",
        append=False)

    model.fit_generator(generator=train_gen,
                        validation_data=val_gen,
                        use_multiprocessing=True,
                        workers=8,
                        epochs=num_epochs,
                        callbacks=[
                            tensorboard,
                            checkpoint,
                            early_stop,
                            csvlog
                        ]
                        )


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = load_data()

    num_features = len(X_train.columns)
    layers = [64, 64, 64]
    num_outputs = len(np.unique(y_train.astype(int)))
    model_name = 'ann-vanilla'

    create_dir('log')
    create_dir(f'log/{model_name}')

    model = Prediction_Model((num_features, layers, num_outputs, model_name))

    # Parameters
    params = {'num_features': num_features,
              'batch_size': 128,
              'shuffle': True,
              'num_outputs': num_outputs}

    training_generator = DataGenerator(X_train, y_train, **params)
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
