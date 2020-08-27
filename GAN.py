
import csv
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import *
from keras.models import Model
from util import load_data, create_dir
from vanilla_ann import DataGenerator

################################################################################################


class CreateGANModel():

    def __init__(self, num_features, latent_dim, num_outputs):
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.num_outputs = num_outputs
        self.gen_nn = self._get_generator()
        self.dis_nn = self._get_discriminator()

    def _dense_block(self, num_neurons, in_layer):
        dense = Dense(num_neurons, activation='linear')(in_layer)
        dense = LeakyReLU()(dense)
        dense = Dropout(0.5)(dense)
        return dense

    def _get_discriminator(self):
        feature_in_layer = Input(shape=(self.num_features,))
        feature_in_dense = self._dense_block(128, feature_in_layer)

        severity_in_layer = Input(shape=(self.num_outputs,))
        severity_in_dense = self._dense_block(128, severity_in_layer)

        combined_in = concatenate([feature_in_dense, severity_in_dense])

        dense = self._dense_block(128, combined_in)
        dense = self._dense_block(128, dense)
        dense = self._dense_block(128, dense)
        dense = self._dense_block(128, dense)

        out_layer = self._dense_block(1, dense)

        model = Model(inputs=[feature_in_layer, severity_in_layer], outputs=out_layer, name='discriminator')

        return model

    def _get_generator(self):
        latent_in_layer = Input(shape=(self.latent_dim,))
        latent_in_dense = self._dense_block(128, latent_in_layer)

        severity_in_layer = Input(shape=(self.num_outputs,))
        severity_in_dense = self._dense_block(128, severity_in_layer)

        combined_in = concatenate([latent_in_dense, severity_in_dense])

        dense = self._dense_block(128, combined_in)
        dense = self._dense_block(128, dense)
        dense = self._dense_block(128, dense)
        dense = self._dense_block(128, dense)

        out_layer = self._dense_block(self.num_features, dense)

        model = Model(inputs=[latent_in_layer, severity_in_layer], outputs=out_layer, name='Generator')

        return model


class GAN():
    def __init__(self, discriminator, generator, model_name, batch_size, latent_dim):
        self.discriminator = discriminator
        self.generator = generator
        self.model_name = model_name
        self.log_fname = f'log/{model_name}/log.csv'
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

        self.d_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
        self.g_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)

    #######################################################################################################
    @tf.function
    def train_step(self, real_features, real_labels):

        # Sample random points in the latent space
        batch_size = tf.shape(real_features)[0]
        # half_batch = tf.math.floordiv(batch_size, 2)
        # real_features = real_features[:half_batch]
        # real_labels = real_labels[:half_batch]

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_features = self.generator([random_latent_vectors, real_labels])

        # Train the discriminator
        with tf.GradientTape(persistent=True) as tape:

            predictions_fake = self.discriminator([generated_features, real_labels])
            predictions_real = self.discriminator([real_features, real_labels])

            # One-sided label smoothing
            labels_real = tf.zeros_like(predictions_real)
            labels_fake = tf.ones_like(predictions_fake)
            labels_fake += 0.05 * tf.random.normal(tf.shape(labels_fake))

            fake_loss = self.loss_fn(labels_fake, predictions_fake)
            real_loss = self.loss_fn(labels_real, predictions_real)
            d_loss = fake_loss + real_loss

        grads_real = tape.gradient(real_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads_real, self.discriminator.trainable_weights)
        )

        grads_fake = tape.gradient(fake_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads_fake, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:

            generated_features = self.generator([random_latent_vectors, real_labels])
            predictions = self.discriminator([generated_features, real_labels])

            # Assemble labels that say "all real images"
            misleading_labels = tf.zeros_like(predictions)

            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return d_loss, g_loss

    def write_log(self, data, first=False):
        if first:
            mode = 'w+'
        else:
            mode = 'a'

        with open(self.log_fname, mode) as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def fit(self, train_gen, num_epochs, initial_epoch, warm_start=False):

        if warm_start:
            #             self.write_log(['epoch', 'step', 'd_loss_fake', 'd_loss_real', 'g_loss'], first=True)
            self.write_log(['epoch', 'd_loss', 'g_loss'], first=True)

        d_loss, g_loss = None, None

        dataset_length = train_gen.__len__()

        for epoch in tqdm(range(initial_epoch, num_epochs, 1), total=num_epochs):
            #         for epoch in tf.range(initial_epoch+1, num_epochs, 1):

            print(f'Epoch {epoch} of {num_epochs}')

            # for i in tqdm(range(dataset_length)):

            for i in range(dataset_length):

                real_features, real_labels = train_gen.__getitem__(i)
                real_features = tf.convert_to_tensor(real_features, dtype=tf.float32)
                real_labels = tf.convert_to_tensor(real_labels, dtype=tf.float32)
                d_loss, g_loss = self.train_step(real_features, real_labels)

            print(f'Dis loss {d_loss.numpy()}, Gen loss {g_loss.numpy()}')

            self.write_log([epoch, d_loss.numpy(), g_loss.numpy()])

            train_gen.on_epoch_end()

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self.generator.save(f'log/{self.model_name}/gen_{str(epoch).zfill(2)}.h5')
                # self.discriminator.save(f'log/{self.model_name}/dis_{epoch}.h5')


if __name__ == '__main__':

    (X_train, y_train), _, _ = load_data()

    num_features = len(X_train.columns)
    latent_dim = 64
    batch_size = 128
    model_name = 'GAN'
    num_epochs = 100
    initial_epoch = 0
    num_outputs = len(np.unique(y_train.astype(int)))

    create_dir('log')
    create_dir(f'log/{model_name}')

    # Create models
    gan_obj = CreateGANModel(num_features, latent_dim, num_outputs)
    gan = GAN(gan_obj.dis_nn, gan_obj.gen_nn, model_name, batch_size, latent_dim)

    print('generator', gan_obj.gen_nn.count_params())
    print('discrim', gan_obj.dis_nn.count_params())

    # Parameters
    params = {'num_features': num_features,
              'batch_size': batch_size,
              'shuffle': True,
              'num_outputs': num_outputs}

    training_generator = DataGenerator(X_train, y_train, **params)

    # Train models
    gan.fit(
        training_generator,
        num_epochs,
        initial_epoch,
        False
    )
