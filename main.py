#!/usr/bin/env python

import argparse, os, sys
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import datetime
import matplotlib.pyplot as plt

gpu_list = tf.config.list_physical_devices('GPU')
# Calling GPUs by default with Keras will reserve the rest of the remaining memory
# To avoid this, allow memory growth to dynamically allocate memory over the program life
if gpu_list:
    try:
        for gpu in gpu_list:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print(f'TensorFlow Version: {tf.__version__}')
print(f'Num of GPUs: {len(tf.config.list_physical_devices("GPU"))}')

class FuzzyVAE(tf.keras.Model):

    def __init__(self, input_shape, latent_size, beta=1E-5):
        super().__init__()

        self.beta = beta
        self.encoder_input_shape = input_shape
        self.latent_size = latent_size
        self.encoder, self.latent_vector = self._build_encoder()
        self.decoder = self._build_decoder()


    def _build_encoder(self):

        encoder_layers = [
            tf.keras.layers.Input(shape=self.encoder_input_shape),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.latent_size + self.latent_size),
        ]
        return tf.keras.Sequential(encoder_layers), encoder_layers[-1]
    
    def _build_decoder(self):
        decoder_layers = [
            tf.keras.layers.Input(shape=(self.latent_size,)),
            tf.keras.layers.Dense(units=7*7*32, activation='relu'),
            tf.keras.layers.Reshape(target_shape=(7,7,32)),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'),
            tf.keras.layers.BatchNormalization(),
        ]
        return tf.keras.Sequential(decoder_layers)


    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_size))
        return self.decode(eps, apply_sigmoid=True)
    
    def encode(self, x, training=False):
        fuzz_x = x
        if training:
            fuzz_x += tf.random.normal(shape=x.shape, mean=0, stddev=self.beta)
        z = self.encoder(fuzz_x)
        mean, logvar = tf.split(z, num_or_size_splits=2, axis=1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps + tf.exp(logvar * .5) + mean
    
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum( -0.5 * ((sample - mean)**2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
    
    def call_detailed(self, x, training=False):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z, True)

        return x_logit, z, mean, logvar
    
    def call(self, x, training=False):
        mean, logvar = self.encode(x, training=False)
        z = self.reparameterize(mean, logvar)
        return self.decode(z, True)
    
    def compute_loss(self, x, training=False):
        x_logit, z, mean, logvar = self.call_detailed(x, training)

        #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        cross_entropy = tf.pow(x_logit - x, 2) # cross entropy blows up

        logpx_z = -tf.reduce_sum(cross_entropy)
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)

        loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)

        try:
            tf.debugging.assert_all_finite(loss, message='Loss NaN')
            
        except Exception as e:
            print(f'NaN Detected: {e}')
            print(tf.reduce_min(x_logit), tf.reduce_max(x_logit), x_logit.shape)
            raise Exception(f"Min: {tf.reduce_min(x_logit)}, Max: {tf.reduce_max(x_logit)}, Shape: {x_logit.shape}")

        return {
            'loss': loss,
            'max': tf.reduce_max(x),
            'min': tf.reduce_min(x),
            'r_max': tf.reduce_max(x_logit),
            'r_min': tf.reduce_min(x_logit),
            'mse': tf.reduce_mean(tf.math.pow(x_logit, 2)),
            'logvar': tf.reduce_sum(tf.exp(-logvar)),
        }
        

    def train_step(self, x):

        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, training=True)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return loss
    
    def test_step(self, x):
        return self.call(x, training=False)




def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--latent-dim', '-l', type=int, default=64)
    parser.add_argument('--beta', type=float, default=1E-5)
    parser.add_argument('--batch-size', '-b', type=int, default=32)
    parser.add_argument('--max-epochs', '-e', type=int, default=10)
    
    return parser.parse_args()

def load_data(args):

    train_ds, ds_info = tfds.load('emnist', split='train', shuffle_files=True, download=False, with_info=True)
    test_ds = tfds.load('emnist', split='test', shuffle_files=True, download=False, with_info=False)

    train_ds = train_ds.batch(args.batch_size)
    test_ds = test_ds.batch(args.batch_size)

    def normalize_img(element):
        return tf.cast(element['image'], tf.float32) / 255.
    
    train_ds = train_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

    print(f'Length of Training data: {len(train_ds)}')

    return {
        'train': train_ds,
        'test': test_ds,
        'info': ds_info,
    }


def build_model(args, data):
    input_shape = data['info'].features['image'].shape
    vae = FuzzyVAE(input_shape=input_shape, latent_size = args.latent_dim, beta=args.beta)

    vae.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=1E-4
    ))

    #vae.build((None, ) + vae.encoder_input_shape)
    #vae.summary()

    return vae


def train_model(args, model:tf.keras.Model, data):

    logdir = os.path.join('./logs', f'fit_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=logdir)
    ]

    model.fit(data['train'], batch_size=args.batch_size, callbacks=callbacks, shuffle=True, epochs=args.max_epochs)
    return model    



def evaluate(model:tf.keras.Model, data):
    
    #test_loss = model.compute_loss(data['test'])
    y = model.predict(data['test'])

    x_i = list(data['test'].take(1).as_numpy_iterator())[0][0,:,:,0]
    #x_i = data['test'][0, :, :, 0]
    y_i = y[0, :, :, 0]

    print(x_i.shape)
    print(y_i.shape)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10,8))

    ax0.imshow(x_i, cmap='gray')
    ax0.set_title('Original')

    ax1.imshow(y_i, cmap='gray')
    ax1.set_title('Prediction')

    plt.show()


def main():

    
    args = get_args()

    data = load_data(args)
    model = build_model(args, data)
    model = train_model(args, model, data)
    evaluate(model, data)


if __name__ == '__main__':
    main()
