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


class BetaAnnealingCallback(tf.keras.callbacks.Callback):

    def __init__(self, rate=0.98):
        self.rate = rate
        super(BetaAnnealingCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.model.beta *= self.rate



class FuzzyVAE(tf.keras.Model):

    def __init__(self, input_shape, latent_size, beta=1E-5):
        super().__init__()

        self.beta = beta
        self.encoder_input_shape = input_shape
        self.latent_size = latent_size
        #self.initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1E-3)
        #self.initializer = None
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()


    def _build_encoder(self):

        encoder_layers = [
            tf.keras.layers.Input(shape=self.encoder_input_shape),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.latent_size + self.latent_size),
        ]
        return tf.keras.Sequential(encoder_layers, name='encoder')
    
    def _build_decoder(self):

        decoder_layers = [
            tf.keras.layers.Input(shape=(self.latent_size,)),
            tf.keras.layers.Dense(units=7*7*32, activation='relu'),
            tf.keras.layers.Reshape(target_shape=(7,7,32)),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'),
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.Softmax(),
        ]
        return tf.keras.Sequential(decoder_layers, name='decoder')


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
    
    def reparameterize(self, mean, logvar, training=False):
        eps = tf.zeros(shape=tf.shape(mean))
        if training:
            eps = tf.random.normal(shape=tf.shape(mean))
        #z = eps + tf.exp(logvar * .5) + mean
        z = mean + (logvar * 0.5) + eps
        return z
    
    def decode(self, z, apply_sigmoid=False):
        #logits = tf.clip_by_value(self.decoder(z), 1E-10, 1E10)
        logits = self.decoder(z)

        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.abs(tf.reduce_mean( -0.5 * (((sample - mean)**2.) * tf.exp(-logvar) + logvar + log2pi), axis=raxis))
    
    def call_detailed(self, x, training=False):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar, training)
        x_logit = self.decode(z, True)

        return x_logit, z, mean, logvar
    
    def call(self, x, training=False):
        mean, logvar = self.encode(x, training=False)
        z = self.reparameterize(mean, logvar, training)
        return self.decode(z, True)
    
    def compute_loss(self, x, training=False):
        #return self.compute_loss_old(x, training)
        return self.compute_loss_new(x, training)

    def compute_loss_new(self, x, training=False):

        KURTOSIS_TARGET = 3.0
        
        x_logit, z, mean, logvar = self.call_detailed(x, training)

        mse = tf.reduce_mean(tf.pow(x - x_logit, 2))
        mean_loss = tf.pow(tf.reduce_mean(mean),2)
        var_loss = tf.abs(1. - tf.reduce_mean(tf.pow(logvar, 2)))

        # Skewness - Balance around tails
        z_skew_loss = tf.abs(tf.reduce_mean(tf.pow((z-tf.reduce_mean(z)) / tf.math.reduce_std(z), 3)))
        # Kurtosis - Tailness of distribution (3=normal, 1.8=uniform)
        z_kurtosis_loss = tf.abs(KURTOSIS_TARGET - tf.reduce_mean(tf.pow((z - tf.reduce_mean(z))/ tf.math.reduce_std(z), 4)))

        kl_div_gaus = 0.5 * tf.abs(tf.reduce_sum(1. + tf.abs(logvar) - 2.*tf.abs(mean) - tf.exp(2.*tf.abs(logvar))))

        loss = 0.4 * mse + 0.2 * mean_loss + 0.2 * var_loss + 0.2 + z_kurtosis_loss
        #loss = mse + 5E-4*kl_div_gaus + mean_loss + var_loss + z_kurtosis_loss

        return {
            'loss': loss,
            'mse': mse,
            'mean_loss': mean_loss,
            'var_loss': var_loss,
            'skew_loss': z_skew_loss,
            'z_kurtosis_loss': z_kurtosis_loss,
            'r_min': tf.reduce_min(x_logit),
            'r_max': tf.reduce_max(x_logit),
            'kl_div': kl_div_gaus,
        }
    
    def compute_loss_old(self, x, training=False):
        x_logit, z, mean, logvar = self.call_detailed(x, training)

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        #mse = tf.pow(x_logit - x, 2) # cross entropy blows up

        logpx_z = tf.abs(tf.reduce_mean(cross_entropy))
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)

        loss = tf.abs(tf.reduce_mean(logpx_z + logpz - logqz_x))

        return {
            'loss': loss,
            'r_max': tf.reduce_max(x_logit),
            'r_min': tf.reduce_min(x_logit),
            'logpx_z': logpx_z,
            'logvar': tf.reduce_sum(tf.exp(-logvar)),
            'logpz': tf.reduce_sum(logpz),
            'logqz_x': tf.reduce_sum(logqz_x),
        }
        

    def train_step(self, x):

        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, training=True)

        #self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        grads = tape.gradient(loss['loss'], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss
    
    def test_step(self, x):

        loss = self.compute_loss(x, training=False)

        return loss




def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--latent-dim', '-d', type=int, default=128)
    parser.add_argument('--beta', type=float, default=1E-5)
    parser.add_argument('--batch-size', '-b', type=int, default=32)
    parser.add_argument('--max-epochs', '-e', type=int, default=10)
    parser.add_argument('--learning-rate', '-l', type=float, default=1E-4)
    
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
        learning_rate=args.learning_rate
    ))

    #vae.build((None, ) + vae.encoder_input_shape)
    #vae.summary()

    return vae


def train_model(args, model:tf.keras.Model, data):

    logdir = os.path.join('./logs', f'fit_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=logdir),
        BetaAnnealingCallback(0.98),
    ]

    try:
        model.fit(data['train'], validation_data=data['test'], batch_size=args.batch_size, callbacks=callbacks, shuffle=True, epochs=args.max_epochs)
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
    
    model.encoder.save(os.path.join(logdir, 'encoder'))
    model.decoder.save(os.path.join(logdir, 'decoder'))
    #model.save(os.path.join(logdir, 'model'))
    
    return model



def evaluate(model:tf.keras.Model, data):

    #n = 10
    #grid_x = np.linspace(0.0, 1.0, n)
    #grid_y = np.linspace(0.0, 1.0, n)
    
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
