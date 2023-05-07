#!/usr/bin/env python

import argparse, os, sys
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import datetime
import yaml

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import plotly.express as px

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


def learning_rate_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


class BetaAnnealingCallback(tf.keras.callbacks.Callback):

    def __init__(self, rate=0.98):
        self.rate = rate
        super(BetaAnnealingCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.model.beta *= self.rate



class FuzzyVAE(tf.keras.Model):

    def __init__(self, config): #input_shape, latent_size, beta=1E-5):
        super().__init__()

        self.config = config

        self.beta = config['training']['beta']
        self.encoder_input_shape = config['data']['image_size']
        self.latent_size = config['model']['latent_dimensions']
        
        loss_config = config['loss']
        self.kurtosis_target = float(loss_config['kurtosis'])
        self.w_mse = float(loss_config['w_mse'])
        self.w_kurtosis = float(loss_config['w_kurtosis'])
        self.w_skew = float(loss_config['w_skew'])
        self.w_kl_divergence = float(loss_config['w_kl_divergence'])
        self.w_z_l1_reg = float(loss_config['w_z_l1_reg'])
        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        self.encoder.summary()
        self.decoder.summary()


    def _build_encoder(self):

        # Encoder Input
        encoder_layers = [
            tf.keras.layers.Input(shape=self.encoder_input_shape),
        ]

        # Encoder Hidden Layers
        for filters in self.config['model']['layers']:
            encoder_layers.append(
                tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=(2,2), padding='same', activation='relu')
            )
        
        # Encoder Latent Output
        encoder_layers.extend([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.latent_size + self.latent_size),
        ])

        # Return Encoder
        return tf.keras.Sequential(encoder_layers, name='encoder')
    

    def _build_decoder(self):

        image_size = self.config['data']['image_size']
        image_width = image_size[0]
        image_height = image_size[1]
        output_channels = image_size[2]
        layer_count = len(self.config['model']['layers'])

        # Calculate Dense Transformation
        dense_width = int(float(image_width) / float(2**layer_count))
        dense_height = int(float(image_height) / float(2**layer_count))
        decoder_dense_filters = self.config['model']['decoder_dense_filters']

        # Error check dense vs image size vs layers
        if dense_width == 0:
            raise RuntimeError(f'Error: Build Decoder: Width Collapse: Too many layers, check configuration file: {image_width} -> {dense_width}: {layer_count} Layers')
        if  dense_height == 0:
            raise RuntimeError(f'Error: Build Decoder: Height Collapse: Too many layers, check configuration file: {image_height} -> {dense_height}: {layer_count} Layers')
        
        dense_units = dense_width * dense_height * decoder_dense_filters
        dense_shape = (dense_width, dense_height, decoder_dense_filters)

        # Decoder Latent Input
        decoder_layers = [
            tf.keras.layers.Input(shape=(self.latent_size,)),
            tf.keras.layers.Dense(units=dense_units, activation='relu'),
            tf.keras.layers.Reshape(target_shape=dense_shape),
        ]

        # Decoder Hidden Layers
        for filters in self.config['model']['layers']:
            decoder_layers.append(
                tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same', activation='relu')
            )

        # Decoder Reconstruction Output
        decoder_layers.append(
            tf.keras.layers.Conv2DTranspose(filters=output_channels, kernel_size=3, strides=1, padding='same')
        )

        # Return Decoder
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
        z = mean + (logvar * 0.5) + eps
        return z
    
    def decode(self, z, apply_sigmoid=False):
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
        x_prob = self.decode(z, True)

        return x_prob, z, mean, logvar
    
    def call(self, x, training=False):
        mean, logvar = self.encode(x, training=False)
        z = self.reparameterize(mean, logvar, training)
        return self.decode(z, True)
    
    def compute_loss(self, x, training=False):
        #return self.compute_loss_old(x, training)
        return self.compute_loss_new(x, training)
    
    def kl_divergence_gaussian(self, z_mean:tf.Tensor, z_logvar:tf.Tensor):
        # kl_div_gaussian = 1/2 * sum(1 + log(sigma^2) - mue^2 - sigma^2)
        return 0.5 * tf.reduce_sum(tf.abs(1. + z_logvar**2 - z_mean**2 - tf.exp(z_logvar**2)))

    def compute_loss_new(self, x, training=False):
        
        # Get VAE Outputs
        x_hat_prob, z, mean, logvar = self.call_detailed(x, training)
        
        # Calculate Entropy
        x_logit = tf.math.log(tf.exp(x) / tf.reduce_sum(tf.exp(x)))
        likelihood_cross_entropy = -tf.reduce_mean(x_hat_prob * x_logit)

        # Calculate Mean Squared Error
        mse = tf.reduce_mean(tf.pow(x - x_hat_prob, 2))
        
        # Old Statistics Calculations
        #mean_loss = tf.pow(tf.reduce_mean(mean),2)
        #var_loss = tf.abs(1. - tf.reduce_mean(tf.pow(logvar, 2)))
        
        # Statistics
        z_mean = tf.reduce_mean(z)
        z_std = tf.math.reduce_std(z)
        z_var = tf.math.reduce_variance(z)
        z_score = (z - z_mean) / z_std
        z_skew = tf.reduce_mean(tf.pow(z_score, 3))
        z_kurtosis = tf.reduce_mean(tf.pow(z_score, 4))
        
        # Losses
        
        # Mean = 0
        mean_loss = tf.pow(z_mean, 2)
        # Var = 1
        var_loss = tf.abs(1. - z_var)
        # Skewness - Balance around tails
        z_skew_loss = tf.abs(z_skew)
        # Kurtosis - Tailness of distribution (3=normal, 1.8=uniform)
        z_kurtosis_loss = tf.abs(self.kurtosis_target - z_kurtosis)

        # KL Divergence from Raw Encoder Output
        kl_div_gaus = self.kl_divergence_gaussian(mean, logvar)

        # Z L1 Regularization
        z_l1_reg = tf.reduce_mean(tf.abs(z))

        ##loss = mse
        #loss = mse + 1E-5 * z_kurtosis_loss
        ##loss = 0.4 * mse + 0.2 * mean_loss + 0.2 * var_loss + 0.2 * z_kurtosis_loss
        ##loss = mse + 1E-5 * kl_div_gaus

        #loss = self.w_mse * mse + self.w_kurtosis * z_kurtosis_loss + self.w_skew * z_skew_loss + self.w_kl_divergence * kl_div_gaus + self.w_z_l1_reg * z_l1_reg
        loss = self.w_mse * mse + self.w_kurtosis * z_kurtosis_loss + self.w_skew * z_skew_loss + self.w_z_l1_reg * z_l1_reg

        return {
            'loss': loss,
            'mse': mse,
            'z_l1': z_l1_reg,
            'var_loss': var_loss,
            'skew_loss': z_skew_loss,
            'z_kurtosis_loss': z_kurtosis_loss,
            'r_min': tf.reduce_min(x_hat_prob),
            'r_max': tf.reduce_max(x_hat_prob),
            'cross_entropy': likelihood_cross_entropy,
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

    parser.add_argument('config_filename', type=str, help='YAML configuration file')
    
    return parser.parse_args()

def load_config(config_filename: str):

    assert(os.path.exists(config_filename))
    assert(os.path.isfile(config_filename))

    # Load config file
    config = None
    try:
        with open(config_filename, 'r') as ifile:
            config = yaml.safe_load(ifile)

    except IOError as e:
        raise e
    except yaml.YAMLError as e:
        raise e
    
    # Assign and create logdir now
    config['logdir'] =  os.path.join('./logs', f'fit_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    if not os.path.exists(config['logdir']):
        os.makedirs(config['logdir'])
    else:
        assert(os.path.isdir(config['logdir']))

    # Save YAML config file to log directory
    try:
        log_config_filepath = os.path.join(config['logdir'], 'config.yml')
        with open(log_config_filepath, 'w') as ofile:
            yaml.safe_dump(config, ofile)
    except IOError as e:
        raise e
    except yaml.YAMLError as e:
        raise e

    return config 
    

def load_data(config: dict):

    data_config = config['data']
    dataset_name = data_config['dataset']
    train_split = data_config['train_split']
    val_split = data_config['val_split']
    img_size = data_config['image_size']
    batch_size = config['training']['batch_size']

    r_img_size = (img_size[0], img_size[1])

    train_ds, ds_info = tfds.load(dataset_name, split=train_split, shuffle_files=True, download=False, with_info=True)
    val_ds = tfds.load(dataset_name, split=val_split, shuffle_files=True, download=False, with_info=False)

    def normalize_img(element):
        return tf.cast(element['image'], tf.float32) / 255.
    
    train_ds = train_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

    def resize_img(element, img_size):
        return tf.image.resize(element, size=img_size)
    
    train_ds = train_ds.map(lambda x: resize_img(x, r_img_size), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x: resize_img(x, r_img_size), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    print(f'Length of Training data: {len(train_ds)}')

    return {
        'train': train_ds,
        'val': val_ds,
        'info': ds_info,
    }


def build_model(config: dict, data: tf.data.Dataset):

    vae = FuzzyVAE(config)

    vae.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=float(config['training']['learning_rate'])
    ))

    #vae.build((None, ) + vae.encoder_input_shape)
    #vae.summary()

    return vae


def train_model(config, model:tf.keras.Model, data):

    logdir = config['logdir']
    beta = config['training']['beta']
    batch_size = config['training']['batch_size']
    max_epochs = config['training']['max_epochs']

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=logdir),
        BetaAnnealingCallback(beta),
        tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule, verbose=True),
    ]

    try:
        model.fit(data['train'], validation_data=data['val'], batch_size=batch_size, callbacks=callbacks, shuffle=True, epochs=max_epochs, use_multiprocessing=True, workers=8)
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
    
    model.encoder.save(os.path.join(logdir, 'encoder'))
    model.decoder.save(os.path.join(logdir, 'decoder'))
    #model.save(os.path.join(logdir, 'model'))
    
    return model



def evaluate(config: dict, model:tf.keras.Model, data):

    logdir = config['logdir']

    n = 10
    
    test_data = data['val'].unbatch()
    x_i = np.array(list(test_data.take(n).as_numpy_iterator()))
    y = model.predict(x_i)
    z = model.encode(x_i)
    
    y_i = (y - np.min(y)) / (np.max(y) - np.min(y))
    
    #x_i = x_i[:,:,:,0]
    #y_i = y_i[:,:,:,0]

    print(x_i.shape)
    print(y_i.shape)

    print(tf.reduce_max(x_i))
    print(tf.reduce_max(y_i))

    #fig, ax_vec = plt.subplots(n, 2, figsize=(5,12))
    #
    #for i,ax_tuple,x, x_hat in zip(range(n),ax_vec, x_i, y_i):
    #    ax_tuple[0].imshow(x, cmap='gray')
    #    
    #    ax_tuple[1].imshow(x_hat, cmap='gray')
    #    
    #    if i == 0:
    #        ax_tuple[0].set_title('Original')
    #        ax_tuple[1].set_title('Prediction')
    #
    #plt.show()
    
    fig_original = px.imshow(np.round(255. * x_i), facet_col=0, facet_col_wrap=5)
    fig_reconstruction = px.imshow(np.round(255. * y_i), facet_col=0, facet_col_wrap=5)
    
    print('Saving Original')
    fig_original.write_image(os.path.join(logdir, "original.png"))
    print('Saving Reconstruction')
    fig_reconstruction.write_image(os.path.join(logdir, "reconstruction.png"))
    
    
    print('Generating Image Histogram')
    fig, ax = plt.subplots(1, 1)
    ax.hist(x_i.flatten(), bins=64, label='Original', alpha=0.65)
    ax.hist(y_i.flatten(), bins=64, label='Reconstruction', alpha=0.65)
    ax.grid()
    ax.legend()
    ax.set_title('Flat Image Histogram')
    fig.savefig(os.path.join(logdir, "output_histogram.png"))

    print('Generating Latent Histogram')
    fig, ax = plt.subplots(1, 1)
    ax.hist(tf.reshape(z, [-1]), bins=64)
    ax.grid()
    ax.set_title('Latent Vector Histogram')
    fig.savefig(os.path.join(logdir, "latent_histogram.png"))
    
    


def main():

    
    args = get_args()
    
    config = load_config(args.config_filename)
    data = load_data(config)
    model = build_model(config, data)
    model = train_model(config, model, data)
    evaluate(config, model, data)


if __name__ == '__main__':
    main()
