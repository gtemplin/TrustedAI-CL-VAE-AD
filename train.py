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

from src.fuzzy_vae import FuzzyVAE
from src.data_loader import load_data

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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('config_filename', type=str, help='YAML configuration file')
    parser.add_argument('--dry-run', action='store_true', help='Quit before executing training')
    
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
    #beta = float(config['training']['beta'])
    batch_size = int(config['training']['batch_size'])
    max_epochs = int(config['training']['max_epochs'])

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=logdir),
        BetaAnnealingCallback(),
        #tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule, verbose=True),
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
    if args.dry_run:
        exit()
    model = train_model(config, model, data)
    evaluate(config, model, data)


if __name__ == '__main__':
    main()
