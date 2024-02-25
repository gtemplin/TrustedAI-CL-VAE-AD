#!/usr/bin/env python3

import unittest

import os
os.environ['PYTHONHASHSEED']=str(42)

import numpy as np
import tensorflow as tf
import random

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

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


def get_test_config():
    return {
        'data': {
            'image_size': [ 224, 300, 3],
        },
        'loss': {
            'kurtosis': 3.0,
            'w_kl_divergence': 0.0,
            'w_kurtosis': 1E-3,
            'w_mse': 1.0,
            'w_skew': 0.0,
            'w_x_std': 1E-10,
            'w_z_l1_reg': 1E-3,
        },
        'model': {
            'decoder_dense_filters': 4,
            'encoder_dense_filters': 4,
            'latent_dimensions': 2,
            'layers': [ 5, 5 ]
        },
        'training': {
            'batch_size': 16,
            'beta': 1E-6,
            'learning_rate': 1E-4,
            'max_epochs': 10,
        },
    }

class TestKurtosisGlobalCVAE(unittest.TestCase):

    def test_import_kurtosis_global_cvae(self):
        from src.kurtosis_global_cvae import KurtosisGlobalCVAE
        self.assertIsNotNone(KurtosisGlobalCVAE)

    def test_dummy_build(self):

        from src.kurtosis_global_cvae import KurtosisGlobalCVAE

        config = get_test_config()
        model = KurtosisGlobalCVAE(config)
        self.assertIsNotNone(model)

    def test_encoder_layers(self):

        from src.kurtosis_global_cvae import KurtosisGlobalCVAE

        config = get_test_config()
        model = KurtosisGlobalCVAE(config)
        self.assertEqual(len(config['model']['layers']) + 3, len(model.encoder.layers))


    def test_decoder_layers(self):

        from src.kurtosis_global_cvae import KurtosisGlobalCVAE

        config = get_test_config()
        model = KurtosisGlobalCVAE(config)
        self.assertEqual(len(config['model']['layers']) + 3, len(model.decoder.layers))

    def test_encoder_layers(self):

        from src.kurtosis_global_cvae import KurtosisGlobalCVAE

        config = get_test_config()
        model = KurtosisGlobalCVAE(config)
        latent_output_size = model.encoder.layers[-1].variables[0].shape[0]
        self.assertEqual(config['model']['latent_dimensions'] * 2, latent_output_size)

    def test_input_output_shape(self):

        from src.kurtosis_global_cvae import KurtosisGlobalCVAE

        config = get_test_config()
        model = KurtosisGlobalCVAE(config)
        input_shape = list(model.encoder.layers[0].input_shape[1:])
        config_input_shape = list(config['data']['image_size'])
        self.assertListEqual(input_shape, config_input_shape)

    def test_encoder_filters(self):

        from src.kurtosis_global_cvae import KurtosisGlobalCVAE

        config = get_test_config()
        model = KurtosisGlobalCVAE(config)

        for idx in range(0, len(model.encoder.layers)-3):
            layer = model.encoder.layers[idx]
            filters = layer.filters
            config_filters = config['model']['layers'][idx]
            self.assertEqual(filters, config_filters)

        filters = model.encoder.layers[-2].units
        config_filters = config['model']['encoder_dense_filters']
        self.assertEqual(filters, config_filters)


    def test_decoder_filters(self):
        
        from src.kurtosis_global_cvae import KurtosisGlobalCVAE

        config = get_test_config()
        model = KurtosisGlobalCVAE(config)

        for idx in range(2, len(model.decoder.layers)-1):
            layer = model.decoder.layers[idx]
            filters = layer.filters
            cidx = len(config['model']['layers']) - idx + 1
            config_filters = config['model']['layers'][cidx]
            self.assertEqual(filters, config_filters)

        dense_units = model.decoder.layers[0].units

        image_width, image_height, _ = config['data']['image_size']
        layer_count = len(config['model']['layers'])
        dense_width = int(float(image_width) / float(2**layer_count))
        dense_height = int(float(image_height) / float(2**layer_count))
        config_dense_units = dense_width * dense_height * config['model']['decoder_dense_filters']
        
        self.assertEqual(dense_units, config_dense_units)


    def test_loss(self):

        from src.kurtosis_global_cvae import KurtosisGlobalCVAE

        exp_loss = {
            'loss': 0.08541792,
            'mse': 0.083257124,
            'z_l1': 0.16079533,
            'var_loss': 0.9741449,
            'skew_loss': 0.0, 
            'z_kurtosis_loss': 2.0, 
            'r_min': 0.49963754,
            'r_max': 0.5003504, 
            'cross_entropy': 6.1276054, 
            'kl_div': 0.03022772, 
            'x_std_loss': 0.0,
        }

        with tf.device('/CPU:0'):
            config = get_test_config()
            model = KurtosisGlobalCVAE(config)

            x = np.random.random(size= [1, ] + config['data']['image_size']).astype(np.float32)
            loss = model.compute_loss(x, training=False)

            for k,v in loss.items():
                self.assertAlmostEqual(v.numpy(), exp_loss[k], places=6)


