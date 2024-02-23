#!/usr/bin/env python3

import os
import tensorflow as tf
import numpy as np

class AbstractCVAE(tf.keras.Model):

    def __init__(self, config): #input_shape, latent_size, beta=1E-5):
        super().__init__()

        self.config = config

        self.beta = float(config['training']['beta'])
        self.encoder_input_shape = config['data']['image_size']
        self.latent_size = config['model']['latent_dimensions']
        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()


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
        #encoder_layers.extend([
        #    tf.keras.layers.Flatten(),
        #    tf.keras.layers.Dense(self.latent_size + self.latent_size),
        #])

        encoder_layers.append(tf.keras.layers.Flatten())
        encoder_dense_filters = self.config['model'].get('encoder_dense_filters')
        if encoder_dense_filters:
            encoder_layers.append(tf.keras.layers.Dense(units=int(encoder_dense_filters)))
        encoder_layers.append(tf.keras.layers.Dense(self.latent_size + self.latent_size))

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
        for filters in reversed(self.config['model']['layers']):
            decoder_layers.append(
                tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same', activation='relu')
            )

        # Decoder Reconstruction Output
        decoder_layers.append(
            tf.keras.layers.Conv2DTranspose(filters=output_channels, kernel_size=3, strides=1, padding='same')
        )

        # Return Decoder
        return tf.keras.Sequential(decoder_layers, name='decoder')
    
    
    def load_model(self, model_path):
        assert(os.path.exists(model_path))
        assert(os.path.isdir(model_path))

        encoder_path = os.path.join(model_path, 'encoder')
        assert(os.path.exists(encoder_path))

        decoder_path = os.path.join(model_path, 'decoder')
        assert(os.path.exists(decoder_path))

        self.encoder = tf.keras.models.load_model(encoder_path)
        self.decoder = tf.keras.models.load_model(decoder_path)


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
    
    def call_detailed(self, x, training=False):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar, training)
        x_prob = self.decode(z, True)

        return x_prob, z, mean, logvar
    
    def call(self, x, training=False):
        mean, logvar = self.encode(x, training=False)
        z = self.reparameterize(mean, logvar, training)
        return self.decode(z, True)
    
    def compute_loss(self, x, training=False, return_inf=False):
        raise NotImplementedError('Error, compute_loss must be implemented')
    
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
    
    def train_step_and_run(self, x):

        x_hat = None
        with tf.GradientTape() as tape:
            loss, x_hat = self.compute_loss(x, training=True, return_inf=True)
        
        grads = tape.gradient(loss['loss'], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss, x_hat
