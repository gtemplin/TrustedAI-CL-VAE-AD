#!/usr/bin/env python3

import os
import tensorflow as tf

class FuzzyVAE(tf.keras.Model):

    def __init__(self, config): #input_shape, latent_size, beta=1E-5):
        super().__init__()

        self.config = config

        self.beta = float(config['training']['beta'])
        self.encoder_input_shape = config['data']['image_size']
        self.latent_size = config['model']['latent_dimensions']
        
        loss_config = config['loss']
        self.kurtosis_target = float(loss_config['kurtosis'])
        self.w_mse = float(loss_config['w_mse'])
        self.w_kurtosis = float(loss_config['w_kurtosis'])
        self.w_skew = float(loss_config['w_skew'])
        self.w_kl_divergence = float(loss_config['w_kl_divergence'])
        self.w_z_l1_reg = float(loss_config['w_z_l1_reg'])
        self.w_x_std = float(loss_config['w_x_std'])
        
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

        x_std = tf.math.reduce_std(x, axis=0)
        x_hat_std = tf.math.reduce_std(x_hat_prob, axis=0)
        x_std_loss = tf.math.reduce_mean(tf.math.pow(x_std - x_hat_std, 2))
        
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

        #loss = self.w_mse * mse + self.w_kurtosis * z_kurtosis_loss + self.w_skew * z_skew_loss + self.w_z_l1_reg * z_l1_reg + self.w_x_std * x_std_loss
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
            'x_std_loss': x_std_loss,
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