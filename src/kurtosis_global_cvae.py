#!/usr/bin/env python3

import os
import tensorflow as tf
import numpy as np

from src.abstract_cvae import AbstractCVAE

class KurtosisGlobalCVAE(AbstractCVAE):

    def __init__(self, config): #input_shape, latent_size, beta=1E-5):
        super().__init__(config)
        
        loss_config = config['loss']
        self.kurtosis_target = float(loss_config['kurtosis'])
        self.w_mse = float(loss_config['w_mse'])
        self.w_kurtosis = float(loss_config['w_kurtosis'])
        self.w_skew = float(loss_config['w_skew'])
        self.w_kl_divergence = float(loss_config['w_kl_divergence'])
        self.w_z_l1_reg = float(loss_config['w_z_l1_reg'])
        self.w_x_std = float(loss_config['w_x_std'])

        self.encoder.summary()
        self.decoder.summary()

    
    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.abs(tf.reduce_mean( -0.5 * (((sample - mean)**2.) * tf.exp(-logvar) + logvar + log2pi), axis=raxis))
    
    
    def compute_loss(self, x, training=False, return_inf=False):
        #return self.compute_loss_old(x, training)
        return self.compute_loss_new(x, training, return_inf)
    
    def kl_divergence_gaussian(self, z_mean:tf.Tensor, z_logvar:tf.Tensor):
        # kl_div_gaussian = 1/2 * sum(1 + log(sigma^2) - mue^2 - sigma^2)
        return 0.5 * tf.reduce_sum(tf.abs(1. + z_logvar**2 - z_mean**2 - tf.exp(z_logvar**2)))

    def compute_loss_new(self, x, training=False, return_inf=False):
        
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
        z_score = tf.math.divide_no_nan((z - z_mean), z_std)
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

        d = {
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
        if return_inf:
            return d, x_hat_prob
        else:
            return d
        
    
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
        
