#!/usr/bin/env python3

import os
import tensorflow as tf
import numpy as np

from src.abstract_cvae import AbstractCVAE

class KurtosisSingleCVAE(AbstractCVAE):

    def __init__(self, config):
        super().__init__(config)
        
        loss_config = config['loss']
        self.kurtosis_target = float(loss_config['kurtosis'])
        self.w_mse = float(loss_config['w_mse'])
        self.w_kurtosis = float(loss_config['w_kurtosis'])
        self.w_skew = float(loss_config['w_skew'])
        self.w_z_l1_reg = float(loss_config['w_z_l1_reg'])

        self.encoder.summary()
        self.decoder.summary()


    def compute_loss(self, x, training=False, return_inf=False):
        
        # Get VAE Outputs
        x_hat_prob, z, _, _ = self.call_detailed(x, training)

        # Calculate Mean Squared Error
        mse = tf.reduce_mean(tf.pow(x - x_hat_prob, 2))

        # Image Statistics
        x_std = tf.math.reduce_std(x, axis=0)
        x_hat_std = tf.math.reduce_std(x_hat_prob, axis=0)
        x_std_loss = tf.math.reduce_mean(tf.math.pow(x_std - x_hat_std, 2))
        
        # Statistics
        z_meu = tf.math.reduce_mean(z, axis=0)
        z_std = tf.math.reduce_std(z, axis=0)
        z_score = tf.math.divide_no_nan((z - z_meu), z_std)

        z_skew = tf.reduce_mean(tf.pow(z_score, 3), axis=0)
        z_kurtosis = tf.reduce_mean(tf.pow(z_score, 4), axis=0)
        
        # Estimator Losses
        z_kurtosis_loss = tf.math.reduce_mean(tf.math.pow(z_kurtosis - self.kurtosis_target, 2))
        z_skew_loss = tf.math.reduce_mean(tf.math.pow(z_skew, 2))

        # Z L2 Regularization
        z_l2_reg = tf.math.sqrt(tf.math.reduce_sum(tf.pow(z_meu, 2)))

        # Z L1 Regularization
        z_l1_reg = tf.reduce_mean(tf.abs(z))

        loss = \
            self.w_mse * mse + \
            self.w_kurtosis * z_kurtosis_loss + \
            self.w_skew * z_skew_loss + \
            self.w_z_l1_reg * z_l2_reg

        d = {
            'loss': loss,
            'mse': mse,
            'z_l1': z_l1_reg,
            'z_l2': z_l2_reg,
            'skew_loss': z_skew_loss,
            'z_kurtosis_loss': z_kurtosis_loss,
            'r_min': tf.reduce_min(x_hat_prob),
            'r_max': tf.reduce_max(x_hat_prob),
            'x_std_loss': x_std_loss,
        }
        if return_inf:
            return d, x_hat_prob
        else:
            return d
