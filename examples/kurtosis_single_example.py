#!/usr/bin/env python3

import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--latent-variables', '-l', type=int, default=32)
    parser.add_argument('--batch-size', '-b', type=int, default=16)
    parser.add_argument('--num-steps', '-n', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1E-4)
    parser.add_argument('--target-kurtosis', '-t', type=float, default=3.0)
    parser.add_argument('--gaussian-init', action='store_true', help='Initialize with Gaussian instead of Uniform')

    args = parser.parse_args()

    num_latent_variables = args.latent_variables
    batch_size = args.batch_size
    num_steps = args.num_steps
    learning_rate = args.learning_rate
    target_kurtosis = args.target_kurtosis

    fig, ax = plt.subplots(1,1)

    with tf.device('/CPU:0'):

        if args.gaussian_init:
            x = tf.Variable(tf.random.normal(shape=(batch_size, num_latent_variables), mean=0.0, stddev=1.0, dtype=tf.float32), trainable=True, name='x')
        else:
            x = tf.Variable(tf.random.uniform(shape=(batch_size, num_latent_variables), minval=0., maxval=1.0, dtype=tf.float32), trainable=True, name='x')

        for epoch in range(num_steps):
            with tf.GradientTape() as tape:

                eps = tf.random.normal(x.shape, x, 1E-5)

                meu = tf.math.reduce_mean(eps, axis=0)
                std = tf.math.reduce_std(eps, axis=0)
                z = (eps - meu) / std
                skew = tf.math.reduce_mean(z**3, axis=0)
                kurtosis = tf.math.reduce_mean(z**4, axis=0)

                kurtosis_mean = tf.math.reduce_mean(tf.math.pow(kurtosis - target_kurtosis, 2.0))
                skew_mean = tf.math.reduce_mean(tf.math.pow(skew, 2.0))
                mean_loss = tf.math.sqrt(tf.reduce_mean(tf.pow(meu, 2)))

                loss = kurtosis_mean + mean_loss + skew_mean
                
                print(f'Epoch: {epoch}, Min Kurtosis: {tf.reduce_min(kurtosis):0.6f}, Max Kurtosis: {tf.reduce_max(kurtosis):0.6f}, Mean Kurtosis: {tf.reduce_mean(kurtosis):0.6f}, Mean: {mean_loss:0.6f}, Skew: {skew_mean:0.6f} Loss: {loss:0.6f}')

                ax.clear()
                for idx in range(num_latent_variables):
                    ax.hist(x.numpy()[:,idx], bins='auto', label=idx, alpha=0.35, density=True)
                ax.set_xlabel('Latent Value')
                ax.set_ylabel('Density')
                fig.canvas.draw()
                plt.pause(0.05)
            
            grad = tape.gradient(loss, x)
            x.assign_add(-learning_rate * grad)
        
    plt.show()


if __name__ == '__main__':
    main()
