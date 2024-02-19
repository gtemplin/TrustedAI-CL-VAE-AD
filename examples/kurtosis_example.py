#!/usr/bin/env python3

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', '-n', type=int, default=10_000, help='Number of samples')
    parser.add_argument('--dimensions', '-d', type=int, default=32, help='Number of dimensions')
    args = parser.parse_args()

    assert(args.num_samples > 0)
    assert(args.dimensions > 0)

    N = args.num_samples
    d = args.dimensions

    x = np.random.uniform(0., 1., size=(N, d))
    #x = np.random.normal(0.0, 1.0, size=(N,d))
    #x = np.random.gamma(255., 0.25, size=(N,d))

    meu = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    z = (x - meu) / std

    skew = np.mean(np.power(z, 3), axis=0)
    kurt = np.mean(np.power(z, 4), axis=0)

    print(f'Mean: \n{meu}')
    print(f'Std.Dev: \n{std}')
    print(f'Skew: \n{skew}')
    print(f'Kurt: \n{kurt}')

    for i in range(x.shape[1]):
        plt.hist(x[:,i], bins='auto', alpha=0.25)
    plt.show()




if __name__ == '__main__':
    main()
