#!/usr/env/bin python

import argparse
import numpy as np
import matplotlib.pyplot as plt

def vec_mag(x):
    #return np.sqrt(np.dot(x, x))
    return np.sqrt(np.sum(np.power(x,2)))

def slerp(theta, t, x1, x2):
    a = (np.sin((1. - t) * theta) / np.sin(theta)) * x1
    b = (np.sin(t * theta) / np.sin(theta)) * x2
    return a + b

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', '-n', type=int, default=50)
    parser.add_argument('--num-dims', '-d', type=int, default=3)
    args = parser.parse_args()
    
    num_samples = args.num_samples
    num_dims = args.num_dims
    assert(num_dims >= 3)

    v = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, num_dims))
    v_mag = np.sqrt(np.sum(np.power(v, 2), axis=1))
    #v = np.divide(v, v_mag)
    for i in range(v.shape[0]):
        v[i] = v[i] / v_mag[i]
    #print(vec_mag(v))

    x1 = 2*np.random.random(size=(num_dims,)) - 1
    x2 = 2*np.random.random(size=(num_dims,)) - 1

    mag_x1 = vec_mag(x1)
    mag_x2 = vec_mag(x2)

    x1 = x1 / mag_x1
    x2 = x2 / mag_x2

    mag_x1 = vec_mag(x1)
    mag_x2 = vec_mag(x2)

    print(f'X1: {x1}')
    print(f'X2: {x2}')

    print(f'Mag X1: {mag_x1}')
    print(f'Mag X2: {mag_x2}')

    dot_prod = np.dot(x1, x2)
    print(f'Dot Product: {dot_prod}')
    mag = mag_x1 * mag_x2
    print(f'Magnitude: {mag}')
    costheta = dot_prod / mag
    print(f'Cos(theta): {costheta}')
    #if costheta > 1.:
    #    costheta -= 1
    #if costheta < -1.:
    #    costheta += 1
    #print(f'Cos(theta): {costheta}')
    theta = np.arccos(costheta)
    theta_deg = 180. * theta / np.pi
    arc_len = theta * mag
    print(f'Theta: {theta}, {theta_deg}º')

    N = num_samples
    theta_delta = np.pi / (2*N)
    x_steps = list()
    for i in range(N-1):
        #t = (i+1)*theta_delta
        #x1_comp = mag_x1*mag_x1*norm_x1 * np.cos(t)
        #x2_comp = mag_x2*mag_x2*norm_x2 * np.sin(t)
        #z = x1_comp + x2_comp
        #z_mag = vec_mag(z)
        #z_norm = z / z_mag
        #step = np.sqrt(z_mag) * z_norm
        #x_steps.append(step)

        # SLERP
        t = (1+i) / N
        step = slerp(theta, t, x1, x2)
        x_steps.append(step)
    y = np.array(x_steps)

    x = np.vstack((x1, x2))
    print(x.shape)
    print(y.shape)

    fig = plt.figure()
    fig.suptitle(f'Spherical Interpolation on Shell :: Dimensions: {num_dims}\nTheta (D): {theta_deg:0.3f}º, Est. Arc Length: {arc_len:0.3f}')
    ax = fig.add_subplot(projection='3d')

    ax.plot((0., x[0,0]), (0., x[0,1]), (0., x[0, 2]), color='blue')
    ax.plot((0., x[1,0]), (0., x[1,1]), (0., x[1, 2]), color='blue')
    ax.scatter(x[:,0], x[:,1], x[:,2], color='blue')
    ax.scatter(y[:,0], y[:,1], y[:,2], color='red', s=5)
    ax.scatter(v[:,0], v[:,1], v[:,2], color='green', alpha=0.35)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)

    plt.show()


if __name__ == '__main__':
    main()
