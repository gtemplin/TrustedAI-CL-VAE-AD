#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

class Particle(object):

    def __init__(self, config: dict):
        self.config = config
        self.x = np.random.normal(0.0, 1.0, size=(config['dimensions'],))
        self.v = np.random.uniform(-0.5, 0.5, size=(config['dimensions'],))
        self.fitness = -float('inf')

        self.pbx = deepcopy(self.x)
        self.pb_fitness = -float('inf')

        self.lbx = deepcopy(self.x)
        self.lb_fitness = -float('inf')

        self.neighbors = []

    def fly(self):

        d = self.x.shape[0]

        I = self.config['I'] * np.random.uniform(0.75, 1.0, size=d) * self.v
        v_pb = self.config['C1'] * np.random.uniform(0.75, 1.0, size=(d,)) * (self.pbx - self.x)
        v_lb = self.config['C2'] * np.random.uniform(0.75, 1.0, size=(d,)) * (self.lbx - self.x)

        alpha = np.exp(self.pb_fitness)
        beta = np.exp(self.lb_fitness)
        T = alpha + beta
        alpha /= T
        beta /= T

        if T != 0.0:
            self.v = I + alpha * v_pb + beta * v_lb
        else:
            self.v = I + v_pb + v_lb
        self.x += self.v * self.config['dt']
        return np.sum(np.power(self.v, 2))

    def evaluate(self):
        self.fitness = self.get_fitness()
        self.update_pb()
        return self.fitness

    def update_pb(self):
        if self.fitness > self.pb_fitness:
            self.pb_fitness = deepcopy(self.fitness)
            self.pbx = deepcopy(self.x)

    def update_lb(self):
        lb_fitness = deepcopy(self.lb_fitness)
        lb_idx = None

        for i,n in enumerate(self.neighbors):
            if n.pb_fitness > lb_fitness:
                lb_fitness = n.pb_fitness
                lb_idx = i

        if lb_idx:
            self.lb_fitness = deepcopy(lb_fitness)
            self.lbx = deepcopy(self.neighbors[lb_idx].pbx)

    def get_fitness(self):

        meu = np.mean(self.x)
        std = np.std(self.x)
        if std > 0:
            z = (self.x - meu) / std
        else:
            z = 0.0
        skew = np.mean(np.power(z, 3))
        kurt = np.mean(np.power(z, 4))

        target_kurtosis = self.config['target_kurtosis']
        #loss = np.abs(target_kurtosis - kurt) + np.power(target_kurtosis - kurt, 2)
        #loss = np.power(target_kurtosis - kurt, 2) + np.power(skew, 2)
        loss = np.power(target_kurtosis - kurt, 2)
        #loss = np.abs(target_kurtosis - kurt)

        #max_x = np.max(self.x)
        #max_x_loss = np.power(max_x - 1.0, 2)
        #loss += self.config['max_weight'] * max_x_loss

        #min_x = np.min(self.x)
        #min_x_loss = np.power(min_x + 1.0, 2)
        #loss += self.config['max_weight'] * min_x_loss

        self.fitness = -loss
        return self.fitness

class PSO(object):

    def __init__(self, config:dict):
        self.config = config
        self.particles = [Particle(self.config) for _ in range(self.config['population'])]

        N = self.config['neighbors']
        N_2 = N//2
        for i,p in enumerate(self.particles):
            left = (i - N_2 + len(self.particles)) % len(self.particles)
            right = (i + N_2 + len(self.particles)) % len(self.particles)
            for n_idx in range(left, right):
                if n_idx == i:
                    continue
                p.neighbors.append(self.particles[n_idx])

    def process(self):

        plt.ion()

        epoch = 0
        history = []

        if self.config['plotter']:
            fig, ax = plt.subplots(1,1)

        try:
            while True:
                energy = self.fly()
                epoch_best = self.evaluate()
                history.append(epoch_best)

                print(f'Epoch {epoch}: {epoch_best}, E({energy})')

                epoch += 1

                if len(history) > 1 and self.config['plotter']:
                    if history[-2] != epoch_best:
                        plt.cla()
                        ax.hist(self.get_gb().pbx, bins='auto')
                        fig.canvas.draw()
                        plt.pause(0.1)

                if epoch > self.config['max_iterations']:
                    break
                if epoch_best >= self.config['min_fitness']:
                    break
        except KeyboardInterrupt:
            print('Terminating by keyboard')
        except Exception as e:
            raise e
            
        plt.ioff()
        return self.get_gb()
    
    def fly(self):
        energy = list()
        for p in self.particles:
            energy.append(p.fly())
        return np.mean(energy)
    def evaluate(self):
        for p in self.particles:
            p.evaluate()
        for p in self.particles:
            p.update_lb()
        return self.get_gb().pb_fitness
    def get_gb(self):
        pb_fit = -float('inf')
        pb_idx = None
        for i,p in enumerate(self.particles):
            if p.pb_fitness > pb_fit:
                pb_fit = p.pb_fitness
                pb_idx = i
        if pb_idx:
            return self.particles[pb_idx]
        else:
            return self.particles[0]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--target-kurtosis', '-k', type=float, default=1.8, help='Set target kurtosis (1.8~U, 3.0~N, 6.0~L)')
    parser.add_argument('--max-iterations', '-m', type=int, default=200, help='Max epoch iterations')
    parser.add_argument('--dimensions', '-d', type=int, default=1000, help='Dimensions of particles')
    parser.add_argument('--plotter', '-p', action='store_true', help='Plot over epochs')
    parser.add_argument('--batch-mode', '-b', action='store_true', help='Run plot on various distributions')
    args = parser.parse_args()

    assert(args.target_kurtosis >= 0)
    assert(args.max_iterations > 0)
    assert(args.dimensions > 0)

    config = {
        'population': 200,
        'neighbors': 30,
        'dimensions': args.dimensions,
        'max_iterations': args.max_iterations,
        'min_fitness': -1E-20,
        'I': 1.0,
        'C1': 1.0,
        'C2': 1.0,
        'dt': .5,
        'target_kurtosis': args.target_kurtosis,
        #'max_weight': 1.0E-3,
        'plotter': args.plotter,
    }

    if args.batch_mode:

        arg_list = [
            (f'Uniform: K= {1.8}', 1.8),
            (f'Gaussian: K= {3.0}', 3.0),
            (f'Laplace: K= {6.0}', 6.0),
            #(f'Gamma: K= {9.0}', 9.0),
        ]

        pso_best_list = []
        for row in arg_list:
            config['target_kurtosis'] = row[1]
            pso = PSO(config)
            gb = pso.process()
            pso_best_list.append(gb.pbx)
            
        fig, ax = plt.subplots(len(arg_list), 1)

        fig.suptitle(f'Dimensions: {args.dimensions}')

        for i,(row,x) in enumerate(zip(arg_list, pso_best_list)):
            ax[i].hist(x, bins='auto', density=True)
            ax[i].set_title(row[0])
            ax[i].grid()
            ax[i].set_xlabel('X Values')
            ax[i].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    else:

        pso = PSO(config)
        gb = pso.process()

        print(f'Best Fit: {gb.pb_fitness}')
        print(f'GB X: {gb.pbx}')

        plt.suptitle(f'Kurtosis: {args.target_kurtosis}, Dimensions: {args.dimensions}')
        plt.hist(gb.pbx, bins='auto', density=True)
        plt.xlabel('X Values')
        plt.ylabel('Frequency')
        plt.grid()
        plt.show()



if __name__ == '__main__':
    main()
