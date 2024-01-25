#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt


class BSTProb(object):

    def __init__(self, x:[list, np.ndarray], probs:[list, np.ndarray]):
        self.x = x
        self.probs = probs

        assert(len(x) == len(probs))
        assert(len(x) > 0)

        self._build_tree()

    def _build_tree(self):

        meu = np.dot(self.x, self.probs)
        self._tree = self._step_down(self.x, self.probs, meu, 0)

    def _step_down(self, x, probs, meu, parent_depth):

        if len(x) == 1:
            return {
                'key': meu,
                'prob': probs[0],
                'depth': parent_depth + 1,
            }
        elif len(x) == 0:
            return {}
        else:
            r = {}

            left_x = x[x <= meu]
            if len(left_x) > 0:
                left_probs = probs[x <= meu]
                left_meu = np.dot(left_x, left_probs)
                r['left'] = self._step_down(left_x, left_probs, left_meu, parent_depth+1)

            right_x = x[x > meu]
            if len(right_x) > 0:
                right_probs = probs[x > meu]
                right_meu = np.dot(right_x, right_probs)
                r['right'] = self._step_down(right_x, right_probs, right_meu, parent_depth+1)
            
            return r
        
    def __getitem__(self, x):
        





class CDFObject(object):

    def __init__(self, x:[list,np.ndarray], bins:[int,str]='auto'):
        self.x = x
        self.hist, self.bin_edges = np.histogram(x, bins=bins, density=True)
        self.hist = self.hist / np.sum(self.hist)
        self.bin_mid = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2.
        self.bin_width = np.mean(self.bin_edges[1:] - self.bin_edges[:-1])
        self.meu = np.dot(self.hist, self.bin_mid)

        self.cdf_tree = BSTProb(x=self.bin_edges[1:], probs=self.hist)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', '-a', type=float, default=2.0, help='Gamma(Alpha, beta)')
    parser.add_argument('--beta', '-b', type=float, default=4.0, help='Gamma(alpha, Beta)')
    parser.add_argument('--num-samples', '-n', type=int, default=10000, help='Number of RV samples')
    args = parser.parse_args()

    x = np.random.gamma(shape=args.alpha, scale=1./args.beta, size=(args.num_samples,))
    hist, bin_edges = np.histogram(x, bins='auto', density=True)
    
    hist_norm = hist / np.sum(hist)
    bin_mid = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = np.mean(bin_edges[1:] - bin_edges[:-1])
    print(f'Bin Widths: {bin_width}')
    meu = np.dot(hist_norm, bin_mid)
    print(f'Hist Sum: {np.sum(hist_norm)}')
    print(f'Mean: {meu}')

    print(np.sum(x) / len(x))

    fig, (ax0, ax1) = plt.subplots(2, 1)

    ax0.set_title('Histogram (plt)')
    ax0.hist(x, bins='auto', density=True)
    ax0.grid()
    
    ax1.set_title('Histogram (np)')
    ax1.bar(x=bin_mid, height=hist_norm, width=bin_width)
    ax1.grid()
    

    plt.show()



if __name__ == '__main__':
    main()
