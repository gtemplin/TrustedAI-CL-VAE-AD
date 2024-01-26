#!/usr/bin/env python3

import numpy as np


class BSTProb(object):

    def __init__(self, x:[list, np.ndarray], probs:[list, np.ndarray], match_fun=lambda a,b: a <= b):
        self.reset(x, probs, match_fun)

    def reset(self, x, probs, match_fun=None):
        assert(len(x) == len(probs))
        assert(len(x) > 0)
        if match_fun:
            self.match_fun = match_fun
        self.x, self.probs = zip(*sorted(zip(x, probs), key=lambda a: a[0]))
        self.x = np.array(self.x)
        self.probs = np.array(self.probs)
        self._build_tree()


    def _build_tree(self):

        meu = np.mean(self.x)
        self._tree = self._step_down(self.x, self.probs, meu, 0)
        assert(self._tree)
        self._tree['parent'] = None

    def _step_down(self, x, probs, meu, parent_depth):

        if len(x) == 1 or np.min(x) == np.max(x):
            return {
                'key': meu,
                'prob': probs[0],
                'depth': parent_depth + 1,
            }
        elif len(x) == 0:
            return None
        else:
            r = {'key': meu, 'depth': parent_depth + 1}

            left_match = self.match_fun(x, meu)
            left_x = x[left_match]
            if len(left_x) > 0:
                left_probs = probs[left_match]
                left_meu = np.mean(left_x)
                r['left'] = self._step_down(left_x, left_probs, left_meu, parent_depth+1)
                if r['left']:
                    r['left']['parent'] = r

            right_match = np.logical_not(left_match)
            right_x = x[right_match]
            if len(right_x) > 0:
                right_probs = probs[right_match]
                right_meu = np.mean(right_x)
                r['right'] = self._step_down(right_x, right_probs, right_meu, parent_depth+1)
                if r['right']:
                    r['right']['parent'] = r
            
            return r
        
    def __getitem__(self, x):
        walk = self._tree
        while 'prob' not in walk:

            left_match = self.match_fun(x, walk['key'])
            if left_match and 'left' in walk:
                walk = walk['left']
            elif not left_match and 'right' in walk:
                walk = walk['right']
            else:
                raise Exception('Error: BSTProb[], should never reach here')

        if 'prob' not in walk:
            raise Exception('Error, node is missing element')

        return walk['prob']



class CDFObject(object):

    def __init__(self, x:[list,np.ndarray], bins:[int,str]='auto'):
        self.reset(x, bins)

    def reset(self, x, bins=None):
        self.x = x
        if bins:
            self.bins = bins
        self.hist, self.bin_edges = np.histogram(self.x, bins=self.bins, density=True)
        self.hist = self.hist / np.sum(self.hist)
        self.bin_mid = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2.
        self.bin_width = np.mean(self.bin_edges[1:] - self.bin_edges[:-1])
        self.meu = np.dot(self.hist, self.bin_mid)

        mask = np.ones(shape=(len(self.hist), len(self.hist)), dtype=self.hist.dtype)
        mask[np.triu_indices(len(self.hist), 1)] = 0.0
        self.cdf = np.sum(np.multiply(self.hist, mask), axis=-1)

        self.cdf_tree = BSTProb(x=self.bin_edges[1:], probs=self.cdf)
        self.cdf_tree_inv = BSTProb(x=self.cdf, probs=self.bin_edges[1:])
        
    def get_prob_by_value(self, x):
        return self.cdf_tree[x]
    
    def get_value_by_prob(self, p):
        return self.cdf_tree_inv[p]
            


def main():

    import argparse
    import datetime

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

    #bst = BSTProb(bin_edges[1:], hist_norm)
    cdf_bst = CDFObject(x)
    print('Get Probability from Value')
    for a in np.linspace(0.0, 3.0, 30):
        print(f' - {a:0.03f}: {cdf_bst.get_prob_by_value(a):0.03f}')

    print('Get Value from Probability')
    for p in np.linspace(0.0, 1.0, 10):
        print(f' - {p:0.03f}: {cdf_bst.get_value_by_prob(p):0.03f}')

    print(f'95%: {cdf_bst.get_value_by_prob(0.95)}')

    start_time = datetime.datetime.now()
    for i in range(1000):
        x = np.random.gamma(shape=args.alpha, scale=1./args.beta, size=(args.num_samples,))
        cdf_bst.reset(x)
    end_time = datetime.datetime.now()
    time_delta = (end_time - start_time).total_seconds()

    print(f'Time Delta for 1000 runs: {time_delta} s, {time_delta/1000.} s/frame')

if __name__ == '__main__':
    main()
    