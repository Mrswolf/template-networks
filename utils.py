# -*- coding: utf-8 -*-
import torch
import scipy
import numpy as np
import itertools

from brainda.paradigms import MotorImagery, SSVEP

def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y

def generate_tensors(*args, dtype=torch.float):
    new_args = []
    for arg in args:
        new_args.append(torch.as_tensor(arg, dtype=dtype))
    if len(new_args) == 1:
        return new_args[0]
    else:
        return new_args
    
def compute_pvals_wilcoxon(scores, order=None):
    '''Returns kxk matrix of p-values computed via the Wilcoxon rank-sum test,
    order defines the order of rows and columns

    df: DataFrame, samples are index, columns are pipelines, and values are
    scores

    order: list of length (num algorithms) with names corresponding to columns
    of df

    '''
    n_algo = scores.shape[1]
    out = np.zeros((n_algo, n_algo))
    for i in range(n_algo):
        for j in range(n_algo):
            if i != j:
                p = scipy.stats.wilcoxon(scores[:, i], scores[:, j])[1]
                p /= 2
                # we want the one-tailed p-value
                diff = (scores[:, i]-scores[:, j]).mean()
                if diff < 0:
                    p = 1 - p  # was in the other side of the distribution
                out[i, j] = p
    return out


def _pairedttest_exhaustive(data):
    '''Returns p-values for exhaustive ttest that runs through all possible
    permutations of the first dimension. Very bad idea for size greater than 12

    data is a (subj, alg, alg) matrix of differences between scores for each
    pair of algorithms per subject

    '''
    out = np.ones((data.shape[1], data.shape[1]))
    true = data.sum(axis=0)
    nperms = 2**data.shape[0]
    for perm in itertools.product([-1, 1], repeat=data.shape[0]):
        # turn into numpy array
        perm = np.array(perm)
        # multiply permutation by subject dimension and sum over subjects
        randperm = (data * perm[:, None, None]).sum(axis=0)
        # compare to true difference (numpy autocasts bool to 0/1)
        out += (randperm > true)
    out = out / nperms
    # control for cases where pval is 1
    out[out == 1] = 1 - (1 / nperms)
    return out


def _pairedttest_random(data, nperms):
    '''Returns p-values based on nperms permutations of a paired ttest

    data is a (subj, alg, alg) matrix of differences between scores for each
    pair of algorithms per subject
    '''
    out = np.ones((data.shape[1], data.shape[1]))
    true = data.sum(axis=0)
    for i in range(nperms):
        perm = np.random.randint(2, size=(data.shape[0],))
        perm[perm == 0] = -1
        # multiply permutation by subject dimension and sum over subjects
        randperm = (data * perm[:, None, None]).sum(axis=0)
        # compare to true difference (numpy autocasts bool to 0/1)
        out += (randperm > true)
    out[out == nperms] = nperms - 1
    return out / nperms


def compute_pvals_perm(scores):
    '''Returns kxk matrix of p-values computed via permutation test,
    order defines the order of rows and columns

    df: DataFrame, samples are index, columns are pipelines, and values are
    scores

    order: list of length (num algorithms) with names corresponding to columns
    of df

    '''
    # reshape df into matrix (sub, k, k) of differences
    n_sub, n_algo = scores.shape[0], scores.shape[1]
    data = np.zeros((n_sub, n_algo, n_algo))
    for i in range(n_algo):
        for j in range(i + 1, n_algo):
            data[:, i, j] = scores[:, i] - scores[:, j]
            data[:, j, i] = scores[:, j] - scores[:, i]
    if n_sub > 13:
        p = _pairedttest_random(data, 10000)
    else:
        p = _pairedttest_exhaustive(data)
    return p

def compute_pvals(scores, perm_cutoff=20):
    if len(scores) < perm_cutoff:
        p = compute_pvals_perm(scores)
    else:
        p = compute_pvals_wilcoxon(scores)
    return p

def show_p(all_scores, bonferroni_correcton=False):
    Ps, n_subs = [], []
    for scores in all_scores:
        scores = np.array(scores).T
        p = compute_pvals(scores)
        Ps.append(p)
        n_sub = len(scores)
        n_subs.append(n_sub)

    weights = np.sqrt(np.array(n_subs))

    Ps = np.array(Ps)

    P = np.zeros((Ps.shape[1], Ps.shape[1]))
    for i in range(Ps.shape[1]):
        for j in range(Ps.shape[1]):
            P[i, j] = scipy.stats.combine_pvalues(Ps[:, i, j], weights=weights, method='stouffer')[1]
    ind = np.diag_indices(Ps.shape[1], Ps.shape[1])
    P[ind[0], ind[1]] = np.NaN
    pval1, pval2, pval3 = 5e-2, 1e-2, 1e-3
    if bonferroni_correcton:
        n = len(P)*(len(P)-1)/2
        pval1 = pval1/n
        pval2 = pval2/n
        pval3 = pval3/n
    dataP = np.copy(P)
    dataP[dataP>pval1] = np.NaN
    P[P>=pval1] = np.NaN
    P[np.logical_and(P>=pval2, P<pval1)] = 0.2
    P[np.logical_and(P>=pval3, P<pval2)] = 0.5
    P[P<pval3] = 1
    return P, dataP

def get_ssvep_data(
        dataset, srate, channels, duration, events, 
        delay=0.14, 
        raw_hook=None, 
        epochs_hook=None, 
        data_hook=None):
    start_pnt = dataset.events[events[0]][1][0]
    paradigm = SSVEP(
        srate=srate, 
        channels=channels, 
        intervals=[(start_pnt+delay, start_pnt+delay+duration)], 
        events=events)
    if raw_hook:
        paradigm.register_raw_hook(raw_hook)
    if epochs_hook:
        paradigm.register_epochs_hook(epochs_hook)
    if data_hook:
        paradigm.register_data_hook(data_hook)

    X, y, meta = paradigm.get_data(
        dataset, 
        subjects=dataset.subjects,
        return_concat=True,
        n_jobs=-1,
        verbose=False
    )
    return X, y, meta