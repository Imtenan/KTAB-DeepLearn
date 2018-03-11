#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------------------------------
# Copyright KAPSARC. Open Source MIT License.
# ---------------------------------------------
#
#
#
# ---------------------------------------------
#import random
import numpy as np 
import tensorflow as tf

# ---------------------------------------------
#%%
def set_prngs (s = 0):
    prng_seed = s
    if (0 == s):
        #random.seed() # irreproducible 
        np.random.seed()
        seed_min = 100000000.0
        seed_max = 999999999.9
        #prng_seed = round(random.uniform(seed_min, seed_max))
        prng_seed = np.random.randint(seed_min,seed_max+1)
    
    #prng_seed = 615904 # reproducible 
    
    print('prng_seed: '+str(prng_seed)) 
    # Need to set all three PRNGs
    #random.seed(prng_seed)
    tf.set_random_seed(prng_seed)
    np.random.seed(prng_seed)
    return prng_seed

# ---------------------------------------------
#%%
def smooth(ema, x, t, initP): # classic exponential moving average
    if initP:
        s = x
    else:
        s = (t*x) + ((1.0-t)*ema)
    return s

# Out of N alternatives, is this better than random?
# The 0.435 factor makes it indicate number of decimals when N=2:
# LO(0.9)   = 1,
# LO(0.99)  = 2,
# LO(0.999) = 3,
# etc.
def log_odds(p, n=2.0):
    pMin = 0.00000001
    pMax = 0.99999999
    if (p < pMin):
        p = pMin
    if (pMax < p):
        p = pMax
    odds = ((n-1.0)*p)/(1.0 - p)
    low = np.log(odds)  * 0.435244
    return low

# ---------------------------------------------
#%%
# fill it with U[vmin, vmax] values
def uniform_tensor (shape, vmin = -1.0, vmax = +1.0):
    num_el = np.product(shape)
    u_vals = 2.0*(np.random.uniform(0.0, 1.0, num_el) - 0.5)
    ut = np.reshape(u_vals, shape)
    return ut


def weighted_max_row_ndx (vv1, offset, norm):
    dims = vv1.shape
    vdim = dims[0]
    ndx = np.argmax(vv1) 
    vv2 = np.zeros(vdim)
    vv2[ndx]=1.0
    vv2 = vv2 + offset
    if norm:
        vv2 = vv2/np.sum(vv2)
    return vv2

# Return a vector of the 'offset', but with the largest position incremented by 1
# Each column is processed separately
def weighted_max_ndx (vm1, offset=0, norm=True):
    fn = lambda clm : weighted_max_row_ndx(clm, offset, norm)
    vm2 = np.apply_along_axis(fn, 0, vm1)
    return vm2


# ---------------------------------------------
#%%
    
def predFromProb(classProbs):
  '''
  Take in as input an (m x p) matrix of m observations' class probabilities in
  p classes and return an m-length vector of integer class labels (0...p-1). 
  Probabilities at or below 1/p are set to 0, as are NaNs; any unclassed
  observations are randomly assigned to a class.
  '''
  numClasses = classProbs.shape[1]
  # zero out class probs that are at or below chance, or NaN
  probs = classProbs.copy()
  probs[np.isnan(probs)] = 0
  probs = probs*(probs > 1/numClasses)
  # find any un-classed observations
  unpred = ~np.any(probs,axis=1)
  # get the predicted classes
  preds = np.argmax(probs,axis=1)
  # randomly classify un-classed observations
  rnds = np.random.randint(0,numClasses,np.sum(unpred))
  preds[unpred] = rnds
  
  return preds

# ---------------------------------------------
# Copyright KAPSARC. Open Source MIT License.
# ---------------------------------------------