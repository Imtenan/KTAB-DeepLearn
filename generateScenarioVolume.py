#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------------------------------
# Copyright KAPSARC. Open source MIT License.
# ---------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2015 King Abdullah Petroleum Studies and Research Center
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom
# the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ---------------------------------------------
#
# Setup volume of scenarios for hyperparameter tuning search for survey_KT.py.
# Hyperparameters varied include number CDMP actors, learning rate, learning rate
# decay, minibatch size, number hidden layers, hidden layer widths, optimization
# method, RMSProp momentum, dropout keep %, and regularization rate. Hyperparameter
# priorities are:
# - learning_rate,
# - momentum 1 (only RMS Prop)+minibatch size+# hidden units
# - learning rate decay + # hidden layers
#
# ---------------------------------------------

import numpy as np
import numpy.random as rnd
#import itertools
import os
import klib as kl
import string

# get user input for seed & granularity
prng_seed = int(input("Enter scenario generator prng seed (421109) <ENTER>:"))
granu = int(input("Enter number scenarios to generate and hit <ENTER>:"))
num_cdmp_actors = 4*int(input("Enter whether or not to use the CDMP and hit <ENTER>:"))
rndSeeds = int(input("Enter whether or not to use a different seed for each scenario and hit <ENTER>:"))

prng_seed = kl.set_prngs(prng_seed)

# define lambda function for generating scenario 'names'
getName = lambda N: ''.join(rnd.choice(list(string.ascii_uppercase+string.digits),N))

# get random seeds, maybe
if rndSeeds:
  seed_min = 100000000
  seed_max = 999999999
  prng_seeds = rnd.randint(seed_min,seed_max+1,granu)
else:
  prng_seeds = [prng_seed]*granu

''' Random / Varying Values '''
# random uniform selection of CDMP being used or not
#num_cdmp_actors = (rnd.rand(granu)>0.5)*4

# random log-scale search between 0.0001 and 1.0 for learning rate
learn_rate = 10**(-4*rnd.rand(granu))

# random log-scale search between 0.00001 and 0.01 for learning rate decay
tmp = rnd.uniform(-5,-1.000001,granu)
tmp[tmp>-1]=-1
learn_rate_decay = 10**tmp

# random uniform selection of minibatch size
batch_size = rnd.randint(50,101,granu)

# random uniform selection of hidden layer width & hidden layer counts
hiddenLayerWs = rnd.uniform(0.2,1.5000001,granu) # all but first layer uniformly selected
hiddenLayers = rnd.randint(2,11,granu)
# hardcode first layer as 1.5
hiddenLayerWs = [[1.5]+[w]*l for w,l in zip(hiddenLayerWs,hiddenLayers-1)]

# random uniform selection of optimizer
optimMeths = ['GD','RMSPROP','ADAM']
optimMeth = rnd.randint(0,len(optimMeths),granu)

# generate momentum values based on the optimMeth
optimBeta1 = np.repeat(0.9,granu) # just use 0.9 for GD (no momentum) and ADAM
# random log-scale search between 0.9 and 0.999 for momentum
makeBetas = (optimMeth==1)
tmp = rnd.uniform(-3,-1.000001,sum(makeBetas))
tmp[tmp > -1] = -1
optimBeta1[makeBetas] = 1-10**tmp

# random uniform from three specific values of the dropoutKeep
dropoutKeep = rnd.choice([0.8,0.9,1.0],granu)

# random uniform from three specific values of the regularization rate
regulRate = rnd.choice([0.1, 0.5, 1.0],granu)

''' Static Values '''
optimBeta2 = 0.99
trainPerc = 0.95
devePerc = 0.05
epoch = [2000,1000,20,0.0001]

''' Put it all together and write the model input file '''
## create grid search object
#grid = itertools.product(batch_size,learn_rate,learn_rate_decay,regulRate,\
#                         hiddenLayers,hiddenLayerWs,optim,dropoutKeep)
#gridLen = len(batch_size)*len(learn_rate)*len(learn_rate_decay)*\
#  len(regulRate)*len(hiddenLayers)*len(hiddenLayerWs)*len(optim)*\
#  len(dropoutKeep)

# write model input file(s)
modelInput = os.getcwd()+os.sep+'models_%u_%u_%u.txt'%(num_cdmp_actors,prng_seed,granu)
f = open(modelInput,'wt')

# write the stuff that doesn't change
f.write('# tuning model input: %u\n'%prng_seed)
f.write('%u\n'%granu)

# write the stuff that changes per scenario
for i in range(granu):
  print('Writing scenario %u of %u'%(i+1,granu))
  f.write('%s\n'%getName(10)) # scenario 'name'
  f.write('%u\n'%prng_seeds[i]) # seed
  f.write('%u\n'%num_cdmp_actors) # number CDMP actors
  f.write('%u %u %u %0.4f\n'%(epoch[0],epoch[1],epoch[2],epoch[3])) # epoch info
  f.write('%0.4f %0.4f\n'%(trainPerc,devePerc)) # training/development split
  f.write('%u\n'%batch_size[i]) # minibatch size
  f.write('%0.4f %0.4f\n'%(learn_rate[i],learn_rate_decay[i])) # learning rate & its decay
  f.write('%0.4f\n'%regulRate[i]) # regularlization rate
  f.write('%s\n'%(' '.join(map(str,hiddenLayerWs[i])))) # list of hidden layer widths
  f.write('%s %0.4f %0.4f\n'%(optimMeths[optimMeth[i]],optimBeta1[i],optimBeta2)) # optimizer & momentums
  f.write('%0.4f\n'%dropoutKeep[i]) # dropout keep3
  i+=1
f.close()
print('Scenario Volume written to '+ modelInput)

# ---------------------------------------------
# Copyright KAPSARC. Open source MIT License.
# ---------------------------------------------
