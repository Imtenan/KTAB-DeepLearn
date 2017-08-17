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
# Synthesize data, for KTAB estimation using Central Position Theorem.
# Because this just synthesizes structured data, there is no learning (yet)
# ---------------------------------------------

import sys 
import datetime

import os

#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 

import klib as kl

import csv 
from tensorflow.python.framework import ops

# ---------------------------------------------
#%%
sys.stdout.flush()
stime = datetime.datetime.now()
print ("Python run start time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
sys.stdout.flush()


prng_seed = 380139612 # reproducible
prng_seed = 0         # irreproducible
kl.set_prngs(prng_seed)

cwd = os.getcwd()
print('cwd: ' + str(cwd))

np.set_printoptions(precision=4) # avoid printing  -0.28999999 from np.round

# ---------------------------------------------
#%%

# set the dimensions of the synthetic data and the KTAB model using it.
# Note that run time is about NB * BS^2, so more small batches is better.
# (Q: Why not BS=1? A: practice with arrays)
num_batch = 200
batch_size = 500

seed_dim = 8 # dimensionality of the manifold (4 is min to hit all choices, 7 is suggested)
num_survey_cat1 = 3  # first categorical has this many values
num_survey_cat2 = 4  # second categorical has this many values
num_survey_float = 6 # this many floating point values (less than seed_dim is suggested)

# characterization of the KTAB-like choice
num_weight = 4 # 4 is suggested
num_choice = 7 # 7 is suggested
    
show_matrices = (batch_size < 10)
# ---------------------------------------------
#%%
# set the numeric values (fixed for a run) of the generation matrices


# these three matrices determine what the Learner will see, determined entirely by the seed.
print("")
survey_fval_matrix = kl.uniform_tensor([num_survey_float, seed_dim])
print("survey_fval_matrix shape:  %s" % (str(survey_fval_matrix.shape)))

print("")
# map seed into category choices
cat1_matrix = kl.uniform_tensor([num_survey_cat1, seed_dim])
cat2_matrix = kl.uniform_tensor([num_survey_cat2, seed_dim])
print("cat1_matrix shape:  %s" % (str(cat1_matrix.shape)))
print("cat2_matrix shape:  %s" % (str(cat2_matrix.shape)))


# These matrices determine the KTAB model, determined entirely by the seed.
print("")
# map category choices into utility matrices.
#
# In constructing this synthetic data, the categorical variables
# give clues as to what kind of values the respondants place on
# the choices. Specifically, we generate several random matrices
# to map the seed into utilities, and the categorical variables
# select which matrix is used.
# NOTICE that they will be not merely selected but averaged,
# where the selected one gets more weight but all are included
cat1_util_matrix = kl.uniform_tensor([num_weight, num_choice, num_survey_cat1])
cat2_util_matrix = kl.uniform_tensor([num_weight, num_choice, num_survey_cat2])
print("cat1_util_matrix shape:  %s" % (str(cat1_util_matrix.shape)))
print("cat2_util_matrix shape:  %s" % (str(cat2_util_matrix.shape)))

# we will average the util-matrices from each category, as well
# as a util-matrix driven by the seed
srvy_util_matrix = kl.uniform_tensor([num_weight, num_choice, seed_dim])
print("srvy_util_matrix shape:  %s" % (str(srvy_util_matrix.shape)))


print("")
# calculate the vector of "raw-weights" from seed.
# We will take sqrt(abs(rw)) to get final weights
raw_weight_matrix = kl.uniform_tensor([num_weight, seed_dim])
print("\nRaw weight from seed matrix %s" % (str(raw_weight_matrix.shape)))
if show_matrices:
    print(np.round(raw_weight_matrix, 2))

# ---------------------------------------------
#%%

# Preceding code illustrates matrix/tensor manipulations in plain NumPy.
# Following code illustrates matrix/tensor manipulations in TensorFlow.

# At this point, we start to define the computational graph.
ops.reset_default_graph()
sess = tf.Session()
# ---------------------------------------------
#%%
# because of the funny way I've chosen to map seeds into discrete categories,
# then categories into normalized category-weights, it is easiest to consider
# both seed and category weights as placeholders (i.e. inputs)

print("\nBatch_size: %u\n" % batch_size)

print("Inputs (placeholder):")
seed_tensor = tf.placeholder(shape=[seed_dim, batch_size], dtype = tf.float32)
print("seed_tensor.shape: %s " % (seed_tensor.shape))

cat1_wghts = tf.placeholder(shape=[num_survey_cat1, batch_size], dtype = tf.float32)
print("cat1_wghts.shape: %s " % (cat1_wghts.shape))

cat2_wghts = tf.placeholder(shape=[num_survey_cat2, batch_size], dtype = tf.float32)
print("cat2_wghts.shape: %s " % (cat2_wghts.shape))

# ---------------------------------------------
print()
print("Structure nodes (constant):")
cat1_tensor = tf.constant(cat1_util_matrix, dtype = tf.float32)
print("cat1_tensor.shape: %s " % (cat1_tensor.shape))
cat2_tensor = tf.constant(cat2_util_matrix, dtype = tf.float32)
print("cat2_tensor.shape: %s " % (cat2_tensor.shape))
survey_fval_tensor = tf.constant(survey_fval_matrix, dtype = tf.float32)
print("survey_fval_tensor.shape: %s " % (survey_fval_tensor.shape))

survey_util_tensor = tf.constant(srvy_util_matrix, dtype = tf.float32)
print("survey_util_tensor.shape: %s " % (survey_util_tensor.shape))

raw_weight_tensor = tf.constant(raw_weight_matrix, dtype = tf.float32)
print("raw_weight_tensor.shape: %s " % (raw_weight_tensor.shape))

# ---------------------------------------------
print()
print("Interior nodes:")
srvy_rspns_t = tf.transpose(tf.tensordot(survey_fval_tensor, seed_tensor, axes=[[1], [0]]))
print("srvy_rspns_t.shape: %s " % (srvy_rspns_t.shape))
cat1_util = tf.tensordot(cat1_tensor, cat1_wghts, axes=[[2], [0]])
print("cat1_util.shape: %s " % (cat1_util.shape))
cat2_util = tf.tensordot(cat2_tensor, cat2_wghts, axes=[[2], [0]])
print("cat2_util.shape: %s " % (cat2_util.shape))


srvy_util = tf.tensordot(survey_util_tensor, seed_tensor, axes=[[2], [0]])
print("srvy_util.shape: %s " % (srvy_util.shape))

alpha = 2.0
fnl_util = (alpha*srvy_util + cat1_util + cat2_util)/(alpha+1.0+1.0)
print("fnl_util.shape: %s " % (fnl_util.shape))

# actor (interest) weights are just the sqrt of the abs of a matrix multiply
raw_weight = tf.tensordot(raw_weight_tensor, seed_tensor, axes=[[1], [0]])
print("raw_weight.shape: %s " % (raw_weight.shape))

# ---------------------------------------------
#%%

print("\nEach weight should be a row-vector")
fnl_weight = tf.transpose(tf.sqrt(tf.abs(raw_weight))) 
print("fnl_weight.shape: %s " % (fnl_weight.shape))
print("fnl_util.shape: %s " % (fnl_util.shape))

# just a node in the graph, unevaluated as yet
zeta_t = tf.tensordot(fnl_weight, fnl_util, axes=[[1], [0]])
print("Before contraction-by-selection, shape of zeta_t: %s" % (zeta_t.shape))
print("Zeta needs to be %s by %s" % (batch_size, num_choice))

# NOTE the inefficiency: It calculates about B^2 terms, when we only want B.
# Hence, it is better to calculate 100 batches of 100 each
# than 1 batch of 10,000
# ---------------------------------------------
#%%

# This has been verified to pick out the correct indices,
# so it does the (B, N, B) -> (B, N) contraction-by-selection
# correctly (not contraction-by-summation).
#print()
#print("These are the indices which need to be used")
# a tensor of shape (B, M, B) includes all the mis-matches
# where weight-vector(i) was used with util-matrix(j)

slice_ndx = tf.constant([[i,j,k] for i in range(batch_size)
                                    for j in range(num_choice)
                                        for k in range(batch_size) if i==k])


#foo = sess.run(slice_ndx)
#for f in foo:
#    print(f)

# select the large zeta-tensor into a plain vector
zeta_v = tf.gather_nd(zeta_t, slice_ndx)

# reshape that vector into  the desired matrices
zeta = tf.reshape(zeta_v, [batch_size, num_choice])
print("After contraction-by-selection, shape of Zeta: %s" % (zeta.shape))

# At this point, we have finished the computational graph.

# ---------------------------------------------
#%%
# Note that we treat the seed as an (N x 1) column vector,
# so that a batch of them is an (N x B) matrix.
# However, the responses are all printed as row-vectors, for ease of reading.
print("")
st = kl.uniform_tensor([seed_dim, batch_size], -0.25, +1.0)
if show_matrices:
    print("st (seed_dim x batch): \n%s" % st) 

# num_survey_cat1 x batch_size
cv1 = np.matmul(cat1_matrix, st)
cv1_rspns = kl.weighted_max_ndx(cv1, 0.0, False)
cv1_weights = kl.weighted_max_ndx(cv1_rspns, 1.0, True)

if show_matrices:
    print("cv1 (cat x batch): \n%s" % cv1)
    print("cv1_rspns (cat x batch): %s \n%s" % (cv1_rspns.shape, cv1_rspns))
    print("cv1_weights (cat x batch): %s \n%s" % (cv1_weights.shape, cv1_weights))
else:
    print("cv1_rspns (cat x batch): %s" % str(cv1_rspns.shape))
    print("cv1_weights (cat x batch): %s" % str(cv1_weights.shape))

# num_survey_cat2 x batch_size
cv2 = np.matmul(cat2_matrix, st)
cv2_rspns = kl.weighted_max_ndx(cv2, 0.0, False)
cv2_weights = kl.weighted_max_ndx(cv2_rspns, 7.0/5.0, True)

if show_matrices:
    print("cv2 (cat x batch): \n%s" % cv2)
    print("cv2_rspns (cat x batch): %s \n%s" % (cv2_rspns.shape, cv2_rspns))
    print("cv2_weights (cat x batch): %s \n%s" % (cv2_weights.shape, cv2_weights))
else:
    print("cv2_rspns (cat x batch): %s" % str(cv2_rspns.shape))
    print("cv2_weights (cat x batch): %s" % str(cv2_weights.shape))


# ---------------------------------------------
#%%
print()


# calculate survey response numbers:
batch_dict = {seed_tensor:st, cat1_wghts:cv1_weights, cat2_wghts:cv2_weights }
srvy_rspns = sess.run(srvy_rspns_t, feed_dict = batch_dict)
zeta_val = sess.run(zeta, feed_dict = batch_dict)
vhcl_rspns = np.transpose(kl.weighted_max_ndx (np.transpose(zeta_val), 0.0, False))

if show_matrices:
    print("Survey (floating) responses (field x batch): \n%s" % srvy_rspns)
    print("Zeta (batch x choice): \n%s" % zeta_val)
    print("Vehicle choices (batch x choice): \n%s" % vhcl_rspns)
else:
    print("Survey (floating) responses (field x batch): %s" % str(srvy_rspns.shape))
    print("Zeta (batch x choice): %s" % str(zeta_val.shape))
    print("Vehicle choices (batch x choice): %s" % str(vhcl_rspns.shape))




cv1_rspns = np.transpose(cv1_rspns)
cv2_rspns = np.transpose(cv2_rspns)
# ---------------------------------------------
#%%

# ---------------------------------------------
#%%

# At this point, we have the complete survey results, driven by the seeds
print()
print("CV1 response shape: %s" % str(cv1_rspns.shape))
print("CV2 response shape: %s" % str(cv2_rspns.shape)) 
print("Survey response shape: %s" % str(srvy_rspns.shape))
print("Vehicle choice shape: %s" % str(vhcl_rspns.shape))

print_freq = 10

def write_one_survey_record(survey_cat1, survey_cat2, survey_floats, vhcl_choice, my_csv_writer): 
    f1 = list(survey_cat1) 
    f2 = list(survey_cat2) 
    f3 = list(survey_floats)
    f4 = list(vhcl_choice)
    my_csv_writer.writerow(f1+f3+f2+f4)
    return

with open('tmp_survey2.csv', 'w', newline='') as csvfile:
    my_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL) 
    param_list = list(np.array([num_batch*batch_size, num_survey_cat1, num_survey_float, num_survey_cat2, num_choice]))
    my_writer.writerow(param_list)
    for n in range(num_batch):
        if (0 == (n+1)%print_freq):
            print("Batch %u/%u " % (n+1, num_batch))
            
        st = kl.uniform_tensor([seed_dim, batch_size], -0.25, +1.0) 

        cv1 = np.matmul(cat1_matrix, st)
        cv1_rspns = kl.weighted_max_ndx(cv1, 0.0, False)
        cv1_weights = kl.weighted_max_ndx(cv1_rspns, 1.0, True)

        cv2 = np.matmul(cat2_matrix, st)
        cv2_rspns = kl.weighted_max_ndx(cv2, 0.0, False)
        cv2_weights = kl.weighted_max_ndx(cv2_rspns, 7.0/5.0, True)

        batch_dict = {seed_tensor:st, cat1_wghts:cv1_weights, cat2_wghts:cv2_weights }
        srvy_rspns = sess.run(srvy_rspns_t, feed_dict = batch_dict)
        zeta_val = sess.run(zeta, feed_dict = batch_dict)
        vhcl_rspns = np.transpose(kl.weighted_max_ndx (np.transpose(zeta_val), 0.0, False))

        cv1_rspns = np.transpose(cv1_rspns)
        cv2_rspns = np.transpose(cv2_rspns)
        # write out the batch
        for i in range(batch_size): 
            cv1 = cv1_rspns[i,]
            cv2 = cv2_rspns[i,]
            srvy = srvy_rspns[i,]
            vhcl = vhcl_rspns[i,]
            write_one_survey_record( cv1, cv2, srvy, vhcl, my_writer)
    
# ---------------------------------------------
#%%
sys.stdout.flush()
print()
etime = datetime.datetime.now()
dtime = etime - stime
print ("Python run end time: " + etime.strftime("%Y-%m-%d %H:%M:%S") )
print ("Python elapsed time: " + str(dtime))
sys.stdout.flush()

# ---------------------------------------------
# Copyright KAPSARC. Open source MIT License.
# ---------------------------------------------
