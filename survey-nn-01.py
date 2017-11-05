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
# Neural net to learn one-hot encoding,
# one hidden layer
# LO test: 1.0401 +/- 0.0034
# about 23 stdev above survey-nn-00.
#
# ---------------------------------------------

import random
import sys 
import datetime

import csv

import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit


import klib as kl

from tensorflow.python.framework import ops

# ---------------------------------------------
#%%
sys.stdout.flush()
stime = datetime.datetime.now()
print ("Python run start time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
sys.stdout.flush()
sys.stdout.flush()

#prng_seed = 831931061 # reproducible
#prng_seed = 0         # irreproducible
prng_seed = 10241227
kl.set_prngs(prng_seed)

ops.reset_default_graph()
sess = tf.Session()

# ---------------------------------------------
#%%


# start recording for TensorBoard
log_file_name = 'log-'+str(prng_seed)
writer = tf.summary.FileWriter('log/'+log_file_name, sess.graph) 


csv_file_name = 'tmp_survey2.csv'
csvfile = open(csv_file_name, newline='')
print('opened ' + csv_file_name)

csv_data = []
csv_data_obj = csv.reader(csvfile, delimiter=',', quotechar='|')
for row in csv_data_obj:
    #print(' , '.join(row))
    csv_data.append(row)


# ---------------------------------------------
#%%
# The data file format will have to be changed to (num_rows, num_data, num_choice)
print('csv length: ' + str(len(csv_data)))
survey_headers = [int(x) for x in csv_data[0]] # Num rows, cat1, float, cat2, choice

num_rows = survey_headers[0]
print("expected num_rows: %u" % (num_rows))

num_cat1 = survey_headers[1]
print("expected num_cat1: %u" % (num_cat1))

num_float = survey_headers[2]
print("expected num_float: %u" % (num_float))

num_cat2 = survey_headers[3]
print("expected num_cat2: %u" % (num_cat2))

print()
num_data_col = num_cat1 + num_float + num_cat2
print("expected num_data_col: %u" % (num_data_col))

num_choice_col = int(survey_headers[4])
print("expected num_choice_col: %u" % (num_choice_col))

survey_list = csv_data[1:] # list of lists
data_len = len(survey_list)
print('survey data length: ' + str(data_len))

survey_data = [ [float(x) for x in y] for y in survey_list ]

print()
x_data = np.array([x[0:num_data_col] for x in survey_data]) # [0,6) == [0,5]
    
y_data = np.array([y[num_data_col:num_data_col+num_choice_col] for y in survey_data])
x_headers = [h for h in survey_headers[1:6]]

print('x-shape: ' + str(x_data.shape))
print('y-shape: ' + str(y_data.shape)) 

# ---------------------------------------------
#%%
# use sklearn to perform stratified randomized partitioning into training and dev sets
# this is necessary because the vehicle choice dataset is very unbalanced
trainPerc = 0.95 # deep learning uses much higher %'s for training
sss = StratifiedShuffleSplit(n_splits=1, train_size=trainPerc)
train_indices,test_indices = next(sss.split(x_data, y_data))
# create the patitions
x_vals_train = x_data[train_indices]
y_vals_train = y_data[train_indices]

x_vals_test = x_data[test_indices]
y_vals_test = y_data[test_indices]

print("num_train_rows: %u, num_test_rows: %u" %(len(train_indices), len(test_indices)))

# ---------------------------------------------
#%%
# setup training
a_stdv = 0.1
#b_stdv = 0.0 # no longer needed, since bs are now init'd to 0
learn_rate = 1.0
beta = 0.0025 #0.001


layer_0_width = num_data_col    # always number of inputs
layer_1_width = int(2.0*num_choice_col)
layer_2_width = num_choice_col


batch_size = 100
print("batch_size: %u" %  batch_size)

# ---------------------------------------------
#%%
# number of rows could be batch_size, num_rows_train, or num_rows_test.
# So we mark it as None, which really means variable-size
x_data = tf.placeholder(shape=[None, num_data_col], dtype=tf.float32)
y_trgt = tf.placeholder(shape=[None, num_choice_col], dtype=tf.float32)

print("x_data shape: %s" % str(x_data.shape))
print("y_trgt shape: %s" % str(y_trgt.shape))

# ---------------------------------------------
#%%
A1 = tf.Variable(tf.random_normal(shape=[layer_0_width, layer_1_width],  mean=0.0, stddev=a_stdv))
print("A1 shape: %s" % str(A1.shape))
b1 = tf.Variable(tf.zeros(shape=[layer_1_width])) # a 0-rank array?
print("b1 shape: %s" % str(b1.shape))
out1 =  tf.nn.relu(tf.add(tf.matmul(x_data,A1), b1))
print("out1 shape: %s" % str(out1.shape))

# ---------------------------------------------
#%%
A2 = tf.Variable(tf.random_normal(shape=[layer_1_width, layer_2_width],  mean=0.0, stddev=a_stdv))
print("A2 shape: %s" % str(A2.shape))
b2 = tf.Variable(tf.zeros(shape=[layer_2l_width])) # a 0-rank array?
print("b2 shape: %s" % str(b2.shape))
out2 = tf.add(tf.matmul(out1, A2), b2)
print("out2 shape: %s" % str(out2.shape))
 
# ---------------------------------------------
#%%
# note that 'labels' are real numbers in [0,1] interval,
# logits are in (-inf, +inf)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_trgt, logits = out2))
init = tf.global_variables_initializer()
sess.run(init)


regularizer =  tf.add(tf.nn.l2_loss(A1), tf.nn.l2_loss(A2))

my_opt = tf.train.GradientDescentOptimizer(learn_rate)
train_step = my_opt.minimize(loss + beta*regularizer )

print()
# ---------------------------------------------
#%%
# record prediction accuracy as well as cross-entropy loss

prediction = tf.round(tf.sigmoid(out2)) # obviously round to {0,1}
predictions_correct = tf.cast(tf.equal(prediction, y_trgt), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

# ---------------------------------------------
#%%

theta = 0.010 # weight on new value 

loss_ma = 0.0
train_ma = 0.0
test_ma = 0.0

loss_vec = []
train_vec = []
test_vec = []
test_acc = []

num_steps =  5000
print_freq = round(num_steps / 10.0)

xy_test_dict = {x_data:x_vals_test, y_trgt:y_vals_test}

for i in range(num_steps+1):
    #rand_indices = random.sample(range(num_train_rows), batch_size)
    #rand_x = x_vals_train[rand_indices]
    #rand_y = y_vals_train[rand_indices]
    
    #print("rand_x shape: %s" % str(rand_x.shape))
    #print("rand_y shape: %s" % str(rand_y.shape))
    
    
    #print("x_data shape: %s" % str(x_data.shape))
    #print("y_trgt shape: %s" % str(y_trgt.shape))
    
    # train on this batch, and record loss on it
    #xy_dict = {x_data:rand_x, y_trgt:rand_y}
    
    # Note that the first report is after the first training step
    sess.run(train_step, feed_dict=xy_dict)
    
    tmp_loss = sess.run(loss, feed_dict=xy_dict)
    loss_ma = kl.smooth(loss_ma, tmp_loss, theta, (0 == i))
    loss_vec.append(loss_ma)
    
    # record accuracy over current training set
    tmp_acc_train = sess.run(accuracy, feed_dict=xy_dict)
    tmp_acc_train = kl.log_odds(tmp_acc_train) 
    train_ma = kl.smooth(train_ma, tmp_acc_train, theta, (0 == i))
    train_vec.append(train_ma)
    
    # record accuracy over the etire test set
    test_acc.append(sess.run(accuracy,feed_dict = xy_test_dict))
    test_v = kl.log_odds(test_acc[-1])
    test_ma = kl.smooth(test_ma, test_v, theta, (0 == i))
    test_vec.append(test_ma)
    
    # record accuracy over random part of test set
#    num_tmp_test = int(num_test_rows/100.0)
#    rand_indices = random.sample(range(num_test_rows), num_tmp_test)
#    rand_x = x_vals_test[rand_indices]
#    rand_y = y_vals_test[rand_indices]
#    tmp_acc_test = sess.run(accuracy, feed_dict={x_data:rand_x, y_trgt:rand_y})
#    tmp_acc_test = kl.log_odds(tmp_acc_test)
#    test_ma = kl.smooth(test_ma, tmp_acc_test, theta, (0 == i))
#    test_vec.append(test_ma)
    
    if (0 == i % print_freq): 
        print("Smoothed step %u/%u, train batch loss: %.4f, train batch acc: %.4f, Test acc: %.4f" % 
              (i, num_steps, loss_ma, train_ma, test_ma))
        
    #sess.run(train_step, feed_dict=xy_dict)

# ---------------------------------------------
#%%
# Now that all processing is finished, try to save the results
# https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
# https://www.tensorflow.org/programmers_guide/variables
# https://www.tensorflow.org/versions/r0.12/api_docs/python/state_ops/saving_and_restoring_variables

# The saver object will save all the variables
saver = tf.train.Saver()

# Actually save the graph, tagging the file names with the step number.
# This saves in several files, e.g.
#    logit_model-2500.data-00000-of-00001
#    logit_model-2500.index
#    logit_model-2500.meta

saver.save(sess, 'log/'+log_file_name, global_step=num_steps)
# ---------------------------------------------
#%%
# Display performance plots

plt.plot(loss_vec, 'k-')
plt.title('Cross Entropy Loss vs Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.show()

#print('Final learned parameters')
#print ('A on ' + str(x_headers) + ': \n' + str(sess.run(A)))
#print ('b: \n' + str(sess.run(b)))

plt.figure()
plt.plot(train_vec, 'b-', label='Train batch')
plt.plot(test_vec, 'r-', label='Test set')
plt.title('Log-Odds Accuracy vs Generation')
plt.xlabel('Generation')
plt.ylabel('Log-Odds Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.figure()
plt.plot(test_acc)
plt.title('Test Set Classification Accuracy')
print('Final Model Test Set Classification Accuracy = %0.2f%%'%(100*test_acc[-1]))

# ---------------------------------------------
#%%
# close out the session and save the event-files
writer.close()
# ---------------------------------------------
#%%
sys.stdout.flush()
print('')
etime = datetime.datetime.now()
dtime = etime - stime
print ("Python run end time: " + etime.strftime("%Y-%m-%d %H:%M:%S") )
print ("Python elapsed time: " + str(dtime))
sys.stdout.flush()

# ---------------------------------------------
# Copyright KAPSARC. Open source MIT License.
# ---------------------------------------------
