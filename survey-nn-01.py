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

import sys 
import csv
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.model_selection import StratifiedShuffleSplit
import datetime
import klib as kl

# ---------------------------------------------
#%%
sys.stdout.flush()
stime = datetime.datetime.now()
print ("Python run start time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
sys.stdout.flush()
sys.stdout.flush()

#prng_seed = 831931061 # reproducible
prng_seed = 0         # irreproducible
#prng_seed = 10241227
prng_seed = kl.set_prngs(prng_seed)

ops.reset_default_graph()
sess = tf.Session()

# ---------------------------------------------
#%%
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
trainPerc = 0.95; devePerc = 0.05 # deep learning uses much higher %'s for training
sss = StratifiedShuffleSplit(n_splits=1, train_size=trainPerc, test_size = devePerc)
train_indices,deve_indices = next(sss.split(x_data, y_data))
num_train_rows = len(train_indices) # need this later on
# create the patitions
x_vals_train = x_data[train_indices,:]
y_vals_train = y_data[train_indices,:]

x_vals_deve = x_data[deve_indices,:]
y_vals_deve = y_data[deve_indices,:]

print("num_train_rows: %u, num_deve_rows: %u" %(num_train_rows, len(deve_indices)))

# ---------------------------------------------
#%%
# setup training
a_stdv = 0.1          # standard dev. for initialization of node weights
learn_rate = 1.0      # gradient descent learning rate
lambd = 0.5           # rate of normalization of the loss function
batch_size = 100      # size of mini-batches used in training
num_epochs = 100      # number training iterations
hidden_layers = 1     # number hidden layers

layer_0_width = num_data_col    # always number of inputs
layer_1_width = int(2.0*num_choice_col)
layer_2_width = num_choice_col

with tf.name_scope('HyperParms'):
  lr = tf.constant(learn_rate)
  la = tf.constant(lambd)
  bs = tf.constant(batch_size)
  ne = tf.constant(num_epochs)
  dp = tf.constant(devePerc)
  hl = tf.constant(hidden_layers)
  # summaries
  parmSumm = tf.summary.merge([tf.summary.scalar('learn_rate', lr),\
            tf.summary.scalar('regul_rate', la), tf.summary.scalar('minibatch_size',bs),\
            tf.summary.scalar('epochs',ne), tf.summary.scalar('dev_set_size',dp),\
            tf.summary.scalar('hidden_layers',hl)])

# talk some
print('Hyperparameters\n\tlearning rate: %0.2f\n\tregularization rate: %0.2f\n\tminibatch size: %u'\
      %(learn_rate,lambd,batch_size))

# create the logfile prefix
log_file_name = 'log_NN%03d_%05d_%s_%d'%(hidden_layers,num_epochs,stime.strftime('%Y%m%d_%H%M%S'),prng_seed)

# ---------------------------------------------
#%%
# number of rows could be batch_size, num_rows_train, or num_rows_deve.
# So we mark it as None, which really means variable-size
with tf.name_scope('Input'):
  x_data = tf.placeholder(shape=[None, num_data_col], dtype=tf.float32)
  y_trgt = tf.placeholder(shape=[None, num_choice_col], dtype=tf.float32)
  yhat = tf.placeholder(shape=[None, num_choice_col], dtype=tf.float32)

print("x_data shape: %s" % str(x_data.shape))
print("y_trgt shape: %s" % str(y_trgt.shape))

# ---------------------------------------------
#%%
# FIRST HIDDEN layer
with tf.name_scope('Layer1'):
  A1 = tf.Variable(tf.random_normal(shape=[layer_0_width, layer_1_width],  mean=0.0, stddev=a_stdv))
  print("A1 shape: %s" % str(A1.shape))
  b1 = tf.Variable(tf.zeros(shape=[layer_1_width])) # a 0-rank array?
  print("b1 shape: %s" % str(b1.shape))
  out1 =  tf.nn.relu(tf.add(tf.matmul(x_data,A1), b1))
  print("out1 shape: %s" % str(out1.shape))

# ---------------------------------------------
#%%
# OUTPUT layer
with tf.name_scope('Layer2'):
  A2 = tf.Variable(tf.random_normal(shape=[layer_1_width, layer_2_width],  mean=0.0, stddev=a_stdv))
  print("A2 shape: %s" % str(A2.shape))
  b2 = tf.Variable(tf.zeros(shape=[layer_2_width])) # a 0-rank array?
  print("b2 shape: %s" % str(b2.shape))
  out2 = tf.add(tf.matmul(out1, A2), b2)
  print("out2 shape: %s" % str(out2.shape))
 
# ---------------------------------------------
#%%
# note that 'labels' are real numbers in [0,1] interval,
# logits are in (-inf, +inf)
with tf.name_scope('Loss'):
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_trgt, logits = out2))
  losSummary = tf.summary.scalar('Loss', loss)

with tf.name_scope('Train'):
  regularizer =  tf.add(tf.nn.l2_loss(A1), tf.nn.l2_loss(A2))
  my_opt = tf.train.GradientDescentOptimizer(learn_rate)
  train_step = my_opt.minimize(loss + lambd*regularizer/(2*batch_size))

# ---------------------------------------------
#%%
# record standard classification performance metrics
with tf.name_scope('Eval'):
  _,accuracy = tf.metrics.accuracy(y_trgt,yhat)
  _,precision = tf.metrics.precision(y_trgt,yhat)
  _,recall = tf.metrics.recall(y_trgt,yhat)
  F1 = 2*tf.divide(tf.multiply(precision,recall),tf.add(precision,recall))
  # define the writer summaries
  accSummary = tf.summary.scalar('Accuracy', accuracy)
  preSummary = tf.summary.scalar('Precision', precision)
  recSummary = tf.summary.scalar('Recall', recall)
  f1sSummary = tf.summary.scalar('F1',F1)
  perfSumm = tf.summary.merge([accSummary,preSummary,recSummary,f1sSummary])

# ---------------------------------------------
#%%
# finally perform the training simulation
# setup for results & model serialization
writer = tf.summary.FileWriter('log/'+log_file_name, sess.graph)
writer.add_summary(parmSumm.eval(session=sess))
devWriter = tf.summary.FileWriter('log/dev/'+log_file_name,sess.graph)
devWriter.add_summary(parmSumm.eval(session=sess))

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
# The saver object will save all the variables
saver = tf.train.Saver()

# randomly generate the minibatches all at once
batchIndxs = np.random.randint(0,num_train_rows,(num_epochs,batch_size))

# set up for moving averages
theta = 0.010 # weight on new value  - only for printing/plotting MAs
loss_ma = 0.0; train_ma = [0.0]*4 # accuracy, precision, recall, F1

loss_vec = [0.0]*num_epochs
train_vec = [[0.0]*4 for _ in range(num_epochs)] # will record accuracy, precision, recall, F1

print_freq = num_epochs//10   # control console print & checkpoint frequency
save_freq = num_epochs//5     # control tensorboard save frequency
for i in range(num_epochs):
  # get this minibatch
  rand_x = x_vals_train[batchIndxs[i],:]
  rand_y = y_vals_train[batchIndxs[i],:]
  # train on this batch, and record loss on it
  xy_dict = {x_data:rand_x, y_trgt:rand_y}
    
  # Note that the first report is after the first training step
  sess.run(train_step, feed_dict=xy_dict)
  
  tmp_loss = sess.run(loss, feed_dict=xy_dict)
  loss_ma = kl.smooth(loss_ma, tmp_loss, theta, (0 == i))
  loss_vec[i]=loss_ma
    
  # record performance metrics over current minibatch
  pred = sess.run(tf.round(tf.sigmoid(out2)), feed_dict = xy_dict)
  yyhat_dict = {y_trgt:rand_y, yhat:pred}
  tmp_acc_train = sess.run(accuracy, feed_dict=yyhat_dict)
  train_ma[0] = kl.smooth(train_ma[0], tmp_acc_train, theta, (0 == i))
  tmp_pre_train = sess.run(precision, feed_dict=yyhat_dict)
  train_ma[1] = kl.smooth(train_ma[1], tmp_pre_train, theta, (0 == i))
  tmp_rec_train = sess.run(recall, feed_dict=yyhat_dict)
  train_ma[2] = kl.smooth(train_ma[2], tmp_rec_train, theta, (0 == i))
  tmp_f1s_train = sess.run(F1,feed_dict=yyhat_dict)
  train_ma[3] = kl.smooth(train_ma[3], tmp_f1s_train, theta, (0 == i))
  #tmp_acc_train = kl.log_odds(tmp_acc_train) 
  train_vec[i] = train_ma.copy()
       
  if (0 == i % print_freq): 
    print("Smoothed step %u/%u, train batch loss: %0.4f\n\ttrain batch perf: acc=%0.4f,pre=%0.4f,rec=%0.4f,F1=%0.4f"% 
        (i, num_epochs, loss_ma, *train_ma))
    # checkpoint progress
    saver.save(sess, 'log/'+log_file_name+'_ckpt', global_step=i)
  if i % save_freq == 0:
    # save summary stats
    writer.add_summary(losSummary.eval(feed_dict=xy_dict, session=sess),i)
    writer.add_summary(perfSumm.eval(feed_dict = yyhat_dict, session=sess),i)

# ---------------------------------------------
# compute and display the performance metrics for the dev set
print('Computing Dev Set Performance Metrics...')
pred = sess.run(tf.round(tf.sigmoid(out2)), feed_dict = {x_data:x_vals_deve, y_trgt:y_vals_deve})
yyhat_dict = {y_trgt:y_vals_deve, yhat:pred}
acc_deve = sess.run(accuracy, feed_dict=yyhat_dict)
pre_deve = sess.run(precision, feed_dict=yyhat_dict)
rec_deve = sess.run(recall, feed_dict=yyhat_dict)
f1s_deve = sess.run(F1,feed_dict=yyhat_dict)
# save the dev set results separately
devWriter.add_summary(perfSumm.eval(feed_dict = yyhat_dict,session=sess),i+1)
#%%
# display final training set  performance metrics
print('Final Training Set Performance')
print('\tAccuracy = %0.4f'%train_vec[-1][0])
print('\tPrecision = %0.4f'%train_vec[-1][1])
print('\tRecall = %0.4f'%train_vec[-1][2])
print('\tF1 = %0.4f'%train_vec[-1][3])

# display final dev set  performance metrics
print('Final Dev Set Performance')
print('\tAccuracy = %0.4f'%acc_deve)
print('\tPrecision = %0.4f'%pre_deve)
print('\tRecall = %0.4f'%rec_deve)
print('\tF1 = %0.4f'%f1s_deve)

# ---------------------------------------------
#%%
# Display performance plots - no longer need these because it's in tensorboard now
# loss
#plt.plot(loss_vec, 'k-')
#plt.title('Smoothed Cross Entropy Loss')
#plt.xlabel('Generation')
#plt.ylabel('Cross Entropy Loss, EMA')
#plt.show()
#plt.savefig('log/'+log_file_name+'_loss'+'.png')
#
## classification performances
#plt.figure()
#plt.plot(train_vec)
#plt.title('Smoothed Performance Metrics')
#plt.xlabel('Generation')
#plt.ylabel('Metrics, EMA')
#plt.legend(['Accuracy','Precision','Recall','F1'],loc='lower right')
#plt.show()
#plt.savefig('log/'+log_file_name+'_perf'+'.png')

# ---------------------------------------------
#%%
# close out the session and save the event-files
writer.close(); devWriter.close()
# serialize final model results
saver.save(sess, 'log/'+log_file_name+'_final')

sys.stdout.flush()
print('')
etime = datetime.datetime.now()
dtime = etime - stime
print ("Python run end time: " + etime.strftime("%Y-%m-%d %H:%M:%S") )
print ("Python elapsed time: " + str(dtime))
print('Outputs saved with prefix: ./log/%s'%log_file_name)
sys.stdout.flush()
# ---------------------------------------------
# Copyright KAPSARC. Open source MIT License.
# ---------------------------------------------