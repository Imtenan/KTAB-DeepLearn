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
# Neural net to learn one-hot encoding, with a variable number of hidden layers
# tunable hyperparameters are: learning rate, regularization rate, minibatch size
# number of hidden layers, hidden layer(s) width, optimizer, dropout.
#
# Note: the loss function is currently the sigmoid cross entropy of the final
# layer, while I suspect it should be the softmax cross entropy.
#
# ---------------------------------------------

import os
import sys 
import csv
import matplotlib.pyplot as plt
plt.ion()
import numpy as np 
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.model_selection import StratifiedShuffleSplit
import datetime
import klib as kl


''' RUN THE NEURAL NETWORK '''
def RunNN(x_data, y_data, epochs, prng_seed = 0, trainPerc = 0.95, devePerc = 0.05,\
          learn_rate = 1.0, regulRate = 0.5, batch_size = 100, num_cdmp_actors = 4,\
          hiddenLayerWs = [100,100], optimMeth = 'GD', dropoutKeep = 1.0):
  # variable counts
  num_data_col = x_data.shape[1]
  num_choice_col = y_data.shape[1]
  # ---------------------------------------------
  #%%
  sys.stdout.flush()
  stime = datetime.datetime.now()
  print ("Python run start time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
  sys.stdout.flush()
  sys.stdout.flush()
  
  prng_seed = kl.set_prngs(prng_seed)
  
  ops.reset_default_graph()
  sess = tf.Session()
  
  # prepare out and log dirs, if necessary
  try:
    os.mkdir(os.getcwd()+'/out')
  except FileExistsError:
    pass
  try:
    os.mkdir(os.getcwd()+'/log')
  except FileExistsError:
    pass
  
  # ---------------------------------------------
  #%%
  # use sklearn to perform stratified randomized partitioning into training and dev sets
  # this is necessary because the vehicle choice dataset is very unbalanced
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
  modelType = ['NN','KT'][num_cdmp_actors > 0]
  print("num_cdmp_actors: %u" %  num_cdmp_actors)
  
  # nodes per layer - first is the number of features, last is always the number
  # of categories C being predicted
  layerWidths = [num_data_col]
  layerWidths.extend(hiddenLayerWs)
  layerWidths.append(num_choice_col)
  hidden_layers = len(layerWidths)-2     # number hidden layers
  keepProb = tf.placeholder(tf.float32)  # placeholder for dropout
  
  with tf.name_scope('HyperParms'):
    lr = tf.constant(learn_rate)
    la = tf.constant(regulRate)
    bs = tf.constant(batch_size)
    ne = tf.constant(epochs['epochMax'])
    dp = tf.constant(devePerc)
    hl = tf.constant(hidden_layers)
    mt = tf.constant(num_cdmp_actors)
    om = tf.constant(['GD','ADAM','RMSPROP'].index(optimMeth))
    dk = tf.constant(dropoutKeep)
    # summaries
    parmSumm = tf.summary.merge([tf.summary.scalar('learn_rate', lr),\
              tf.summary.scalar('regul_rate', la), tf.summary.scalar('minibatch_size',bs),\
              tf.summary.scalar('epochs',ne), tf.summary.scalar('dev_set_size',dp),\
              tf.summary.scalar('hidden_layers',hl), tf.summary.scalar('num_cdmp_actors',mt),\
              tf.summary.scalar('optim_method',om),tf.summary.scalar('dropout keep',dk)])
  
  # talk some
  print('Hyperparameters\n\tlearning rate: %0.2f\n\tregularization rate: %0.2f\n\tminibatch size: %u\n\tnumber epochs: %d\n\thidden layers: %d\n\tCDMP actors: %d\n\toptim. method: %s\n\tdropout keep: %0.2f'\
        %(learn_rate,regulRate,batch_size,epochs['epochMax'],hidden_layers,num_cdmp_actors,optimMeth,dropoutKeep))
  
  # create the logfile prefix
  log_file_name = 'log_%s%03d_%s_%05d_%s_%d'%(modelType,hidden_layers,optimMeth,epochs['epochMax'],stime.strftime('%Y%m%d_%H%M%S'),prng_seed)
  
  # ---------------------------------------------
  #%%
  # number of rows could be batch_size, num_rows_train, or num_rows_deve.
  # So we mark it as None, which really means variable-size
  with tf.name_scope('Input'):
    x_data = tf.placeholder(shape=[None, num_data_col], dtype=tf.float32,name='X')
    y_trgt = tf.placeholder(shape=[None, num_choice_col], dtype=tf.float32,name='Y')
    yhat = tf.placeholder(shape=[None, num_choice_col], dtype=tf.float32,name='Yhat')
  
  print("x_data shape: %s" % str(x_data.shape))
  print("y_trgt shape: %s" % str(y_trgt.shape))
  
  # ---------------------------------------------
  #%%
  # build each of the layers
  lRng = range(1,len(layerWidths))
  ids = dict.fromkeys(lRng,0.0)
  layerParams = {'A':ids,'b':ids.copy()}
  layerActivs = [None]*(hidden_layers+2) #[0] is an unused placeholder for the data
  for i in lRng:
    # define the scope name
    if (i == (hidden_layers+1)) and (num_cdmp_actors > 0):
      nam = 'CDMP'
    else:
      nam = 'Layer%d'%i
      
    with tf.name_scope(nam):
      # create the variables for the node parameters
      if (i == (hidden_layers+1)) and (num_cdmp_actors > 0):
        # map final hidden layer to the influences vector
        Aw  = tf.Variable(tf.random_normal(shape=[layerWidths[i-1], num_cdmp_actors],\
                                           mean=0.0, stddev=a_stdv))
        print("Aw shape: %s" % str(Aw.shape))
        # NOTE WELL: we do not "learn" on wghts.
        wghts = tf.matmul(layerActivs[i-1], Aw)
        print("wghts shape: %s" % str(wghts.shape))
        # wghts have shape [batch_size, num_cdmp_actors], so there is one vector
        # (1 dim) for each household in this batch
        
        # map final hidden layer to the utilities matrix
        Au =  tf.Variable(tf.random_normal(shape=[layerWidths[i-1], num_cdmp_actors,\
          num_choice_col], mean=0.0, stddev=a_stdv))
        print("Au shape: %s" % str(Au.shape))      
        # NOTE WELL: we do not "learn" on utils.
        utils = tf.tensordot(layerActivs[i-1], Au, axes = [[1], [0]])
        print("utils shape: %s" % str(utils.shape))      
        # utils have shape [batch_size, num_cdmp_actors, num_choice_col], so
        # there is one matrix (2 dim) for each household in this batch
      else:
        layerParams['A'][i] = tf.Variable(tf.random_normal(shape=[layerWidths[i-1],\
                    layerWidths[i]], mean=0.0, stddev=a_stdv),name='A')
        layerParams['b'][i] = tf.Variable(tf.zeros(shape=[layerWidths[i]]),name='b') # a 0-rank array?
        print('A%d shape: %s, b%d shape: %s'%(i,layerParams['A'][i].shape,i,layerParams['b'][i].shape))
      # create the 'variable' for the layer output
      if i == 1:
        # first layer, so input is the data
        layerActivs[i] = tf.nn.relu(tf.add(tf.matmul(x_data,layerParams['A'][i]),\
                   layerParams['b'][i]))
      elif i == (hidden_layers+1):
        # last layer, so no activation function
        if num_cdmp_actors > 0:
          # put the influences and utilities together
          zeta_0 = tf.tensordot(wghts, utils, axes = [[1], [1]])
          print("zeta_0 shape: %s" % str(zeta_0.shape))
          # because it is a *tensor*product*, zeta is tensor of shape (BS, BS, NCC)
          # and includes all the mis-matches where weight-vector(i) was used with
          # util-matrix(j), i.e. Every combination of household weights with every
          # other household's utilities. But we only want the ones where they match.
          # this slice, gather, and reshape picks out the zeta vectors along the 
          # "bi == bj" diagonal.
          slice_ndx = tf.constant([[i,j,k] for i in range(batch_size) \
                                   for j in range(batch_size) for k in \
                                   range(num_choice_col) if i==j])
          # select the large zeta-tensor into a plain vector
          zeta_v = tf.gather_nd(zeta_0, slice_ndx) 
          # NOTE WELL: we do not "learn" on zeta.
          # reshape that vector into  the desired vectors
          layerActivs[i] = tf.reshape(zeta_v, [batch_size, num_choice_col])
          print("After contraction-by-selection, shape of Zeta: %s" % (layerActivs[i].shape))
        else:
          layerActivs[i] =  tf.add(tf.matmul(layerActivs[i-1], layerParams['A'][i]),\
                     layerParams['b'][i])
      else:
        # hidden layer, so apply the activation function ...
        layerActivs[i] = tf.nn.relu(tf.add(tf.matmul(layerActivs[i-1],layerParams['A'][i]),\
                   layerParams['b'][i]))
        # ... then add dropout to the hidden layer
        layerActivs[i] = tf.nn.dropout(layerActivs[i], keepProb)
      print('Activ%d shape: %s'%(i,layerActivs[i].shape))
  
  # ---------------------------------------------
  #%%
  # note that 'labels' are real numbers in [0,1] interval,
  # logits are in (-inf, +inf)
  with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_trgt, logits = layerActivs[-1]))
    losSummary = tf.summary.scalar('Loss', loss)
  
  with tf.name_scope('Train'):
    regularizer =  tf.add_n([tf.nn.l2_loss(A) for A in layerParams['A'].values()])
    if optimMeth == 'GD':
      my_opt = tf.train.GradientDescentOptimizer(learn_rate)
    elif optimMeth == 'ADAM':
      my_opt = tf.train.AdamOptimizer(learn_rate)
    elif optimMeth == 'RMSPROP':
      my_opt = tf.train.RMSPropOptimizer(learn_rate)
    else:
      raise ValueError('Optimizer can only be "GD", "ADAM", or "RMSPROP"!')
    train_step = my_opt.minimize(loss + regulRate*regularizer/(2*batch_size))
  
  # ---------------------------------------------
  #%%
  # record standard classification performance metrics
  with tf.name_scope('Eval'):
    _,accuracy = tf.metrics.accuracy(y_trgt,yhat)
    _,precision = tf.metrics.precision(y_trgt,yhat)
    _,recall = tf.metrics.recall(y_trgt,yhat)
    F1 = 2*tf.divide(tf.multiply(precision,recall),0.0001+tf.add(precision,recall))
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
  writer = tf.summary.FileWriter(os.getcwd()+'/log/train/'+log_file_name, sess.graph)
  writer.add_summary(parmSumm.eval(session=sess))
  devWriter = tf.summary.FileWriter(os.getcwd()+'/log/dev/'+log_file_name,sess.graph)
  devWriter.add_summary(parmSumm.eval(session=sess))
  
  sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
  # The saver object will save all the variables
  saver = tf.train.Saver()
  
  # randomly generate the minibatches all at once
  batchIndxs = np.random.randint(0,num_train_rows,(epochs['epochMax'],batch_size))
  
  # set up for moving averages
  theta = 0.010 # weight on new value  - only for printing/plotting MAs
  loss_ma = 0.0; train_ma = [0.0]*4 # accuracy, precision, recall, F1
  
  loss_vec = [0.0]*epochs['epochMax']
  train_vec = [[0.0]*4 for _ in range(epochs['epochMax'])] # will record accuracy, precision, recall, F1
  
  print_freq = epochs['epochMax']//10   # control console print & checkpoint frequency
  save_freq = epochs['epochMax']//5     # control tensorboard save frequency
  for i in range(epochs['epochMax']):
    # get this minibatch
    rand_x = x_vals_train[batchIndxs[i],:]
    rand_y = y_vals_train[batchIndxs[i],:]
    # train on this batch, and record loss on it
    xy_dict = {x_data:rand_x, y_trgt:rand_y, keepProb:dropoutKeep}
    
    # train, get the loss & it's summary, and predictions
    _,tmp_loss,pred = sess.run([train_step,loss,tf.round(tf.sigmoid(layerActivs[-1]))],\
        feed_dict = xy_dict)
    xy_dict[yhat] = pred
    # compute the evaulation metrics
    tmp_acc_train,tmp_pre_train,tmp_rec_train,tmp_f1s_train = sess.run(\
        [accuracy,precision,recall,F1], xy_dict)
    # smooth it all
    loss_ma = kl.smooth(loss_ma, tmp_loss, theta, (0 == i))
    loss_vec[i]=loss_ma
    train_ma[0] = kl.smooth(train_ma[0], tmp_acc_train, theta, (0 == i))
    train_ma[1] = kl.smooth(train_ma[1], tmp_pre_train, theta, (0 == i))
    train_ma[2] = kl.smooth(train_ma[2], tmp_rec_train, theta, (0 == i))
    train_ma[3] = kl.smooth(train_ma[3], tmp_f1s_train, theta, (0 == i))
    train_vec[i] = train_ma.copy()
         
    if (0 == i % print_freq): 
      print("Smoothed step %u/%u, train batch loss: %0.4f\n\ttrain batch perf: acc=%0.4f,pre=%0.4f,rec=%0.4f,F1=%0.4f"% 
          (i, epochs['epochMax'], loss_ma, *train_ma))
      # checkpoint progress
      saver.save(sess, os.getcwd()+'/log/'+log_file_name+'_ckpt', global_step=i)
    if i % save_freq == 0:
      # save summary stats
      l,p = sess.run([losSummary,perfSumm], xy_dict)
      writer.add_summary(l,i)
      writer.add_summary(p,i)
      
    # check for early termination
    if i > epochs['epochMin']:
      if abs(loss_vec[i]/loss_vec[i-epochs['epochLB']]-1) <= epochs['convgCrit']:
        print('Early termination on epoch %d: relative smoothed loss diff. between %0.6f and %0.6f < %0.6f'%\
              (i,loss_vec[i],loss_vec[i-epochs['epochLB']],epochs['convgCrit']))
        # if terminating early, save the last status
        if i% save_freq != 0:
          l,p = sess.run([losSummary,perfSumm], xy_dict)
          writer.add_summary(l,i)
          writer.add_summary(p,i)
        break
  
  # ---------------------------------------------
  # compute and display the performance metrics for the dev set
  print('Computing Dev Set Performance Metrics...')
  if num_cdmp_actors > 0:
    # get the number of rows short of an even number of batches
    rowsToAdd = batch_size - (x_vals_deve.shape[0] % batch_size)
    if (rowsToAdd > 0) and (rowsToAdd != 100):
      batchesX = np.r_[x_vals_deve,x_vals_deve[-rowsToAdd:,:]]
      batchesY = np.r_[y_vals_deve,y_vals_deve[-rowsToAdd:,:]]
    else:
      batchesX = x_vals_deve; batchesY = y_vals_deve
    # create the minibatches
    bX = np.split(batchesX,x_vals_deve.shape[0]/batch_size,axis=0)
    bY = np.split(batchesY,x_vals_deve.shape[0]/batch_size,axis=0)
    # process the minibatches
    pred = []
    for x,y in zip(bX,bY):
      xy_dict = {x_data:x, y_trgt:y, keepProb:1.0}
      pred.extend(sess.run(tf.round(tf.sigmoid(layerActivs[-1])), xy_dict))
    pred = np.r_[pred]
    # finally build the new dictionary for model evaluation
    xy_dict = {x_data:x_vals_deve, y_trgt:y_vals_deve,\
               yhat:pred[:(x_vals_deve.shape[0]+1)], keepProb:1.0}
  else:
    xy_dict = {x_data:x_vals_deve, y_trgt:y_vals_deve, keepProb:1.0}
    pred = sess.run(tf.round(tf.sigmoid(layerActivs[-1])), xy_dict)
    xy_dict[yhat] = pred
  acc_deve,pre_deve,rec_deve,f1s_deve,summ = sess.run([accuracy,precision,\
                                                       recall,F1,perfSumm],xy_dict)
  # save the summary stats
  devWriter.add_summary(summ,epochs['epochMax']+1)
  
  #%%
  # display final training set performance metrics
  print('Final Training Set Performance')
  print('\tAccuracy = %0.4f'%train_vec[i][0])
  print('\tPrecision = %0.4f'%train_vec[i][1])
  print('\tRecall = %0.4f'%train_vec[i][2])
  print('\tF1 = %0.4f'%train_vec[i][3])
  
  # display final dev set performance metrics
  print('Final Dev Set Performance')
  print('\tAccuracy = %0.4f'%acc_deve)
  print('\tPrecision = %0.4f'%pre_deve)
  print('\tRecall = %0.4f'%rec_deve)
  print('\tF1 = %0.4f'%f1s_deve)
  
  print('Actl. Counts by Class: %r'%np.sum(y_vals_deve,axis=0,dtype=int).tolist())
  print('Pred. Counts by Class: %r'%np.sum(pred,axis=0,dtype=int).tolist())
  # save dev set actuals & preds - might not want to do this permanently
  np.savetxt(os.getcwd()+'/out/actu_pred.csv',np.c_[y_vals_deve,pred],'%d',\
             header='first %d columns are actuals, the rest are predictions'%num_choice_col)
  
  # ---------------------------------------------
  #%%
  # Display performance plots
  # loss
  plt.figure()
  plt.plot(loss_vec[:(i+1)], 'k-')
  plt.title('Smoothed Cross Entropy Loss')
  plt.xlabel('Generation')
  plt.ylabel('Cross Entropy Loss, EMA')
  plt.show()
  plt.savefig(os.getcwd()+'/out/'+log_file_name+'_loss'+'.png')
  
  # classification performances
  plt.figure()
  plt.plot(train_vec[:(i+1)])
  plt.title('Smoothed Performance Metrics')
  plt.xlabel('Generation')
  plt.ylabel('Metrics, EMA')
  plt.legend(['Accuracy','Precision','Recall','F1'],loc='lower right')
  plt.show()
  plt.savefig(os.getcwd()+'/out/'+log_file_name+'_perf'+'.png')
  
  # ---------------------------------------------
  #%%
  # close out the session and save the event-files
  writer.close(); devWriter.close()
  # serialize final model results
  saver.save(sess, os.getcwd()+'/log/'+log_file_name+'_final')
  sess.close()
  
  sys.stdout.flush()
  print('')
  etime = datetime.datetime.now()
  dtime = etime - stime
  print ("Python run end time: " + etime.strftime("%Y-%m-%d %H:%M:%S") )
  print ("Python elapsed time: " + str(dtime))
  print('Outputs saved with prefix: %s'%log_file_name)
  sys.stdout.flush()
  
  return [loss_vec[:(i+1)],acc_deve, pre_deve, rec_deve, f1s_deve]
  # ---------------------------------------------
  #%%
  
  
def ReadData(dataInput):
  '''
  Read in the CSV-formatted data text file.  Format is:
  header line: # rows, # categorical var cols (I1), # float var cols (I2), 
    # categorical var cols (I3), # categorical response cols (D1)
  all else: first I1+I2+I3 columns = independent data; last D1 columns = dependent data
  '''
  
  csvfile = open(dataInput, newline='')  
  csv_data = []
  csv_data_obj = csv.reader(csvfile, delimiter=',', quotechar='|')
  for row in csv_data_obj:
      csv_data.append(row)
  
  # read and parse the header row
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
  num_data_col = num_cat1 + num_float + num_cat2
  print("expected num_data_col: %u" % (num_data_col))
  num_choice_col = int(survey_headers[4])
  print("expected num_choice_col: %u" % (num_choice_col))
  
  # parse data
  survey_list = csv_data[1:] # list of lists
  data_len = len(survey_list)
  print('survey data length: ' + str(data_len))  
  survey_data = [ [float(x) for x in y] for y in survey_list ]
  x_data = np.array([x[0:num_data_col] for x in survey_data]) # [0,6) == [0,5]      
  y_data = np.array([y[num_data_col:num_data_col+num_choice_col] for y in survey_data])
#  x_headers = [h for h in survey_headers[1:6]] 
  print('x-shape: ' + str(x_data.shape))
  print('y-shape: ' + str(y_data.shape)) 
  
  return x_data, y_data



def ReadModel(modelInput):
  '''
  Read in from a text file the parameters for a NN run.  There may be parameters
  for multiple runs.  Format is:
  first line - integer: number of parameter sets in 10 lines per set
  prng_seed = int: psuedo random-number generator seed
  num_cdmp_actors = int: if 0, non-CDMP is used
  epochs = [int, int, int, float]: epochMax, epochMin, epochLookback, convgCrit
  trainPerc, devePerc = two floats: sum to 1 and split the data
  batch_size = int: batch size for training
  learn_rate = float: learning rate for the optimizer
  regulRate = float: regularization rate
  hiddenLayerWs = list of ints: width of all hidden layers
  optimMeth = string: name of optimizer: GD, ADAM, or RMSPROP
  dropoutKeep = float: proportion of data to keep in dropout; 1.0=no dropout
  
  Returns a model runs descriptive text and a list of parameter dicts
  '''
  
  param = {'prng_seed':None,'num_cdmp_actors':None,\
            'epochs':None,'trainPerc':None,'devePerc':None,\
            'batch_size':None,'learn_rate':None,'regulRate':None,\
            'hiddenLayerWs':None,'optimMeth':None,'dropoutKeep':None}

  with open(modelInput,'rt') as f:
    # first line is a descriptive comment, so just read it in and print it
    descrip = f.readline().rstrip('\n')
    # first get the number of runs
    runs = int(f.readline().rstrip('\n'))
    # read in each set of parameters
    params = [param.copy() for _ in range(runs)]
    for i in range(runs):
      params[i]['prng_seed'] = int(f.readline().rstrip('\n'))
      params[i]['num_cdmp_actors'] = int(f.readline().rstrip('\n'))
      tmp = list(f.readline().rstrip('\n').split())
      params[i]['epochs'] = {'epochMax':int(tmp[0]), 'epochMin':int(tmp[1]),\
            'epochLB':int(tmp[2]), 'convgCrit':float(tmp[3])}
      tmp = list(map(float, f.readline().rstrip('\n').split()))
      params[i]['trainPerc'] = tmp[0]
      params[i]['devePerc'] = tmp[1]
      params[i]['batch_size'] = int(f.readline().rstrip('\n'))
      params[i]['learn_rate'] = float(f.readline().rstrip('\n'))
      params[i]['regulRate'] = float(f.readline().rstrip('\n'))
      params[i]['hiddenLayerWs'] = list(map(int, f.readline().rstrip('\n').split()))
      params[i]['optimMeth'] = f.readline().rstrip('\n')
      params[i]['dropoutKeep'] = float(f.readline().rstrip('\n'))

  return descrip,params

if __name__ == '__main__':
  #prng_seed = 831931061 # reproducible
  #prng_seed = 0         # irreproducible
  #prng_seed = 42

  # first need to get the data & model parameters input files
  
  dataInput = sys.argv[1]
  modelInput = sys.argv[2]
  stt = datetime.datetime.now()
  print('Reading data from: %s'%dataInput)
  x_data,y_data = ReadData(dataInput)
  
  print('Reading model parameters from %s'%modelInput)
  descrip,params = ReadModel(modelInput)
  
  # run the model(s)
  print('----------\nRunning model inputs: %s\n----------'%descrip)
  lossDevSetPerfs = [None]*len(params)
  for i,p in enumerate(params):
    lossDevSetPerfs[i] = RunNN(x_data, y_data, p['epochs'], p['prng_seed'],\
                   p['trainPerc'], p['devePerc'], p['learn_rate'], p['regulRate'],\
                   p['batch_size'], p['num_cdmp_actors'], p['hiddenLayerWs'],\
                   p['optimMeth'],p['dropoutKeep'])

  # print elapsed time
  print('----------\nTotal Elapsed Time: %s\n----------'%(datetime.datetime.now()-stt))
# ---------------------------------------------
# Copyright KAPSARC. Open source MIT License.
# ---------------------------------------------