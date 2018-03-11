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
# Neural net to learn one-hot encoding, with a variable number of hidden layers.
#
# All console output is also printed to log file, based on a logging object set
# in main().
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
import logging
import klib as kl


''' RUN THE NEURAL NETWORK '''
def RunNN(x_data, y_data, epochs, prng_seed = 0, trainPerc = 0.95, devePerc = 0.05,\
          learn_rate = 1.0, learn_rate_decay = 0.0, regulRate = 0.5, batch_size = 100, num_cdmp_actors = 4,\
          hiddenLayerWs = [1.5,1.0], optimMeth = 'GD', optimBetas=[0.0,0.0], dropoutKeep = 1.0, scenName = 'noname'):
  # create the integer decoded version of y - this is only needed for
  # calculation of the performance metrics
  y_decode = np.argmax(y_data,axis=1)
  # variable counts
  num_data_col = x_data.shape[1]
  num_choice_col = y_data.shape[1]
  # ---------------------------------------------
  #%%
  sys.stdout.flush()
  stime = datetime.datetime.now()
  logging.info("RunNN start time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
  logging.info('Scenario: '+scenName)
  sys.stdout.flush()
  sys.stdout.flush()
  
  prng_seed = kl.set_prngs(prng_seed)
  
  ops.reset_default_graph()
  sess = tf.Session()
  
  # prepare log dir, if necessary
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
  y_dec_train = y_decode[train_indices]
  
  x_vals_deve = x_data[deve_indices,:]
  y_vals_deve = y_data[deve_indices,:]
  y_dec_deve = y_decode[deve_indices]
  
  logging.info("num_train_rows: %u, num_deve_rows: %u" %(num_train_rows, len(deve_indices)))
  
  # ---------------------------------------------
  #%%
  # setup training
  a_stdv = 0.1          # standard dev. for initialization of node weights
  modelType = ['NN','KT'][num_cdmp_actors > 0]
  logging.info("num_cdmp_actors: %u" %  num_cdmp_actors)
  
  # nodes per layer - first is the number of features, last is always the number
  # of categories C being predicted
  layerWidths = [num_data_col]
  layerWidths.extend([int(round(num_data_col*i,0)) for i in hiddenLayerWs])
  layerWidths.append(num_choice_col)
  hidden_layers = len(layerWidths)-2     # number hidden layers
  keepProb = tf.placeholder(tf.float32)  # placeholder for dropout
  learnRate = tf.placeholder(tf.float32) # placeholder for adaptive learning rate
  
  with tf.name_scope('HyperParms'):
    lr = tf.constant(learn_rate)
    ld = tf.constant(learn_rate_decay)
    la = tf.constant(regulRate)
    bs = tf.constant(batch_size)
    ne = tf.constant(epochs['epochMax'])
    dp = tf.constant(devePerc)
    hl = tf.constant(hidden_layers)
    mt = tf.constant(num_cdmp_actors)
    om = tf.constant(['GD','RMSPROP','ADAM'].index(optimMeth))
    b1 = tf.constant(optimBetas[0])
    b2 = tf.constant(optimBetas[1])
    dk = tf.constant(dropoutKeep)
    # summaries
    parmSumm = tf.summary.merge([tf.summary.scalar('learn_rate', lr),\
              tf.summary.scalar('learn_rate_decay',ld),tf.summary.scalar('regul_rate', la),\
              tf.summary.scalar('minibatch_size',bs),tf.summary.scalar('epochs',ne),\
              tf.summary.scalar('dev_set_size',dp),tf.summary.scalar('hidden_layers',hl),\
              tf.summary.scalar('num_cdmp_actors',mt),tf.summary.scalar('optim_method',om),\
              tf.summary.scalar('optim_beta1',b1),tf.summary.scalar('optim_beta2',b2),\
              tf.summary.scalar('dropout_keep',dk)])
  
  # talk some
  logging.info('Hyperparameters\n\tlearning rate: %0.2f(%0.2f)\n\tregularization rate: %0.2f\n\tminibatch size: %u\n\tnumber epochs: %d\n\thidden layers: %d\n\tCDMP actors: %d\n\toptim. method: %s(%0.2f,%0.2f)\n\tdropout keep: %0.2f'\
        %(learn_rate,learn_rate_decay,regulRate,batch_size,epochs['epochMax'],hidden_layers,num_cdmp_actors,optimMeth,optimBetas[0],optimBetas[1],dropoutKeep))
  
  # create the logfile prefix
  log_file_name = 'log_%s%04d_%s_%s_%05d_%s_%d'%(modelType,hidden_layers,scenName,optimMeth,epochs['epochMax'],stime.strftime('%Y%m%d_%H%M%S'),prng_seed)
  
  # ---------------------------------------------
  #%%
  # number of rows could be batch_size, num_rows_train, or num_rows_deve.
  # So we mark it as None, which really means variable-size
  with tf.name_scope('Input'):
    x_data = tf.placeholder(shape=[None, num_data_col], dtype=tf.float32,name='X')
    y_trgt = tf.placeholder(shape=[None, num_choice_col], dtype=tf.float32,name='Y')
    y_dec = tf.placeholder(shape=[None,], dtype=tf.float32,name='Ydecode')
    yhat_dec = tf.placeholder(shape=[None, ], dtype=tf.float32,name='Yhatdecode')    
  
  logging.info("x_data shape: %s" % str(x_data.shape))
  logging.info("y_trgt shape: %s" % str(y_trgt.shape))
  
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
        logging.info("Aw shape: %s" % str(Aw.shape))
        # NOTE WELL: we do not "learn" on wghts.
        wghts = tf.matmul(layerActivs[i-1], Aw)
        logging.info("wghts shape: %s" % str(wghts.shape))
        # wghts have shape [batch_size, num_cdmp_actors], so there is one vector
        # (1 dim) for each household in this batch
        
        # map final hidden layer to the utilities matrix
        Au =  tf.Variable(tf.random_normal(shape=[layerWidths[i-1], num_cdmp_actors,\
          num_choice_col], mean=0.0, stddev=a_stdv))
        logging.info("Au shape: %s" % str(Au.shape))      
        # NOTE WELL: we do not "learn" on utils.
        utils = tf.tensordot(layerActivs[i-1], Au, axes = [[1], [0]])
        logging.info("utils shape: %s" % str(utils.shape))      
        # utils have shape [batch_size, num_cdmp_actors, num_choice_col], so
        # there is one matrix (2 dim) for each household in this batch
      else:
        layerParams['A'][i] = tf.Variable(tf.random_normal(shape=[layerWidths[i-1],\
                    layerWidths[i]], mean=0.0, stddev=a_stdv),name='A')
        layerParams['b'][i] = tf.Variable(tf.zeros(shape=[layerWidths[i]]),name='b') # a 0-rank array?
        logging.info('A%d shape: %s, b%d shape: %s'%(i,layerParams['A'][i].shape,i,layerParams['b'][i].shape))
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
          logging.info("zeta_0 shape: %s" % str(zeta_0.shape))
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
          logging.info("After contraction-by-selection, shape of Zeta: %s" % (layerActivs[i].shape))
        else:
          layerActivs[i] =  tf.add(tf.matmul(layerActivs[i-1], layerParams['A'][i]),\
                     layerParams['b'][i])
      else:
        # hidden layer, so apply the activation function ...
        layerActivs[i] = tf.nn.relu(tf.add(tf.matmul(layerActivs[i-1],layerParams['A'][i]),\
                   layerParams['b'][i]))
        # ... then add dropout to the hidden layer
        layerActivs[i] = tf.nn.dropout(layerActivs[i], keepProb)
      logging.info('Activ%d shape: %s'%(i,layerActivs[i].shape))
  
  # ---------------------------------------------
  #%%
  # note that 'labels' are real numbers in [0,1] interval,
  # logits are in (-inf, +inf)
  with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_trgt, logits = layerActivs[-1]))
    losSummary = tf.summary.scalar('Loss', loss)
  
  with tf.name_scope('Train'):
    regularizer =  tf.add_n([tf.nn.l2_loss(A) for A in layerParams['A'].values()])
    if optimMeth == 'GD':
      my_opt = tf.train.GradientDescentOptimizer(learnRate)
    elif optimMeth == 'RMSPROP':
      my_opt = tf.train.RMSPropOptimizer(learnRate,momentum=optimBetas[0])
    elif optimMeth == 'ADAM':
      my_opt = tf.train.AdamOptimizer(learnRate,beta1=optimBetas[0],beta2=optimBetas[1])      
    else:
      raise ValueError('Optimizer can only be "GD", "RMSPROP", or "ADAM"!')
    train_step = my_opt.minimize(loss + regulRate*regularizer/(2*batch_size))
  
  # ---------------------------------------------
  #%%
  # record standard classification performance metrics
  # note that these require integer labels *not* OHE!
  with tf.name_scope('Eval'):
    _,accuracy = tf.metrics.accuracy(y_dec,yhat_dec)
    _,precision = tf.metrics.precision(y_dec,yhat_dec)
    _,recall = tf.metrics.recall(y_dec,yhat_dec)
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
    rand_y_dec = y_dec_train[batchIndxs[i]]
    # compute the adaptive learning rate
    learnRateAdapt = learn_rate*(1+learn_rate_decay*i)
    # train on this batch, and record loss on it
    xy_dict = {x_data:rand_x, y_trgt:rand_y, learnRate:learnRateAdapt,\
               keepProb:dropoutKeep, y_dec:rand_y_dec}
    
    # train, get the loss & it's summary, and predictions
    _,tmp_loss,classProbs = sess.run([train_step,loss,tf.nn.softmax(layerActivs[-1])],\
        feed_dict = xy_dict)
    pred = kl.predFromProb(classProbs)
    
    # need to handle nan loss
    if np.isnan(tmp_loss):
      loss_vec[i] = np.inf
      train_vec[i] = [0.0,0.0,0.0,0.0]
      # save summary stats
      xy_dict[yhat_dec] = pred
      l,p = sess.run([losSummary,perfSumm], xy_dict)
      writer.add_summary(l,i)
      writer.add_summary(p,i)
      logging.info('Early termination due to nan loss on epoch %u!'%i)
      break
    
    xy_dict[yhat_dec] = pred
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
      logging.info("Smoothed step %u/%u, train batch loss: %0.4f\n\ttrain batch perf: acc=%0.4f,pre=%0.4f,rec=%0.4f,F1=%0.4f"% 
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
        logging.info('Early termination on epoch %d: relative smoothed loss diff. between %0.6f and %0.6f < %0.6f'%\
              (i,loss_vec[i],loss_vec[i-epochs['epochLB']],epochs['convgCrit']))
        # if terminating early, save the last status
        if i% save_freq != 0:
          l,p = sess.run([losSummary,perfSumm], xy_dict)
          writer.add_summary(l,i)
          writer.add_summary(p,i)
        break
  
  # ---------------------------------------------
  # compute and display the performance metrics for the dev set
  logging.info('Computing Dev Set Performance Metrics...')
  if num_cdmp_actors > 0:
    # get the number of rows short of an even number of batches
    rowsToAdd = (1+x_vals_deve.shape[0]//batch_size)*batch_size - x_vals_deve.shape[0]
    if (rowsToAdd > 0) and (rowsToAdd != batch_size):
      batchesX = np.r_[x_vals_deve,x_vals_deve[-rowsToAdd:,:]]
      batchesY = np.r_[y_vals_deve,y_vals_deve[-rowsToAdd:,:]]
      batchesYD = np.r_[y_dec_deve,y_dec_deve[-rowsToAdd:]]
    else:
      batchesX = x_vals_deve; batchesY = y_vals_deve; batchesYD = y_dec_deve
    # create the minibatches
    bX = np.split(batchesX,batchesX.shape[0]/batch_size,axis=0)
    bY = np.split(batchesY,batchesX.shape[0]/batch_size,axis=0)
    bYD = np.split(batchesYD,batchesX.shape[0]/batch_size,axis=0)
    # process the minibatches
    classProbs = []
    for x,y,yd in zip(bX,bY,bYD):
      xy_dict = {x_data:x, y_trgt:y, keepProb:1.0, y_dec:yd}
      classProbs.extend(sess.run(tf.nn.softmax(layerActivs[-1]), xy_dict))
    classProbs = np.r_[classProbs]
    # now remove the extra repeated observations at the end
    if (rowsToAdd > 0) and (rowsToAdd != batch_size):
      classProbs = classProbs[:(x_vals_deve.shape[0])]
    # finally build the new dictionary for model evaluation
    xy_dict = {x_data:x_vals_deve, y_trgt:y_vals_deve, y_dec:y_dec_deve,\
               keepProb:1.0}
    xy_dict[yhat_dec] = kl.predFromProb(classProbs)
  else:
    xy_dict = {x_data:x_vals_deve, y_trgt:y_vals_deve, keepProb:1.0, y_dec:y_dec_deve}
    classProbs = sess.run(tf.nn.softmax(layerActivs[-1]), xy_dict)
    xy_dict[yhat_dec] = kl.predFromProb(classProbs)
  acc_deve,pre_deve,rec_deve,f1s_deve,summ = sess.run([accuracy,precision,\
                                                       recall,F1,perfSumm],xy_dict)
  
  # convert integer class labels to OHE matrix
  prd = tf.placeholder(shape=xy_dict[yhat_dec].shape, dtype=tf.int32)
  OHE = tf.one_hot(prd,num_choice_col)
  predOHE = sess.run(OHE,feed_dict = {prd:xy_dict[yhat_dec]})
  
  # save the summary stats
  devWriter.add_summary(summ,epochs['epochMax']+1)
  
  #%%
  # display final training set performance metrics
  logging.info('Final Training Set Performance')
  logging.info('\tAccuracy = %0.4f'%train_vec[i][0])
  logging.info('\tPrecision = %0.4f'%train_vec[i][1])
  logging.info('\tRecall = %0.4f'%train_vec[i][2])
  logging.info('\tF1 = %0.4f'%train_vec[i][3])
  
  # display final dev set performance metrics
  logging.info('Final Dev Set Performance')
  logging.info('\tAccuracy = %0.4f'%acc_deve)
  logging.info('\tPrecision = %0.4f'%pre_deve)
  logging.info('\tRecall = %0.4f'%rec_deve)
  logging.info('\tF1 = %0.4f'%f1s_deve)
  
  logging.info('Actl. Counts by Class: %r'%np.sum(y_vals_deve,axis=0,dtype=int).tolist())
  logging.info('Pred. Counts by Class: %r'%np.sum(predOHE,axis=0,dtype=int).tolist())
  # save dev set actuals & preds - might not want to do this permanently
  np.savetxt(os.getcwd()+'/out/'+log_file_name+'_actu_pred.csv',np.c_[y_vals_deve,predOHE],'%d',\
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
  logging.info('')
  etime = datetime.datetime.now()
  dtime = etime - stime
  logging.info("RunNN end time: " + etime.strftime("%Y-%m-%d %H:%M:%S") )
  logging.info("RunNN elapsed time: " + str(dtime))
  logging.info('Outputs saved with prefix: %s'%log_file_name)
  sys.stdout.flush()
  
  return [loss_vec[:(i+1)],acc_deve, pre_deve, rec_deve, f1s_deve]
  # ---------------------------------------------
  #%%
  
  
def ReadData(dataInput):
  '''
  Read in the CSV-formatted data text file.  Format is:
  header line: # rows, # columns independent vars, # columns dependent vars, seed if data is sim'd
  all else: all independent data; all dependent data
  '''
  
  try:
    csvfile = open(dataInput, newline='')
  except FileNotFoundError as e:
    logging.info('data input file %s not found!'%dataInput)
    raise e
  csv_data = []
  csv_data_obj = csv.reader(csvfile, delimiter=',', quotechar='|')
  for row in csv_data_obj:
      csv_data.append(row)
  
  # read and parse the header row
  # The data file format will have to be changed to (num_rows, num_data, num_choice)
  logging.info('csv length: ' + str(len(csv_data)))
  survey_headers = [int(x) for x in csv_data[0]] # Num rows, # indep vars, # dep vars
  num_rows = survey_headers[0]
  logging.info("expected num_rows: %u" % (num_rows))
  num_indeps = survey_headers[1]
  logging.info("expected num_indeps: %u" % (num_indeps))
  num_deps = survey_headers[2]
  logging.info("expected num_deps: %u" % (num_deps))
  if len(survey_headers) == 4:
    logging.info('generating seed: %u'%survey_headers[-1])
  
  # parse data
  survey_list = csv_data[1:] # list of lists
  data_len = len(survey_list)
  logging.info('survey data length: ' + str(data_len))  
  survey_data = [ [float(x) for x in y] for y in survey_list ]
  x_data = np.array([x[0:num_indeps] for x in survey_data]) # [0,6) == [0,5]      
  y_data = np.array([y[num_indeps:num_indeps+num_deps] for y in survey_data])
  #  x_headers = [h for h in survey_headers[1:6]] 
  logging.info('x-shape: ' + str(x_data.shape))
  logging.info('y-shape: ' + str(y_data.shape)) 
  
  return x_data, y_data



def ReadModel(modelInput):
  '''
  Read in from a text file the parameters for a NN run.  There may be parameters
  for multiple runs.  Format is:
  first line = integer: number of parameter sets in 11 lines per set
  scenName = string: psuedorandom 'name' for the scenario
  prng_seed = int: psuedo random-number generator seed
  num_cdmp_actors = int: if 0, non-CDMP is used
  epochs = [int, int, int, float]: epochMax, epochMin, epochLookback, convgCrit
  trainPerc, devePerc = two floats: sum to 1 and split the data
  batch_size = int: batch size for training
  learn_rate, learn_rate_decay = [float float]: learning rate for the optimizer
    tand it's decay
  regulRate = float: regularization rate
  hiddenLayerWs = list of floats: widths of all hidden layers, relative to size
    of input data
  optimMeth,optimBetas = [string, float, float]: name of optimizer: GD, RMSPROP,
    or ADAM; if not 'GD', the floats are the momentums (if 'GD', just put 0.0s)
  dropoutKeep = float: proportion of data to keep in dropout; 1.0=no dropout
  
  Returns a model runs descriptive text and a list of parameter dicts
  '''
  
  param = {'scenName':None,'prng_seed':None,'num_cdmp_actors':None,\
            'epochs':None,'trainPerc':None,'devePerc':None,\
            'batch_size':None,'learn_rate':None,'learn_rate_decay':None,\
            'regulRate':None,'hiddenLayerWs':None,'optimMeth':None,\
            'optimBetas':None,'dropoutKeep':None}

  try:
    with open(modelInput,'rt') as f:
      # first line is a descriptive comment, so just read it in and print it
      descrip = f.readline().rstrip('\n')
      # first get the number of runs
      runs = int(f.readline().rstrip('\n'))
      # read in each set of parameters
      params = [param.copy() for _ in range(runs)]
      for i in range(runs):
        params[i]['scenName'] = f.readline().rstrip('\n')
        params[i]['prng_seed'] = int(f.readline().rstrip('\n'))
        params[i]['num_cdmp_actors'] = int(f.readline().rstrip('\n'))
        tmp = list(f.readline().rstrip('\n').split())
        params[i]['epochs'] = {'epochMax':int(tmp[0]), 'epochMin':int(tmp[1]),\
              'epochLB':int(tmp[2]), 'convgCrit':float(tmp[3])}
        tmp = list(map(float, f.readline().rstrip('\n').split()))
        params[i]['trainPerc'] = tmp[0]
        params[i]['devePerc'] = tmp[1]
        params[i]['batch_size'] = int(f.readline().rstrip('\n'))
        tmp = list(map(float, f.readline().rstrip('\n').split()))
        params[i]['learn_rate'] = tmp[0]
        params[i]['learn_rate_decay'] = tmp[1]
        params[i]['regulRate'] = float(f.readline().rstrip('\n'))
        params[i]['hiddenLayerWs'] = list(map(float, f.readline().rstrip('\n').split()))
        tmp = list(f.readline().rstrip('\n').split())
        params[i]['optimMeth'] = tmp[0]
        params[i]['optimBetas'] = [float(a) for a in tmp[1:]]
        params[i]['dropoutKeep'] = float(f.readline().rstrip('\n'))
  except FileNotFoundError as e:
    logging.info('model input file %s not found!'%modelInput)
    raise e        

  return descrip,params

if __name__ == '__main__':
  #prng_seed = 831931061 # reproducible
  #prng_seed = 0         # irreproducible
  #prng_seed = 42

  # first need to get the data & model parameters input files  
  dataInput = sys.argv[1]
  modelInput = sys.argv[2]
  
  # setup logger
  log = os.getcwd() + '/out/'+dataInput.replace('.','').replace('/','')+\
    '_'+modelInput.replace('.','').replace('/','')+'.log'
  try:
    os.mkdir(os.getcwd()+'/out')
  except FileExistsError:
    pass    
  lev = logging.INFO
  fmt = '%(message)s'
  hnd = [logging.FileHandler(log, mode='w'),logging.StreamHandler()]
  logging.basicConfig(level=lev,format=fmt,handlers=hnd)
  
  stt = datetime.datetime.now()
  logging.info('Reading data from: %s'%dataInput)
  x_data,y_data = ReadData(dataInput)
  
  logging.info('Reading model parameters from %s'%modelInput)
  descrip,params = ReadModel(modelInput)
  
  # run the model(s)
  logging.info('----------\nRunning model inputs: %s\n----------'%descrip)
  lossDevSetPerfs = [None]*len(params)
  for i,p in enumerate(params):
      plt.close('all')
      lossDevSetPerfs[i] = RunNN(x_data, y_data, p['epochs'], p['prng_seed'],\
                     p['trainPerc'], p['devePerc'], p['learn_rate'], \
                     p['learn_rate_decay'],p['regulRate'],\
                     p['batch_size'], p['num_cdmp_actors'], p['hiddenLayerWs'],\
                     p['optimMeth'],p['optimBetas'],p['dropoutKeep'],p['scenName'])

  # print elapsed time
  logging.info('----------\nTotal Elapsed Time: %s\n----------'%(datetime.datetime.now()-stt))
  logging.info('Modeling run log printed to '+log)
  
'''  
# debug setup
epochs = params[0]['epochs']
prng_seed = params[0]['prng_seed']
trainPerc = params[0]['trainPerc']
devePerc = params[0]['devePerc']
learn_rate = params[0]['learn_rate']
learn_rate_decay = params[0]['learn_rate_decay']
regulRate = params[0]['regulRate']
batch_size = params[0]['batch_size']
num_cdmp_actors = params[0]['num_cdmp_actors']
hiddenLayerWs = params[0]['hiddenLayerWs']
optimMeth = params[0]['optimMeth']
optimBetas=params[0]['optimBetas']
dropoutKeep = params[0]['dropoutKeep']
scenName = params[0]['scenName']
'''  
  
# ---------------------------------------------
# Copyright KAPSARC. Open source MIT License.
# ---------------------------------------------
