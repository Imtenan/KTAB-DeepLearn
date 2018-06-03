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
# Parse a model run file(user input) and extract information about the tuning scenarios, including:
# - Scenario name
# - Final Dev Set Performance (4 metrics:Accuracy, Precision, Recall, F1)
# - Hyperparameters:
# Hyperparameters varied include number CDMP actors, learning rate, regularization rate,
# minibatch size, number hidden layers, hidden layer widths, optimization
# method, dropout keep %, and regularization rate. 
# 
# The script also print the top desired numbers of scenarios(user input) based on Accuracy metric. 
# ---------------------------------------------

import sys, os
from collections import OrderedDict

#DIR_PATH = os.path.dirname(os.path.abspath(__file__))
def Convert_pairs(searchlines):
   '''
   Get a line of text and parse in into a key, value pair.  
   '''
   pair= {}
   for j, line1 in enumerate(searchlines):  #j here is not used but needed for the loop
     if (': ' in line1):  
      nsplit=line1.rstrip('\n').lstrip('\t').split(': ')
      ntitle= nsplit[0]
      pair[ntitle] = nsplit[1:]
     if ('=' in line1):
      nsplit=line1.rstrip('\n').lstrip('\t').split(' = ')
      ntitle= nsplit[0]
      pair[ntitle] = nsplit[1:]
     
   return pair
#end of function


def ReadModel(modelOuput):
  '''
  Read a file contain console output from training the nn on the simulated data
  for the tuning scenarios; generated by survey_KT.py 
  , and parse tuning scenarios information
  
  Returns a model run descriptive text and a dictionary of scenarios.
  '''
  try:
    scenario_dict=dict()
    
    with open(modelRun) as f:
      descrip = f.readline().rstrip('\n')
      searchlines = f.readlines()
      
      for i,line in enumerate(searchlines): 
        if "Scenario" in line:
          key=searchlines[i].rstrip('\n')
          sc_name = key.split(': ')
         
        if "Hyperparameters" in line:
          scenario_dict[sc_name[1]]=(Convert_pairs(searchlines[i+1:i+9]))
          tup=list(Convert_pairs(searchlines[i+1:i+9]).items())
          
        if "Final Dev Set Performance" in line:
          tup.extend(list(Convert_pairs(searchlines[i+1:i+5]).items()))   
          scenario_dict[sc_name[1]]=tup
          
  except FileNotFoundError as e:
    raise e
  return descrip,scenario_dict
#end of function


#Get the input file (model run file)
modelRun=input("Enter the Path and Name of the Output File: ")
#modelRun = os.path.join(DIR_PATH, '_Input', 'tmp_survey2csv_models_0_421109_1000txt.log')

#call ReadModel functio to parse the model run file
descrip,scenario_dict = ReadModel(modelRun)

#sort the scenarios based on Accuracy
scenario_list= list(scenario_dict.items())
newlist=sorted(scenario_list, key=lambda d: d[:][1][8], reverse=True)

#Print top scenarios based on Accuracy
x=int(input("Enter the number of top tuning scenarios to display: "))
print('\nBest ', x,' Tuning Scenarios:')
for sce in newlist[:x]:
  print('\n', sce) 

# ---------------------------------------------
# Copyright KAPSARC. Open source MIT License.
# ---------------------------------------------