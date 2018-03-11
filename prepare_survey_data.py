
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
# To read a survey data csv file and convert into the format used by 
# the Nueral Net algorithm: (num_rows, num_data, num_choice).
# 
# User input parameter on the maximum allowed number of null values in each attributes.
# 
# ---------------------------------------------

import numpy as np
import pandas as pd
import csv
import math
import os

data_file_name = 'survey.csv'
output_file_name = 'survey_ready'
outHeader = 'survey_head'

# Read the csv data file
data = pd.read_csv(os.getcwd()+os.sep+data_file_name)

# need to one-hot-encode the dependent variable
ohe = pd.get_dummies(data['fuel_size'],'fuel_size')
#ohe = ohe.apply(pd.DataFrame.astype,axis=0,args=('int')) should these be converted to ints?
data.drop('fuel_size',axis=1,inplace=True)
data = data.join(ohe,how='inner')
num_deps = len(ohe.columns)
                                                                
# Calculate the number of columns (attributes) to keep 
missing = np.sum(pd.isnull(data))
origCount = max(data.count())

''' Get the allowed number of empty cells from the user '''
# first show the user the choices
maxMissPercs = [0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 0.9]
toKeeps = [None]*len(maxMissPercs)
for i,perc in enumerate(maxMissPercs):
  maxMissing = math.floor(perc*origCount)
  toKeeps[i] = missing < maxMissing
  print('%d: Variables with missing cutoff at %d, %d max. rows (%0.2f%%) = %d'%\
        (i,maxMissing,origCount-maxMissing,100*perc,sum(toKeeps[i])))
# now get their choice & implement
maxMissing = int(input("Please enter the index number of the cutoff above:"))
toKeep = toKeeps[maxMissing]
toKeep[-num_deps:] = True # be sure the response variables are kept
# update output file names
output_file_name += '_%d.csv'%(100*maxMissPercs[maxMissing])
outHeader += '_%d.txt'%(100*maxMissPercs[maxMissing])

# Get the reduced dataset
dataKeep = data.loc[:,toKeep].dropna()
del(data)
(num_rows,num_indeps) = dataKeep.shape
num_indeps -= num_deps

# Write the dataset to the output data files
with open(os.getcwd() + os.sep + output_file_name, 'w',newline='') as f:
	writer=csv.writer(f)
	writer.writerow([num_rows,num_indeps,num_deps])
	dataKeep.to_csv(f, header=False, index=False)

with open(os.getcwd() + os.sep + outHeader,'w') as f:
  f.write('\n'.join(dataKeep.columns.tolist()))

print('Wrote %d records with %d indep. variables to %s (column names in %s)'%\
      (num_rows,num_indeps,output_file_name,outHeader))
# ---------------------------------------------
# Copyright KAPSARC. Open source MIT License.
# ---------------------------------------------
