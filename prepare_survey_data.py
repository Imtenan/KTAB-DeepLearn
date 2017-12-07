
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

counter=0
data_file_name = 'survey.csv'
output_file_name = 'survey_ready.csv'

# Read the csv data file
data = pd.read_csv(data_file_name)

# Calculate the number of rows in the dataset, number of data attributes (independent var)
# and number of choice attributes (dependent var)
num_rows, num_indeps = data.shape
num_deps=1
num_indeps-=num_deps

# Calculate the number of columns (attributes) to keep 
missing = np.sum(pd.isnull(data))
origCount = max(data.count())

# Take the refernce number of empty cells from the user
targCount = int(input("Maximum number of null values:"))
maxMissing = origCount - targCount
toKeep = (missing < maxMissing)

# Edit the independent var count
for x in toKeep:
    if toKeep[x] == False:
         counter+=1

num_indeps=counter

# Edit the dataset
data = data[data.columns[toKeep]]

# Write the dataset to the output data file
with open(output_file_name, 'w',newline='') as f:
	writer=csv.writer(f)
	writer.writerow([num_rows,num_indeps,1])
	data.to_csv(f, header=False, index=False)


# ---------------------------------------------
# Copyright KAPSARC. Open source MIT License.
# ---------------------------------------------