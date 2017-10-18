# Script to create a subset of a dataset according to a specific reference number indicating the number of empty cells allowed to have in that subset. It saves the new dataset in a csv file with the same format.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.ion()

#read csv file of the oroginal dataset
data = pd.read_csv('c:\\users\\mubarakii\\Documents\\Transport.csv')

#calculate the number of columns (attributes) to keep
missing = np.sum(pd.isnull(data))
origCount = max(data.count())
#take the refernce number of empty cells from the user
targCount = int(input("Number of not null cells in each attribute:"))
maxMissing = origCount - targCount
toKeep= (missing < maxMissing)

#edit the dataset
data=data[data.columns[toKeep]]

#save it to csv
data.to_csv('d:\\transport_50k.csv', index=False)