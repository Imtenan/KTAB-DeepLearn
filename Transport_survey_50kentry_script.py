import numpy as np
import pandas as pd
data = pd.read_csv('c:\\users\\mubarakii\\Documents\\Transport.csv')
np.sum(ot(pd.isnull(data)))
counts = np.sum(~pd.isnull(data))
type(counts)
len(counts)
pd.describe()
data.describe
import matplotlib.pyplot as plt; plt.ion()
plt.hist(counts)
countnull = np.sum(pd.isnull(data))
plt.hist(countnull)
data.dropna().count()
missing = np.sum(pd.isnull(data))
missing == counts
counts=data.count()
missing == counts
max(data.count())
indxs = missing==max(data.count())
data.columns[indxs]
missing = np.sum(pd.isnull(data))
missing
temp=np.sum(data.isnull(), axis=0)
temp
max(missing)
np.percentile(missing,95)
np.percentile(missing,75)
origCount = 274896
targCount = 60000
maxMissing = origCount - targCount
tooMany = (missing >= maxMissing)
sum(tooMany)
maxMissing
newCounts = data.drop(data.columns[tooMany], axis=1).count()
newCounts
targCount = 50000
maxMissing = origCount - targCount
tooMany = (missing >= maxMissing)
sum(tooMany)
tooMany
newCounts = data.drop(data.columns[tooMany], axis=1).count()
newCounts
data.columns[tooMany]
dropme = data.columns[tooMany].tolist()
dropme
np.savetxt('d:\\test.txt',dropme)
data.drop(data.columns[tooMany], axis=1)
data.to_csv('d:\\test.txt')
data.to_csv('d:\\transport_50k.csv')
data.describe()
data.columns[tooMany]
data.drop(data.columns[tooMany], axis=1).count()
sum(data.drop(data.columns[tooMany], axis=1))
help data.columns
toKeep= (missing < maxMissing)
sum(toKeep)
data=data[toKeep]
toKeep= (missing < maxMissing)
toKeep
data.columns[toKeep]
data=data[data.volumns[toKeep]]
data=data[data.columns[toKeep]]
data.describe()
data.to_csv('d:\\transport_50k.csv')