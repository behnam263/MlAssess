import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

    
hdf=pd.HDFStore('Train.h5',mode='r')
hdf.items()
# get keys dataframe
hdf.keys()
df1=hdf.get('/df').fillna(0)
 
# add wordcounts of tweet
df1['Word Count'] = df1['Tweet content'].str.split().str.len()
# Find good values for wordCount
#q = df1["Word Count"].quantile(0.99)
#df2=df1[df1["Word Count"] < q]
Columnlist = []  
# Remove outliers from float columns except tweet ID 
df2=df1
for c in df1.columns:
    if c!='Tweet Id' and  df2.dtypes[c]== 'float64' :
        q = df2[c].quantile(0.99)
        df2=df2[df1[c] < q]
        Columnlist.append(c)
       # df2.apply(lambda s: df2.corrwith(s))

corr = pd.DataFrame()
for a in Columnlist:
    for b in Columnlist:
        corr.loc[a, b] = df2.corr().loc[a, b]
ax = sns.heatmap(corr)

colors = plt.cm.rainbow(np.linspace(0, 1, len(Columnlist)))

for c in Columnlist:
    if c!="rank":
        #plt.plot(df2[c],df2["rank"], color=colors[Columnlist.index(c)] )
        fig = plt.figure()
        plt.scatter(df2[c],df2["rank"])
        plt.xlabel('rank')
        plt.ylabel(c)
        plt.title('Relations')
        plt.show()