import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets 
import sklearn.model_selection
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import mean_squared_error, r2_score


    
hdf=pd.HDFStore('Train.h5',mode='r')
hdf.items()
# get keys dataframe
hdf.keys()
df1=hdf.get('/df').fillna(0)
 
# add wordcounts of tweet
df1['Word Count'] = df1['Tweet content'].str.split().str.len()

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
        
#Seperate Data for test,validation,train
input_train,input_validation=sklearn.model_selection.train_test_split(df2,test_size=0.2,random_state=100,shuffle=True)

min_max_scaler = preprocessing.MinMaxScaler()
 

   
#linear regression
y=  (input_train['rank']-min(input_train['rank'])) /( max(input_train['rank'])-min(input_train['rank']))
yv= (input_validation['rank']-min(input_validation['rank'])) /( max(input_validation['rank'])-min(input_validation['rank'])) 

least_error=np.finfo(np.float64).max
featurewithLeastSimple=""
#it is our target
Columnlist.remove('rank')

#remove Highly correlated features
Columnlist.remove('reply')


df3=input_train[Columnlist]
#remove outliers
#low = .05
#high = .95
#quant_df = df3.quantile([low, high])
#for col in Columnlist:
#        if is_numeric_dtype(df3[col]):
#            df3 = df3[(df3[col] > quant_df.loc[low, col]) & (df3[col] < quant_df.loc[high, col])]


#Scale all data between 0 and 1
x_scaled=min_max_scaler.fit_transform(df3)
df3 = pd.DataFrame(x_scaled)



for col in Columnlist:
    x=input_train[col]
    Linearmodel = ols("y ~ x", x).fit()
    offset, coef = Linearmodel._results.params
    anova_results = anova_lm(Linearmodel)
    if(least_error> anova_results['mean_sq'][1]):
        least_error=anova_results['mean_sq'][1]
        featurewithLeastSimple=col
print('\nANOVA results')
print(anova_results)
plt.plot(x, x*coef + offset)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
print ('Best Feature was: '+ featurewithLeastSimple )


#multiple feature regression 
multipleReg = LinearRegression()
multipleReg.fit(df3, y) 
print('Weights are : ')
print (multipleReg.coef_)
pridiction=multipleReg.predict(input_validation[Columnlist]);
#print(multipleReg.score(input_validation[Columnlist],input_validation['rank'], sample_weight=None))
mea= sum(abs(yv-pridiction))/len(yv)
print("Mea for multipleReg:",mea)


#Ridge regression
clfRidge = Ridge(alpha=0.01)
clfRidge.fit(df3, y) 
pridiction=clfRidge.predict(input_validation[Columnlist]);
mea= sum(abs(yv-pridiction))/len(yv)
print("Mea for Ridge:",mea)
RSS = (abs(yv - pridiction) ** 2).sum()
print("RSS for Ridge:",RSS)
print(clfRidge.score(input_validation[Columnlist],yv, sample_weight=None))

#Lasso 
reglasso = Lasso(alpha =0.1)
reglasso.fit(df3, y) 
print(reglasso)
pridiction=reglasso.predict(input_validation[Columnlist]);
mea= sum(abs(yv-pridiction))/len(yv)
print("Mea for Lasso:",mea)


