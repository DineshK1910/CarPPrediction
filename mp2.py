import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
df=pd.read_csv("acp1/data.csv")
le=LabelEncoder()
print(df.dtypes)
print("shape before:",df.shape)
#data cleaning
print(df.isnull().sum())

df.drop("Market Category",axis=1,inplace=True)
df.drop("Model",axis=1,inplace=True)
df.drop("Number of Doors",axis=1,inplace=True)
#df=df.drop_duplicates()
print(df.columns)
df["Engine HP"]=df["Engine HP"].fillna(0)
df["Engine Cylinders"]=df["Engine Cylinders"].fillna(0)
df["Engine Fuel Type"]=df["Engine Fuel Type"].fillna("regular unleaded")
print(df.isnull().sum())
#data transforming
for i in df.columns.values:
    if df[i].dtype==object:
        df[i]=le.fit_transform(df[i])
#plotting boxplot
plt.boxplot(df["Engine HP"])
plt.xlabel("HP")
plt.show()
plt.boxplot(df["Year"])
plt.xlabel("Year")
plt.show()
plt.show
plt.boxplot(df["Engine Cylinders"])
plt.xlabel("Cylinders")
plt.show()
#removing outliers
for i in df.columns.values:
    q1=df[i].quantile(0.25)
    q3=df[i].quantile(0.75)
    iqr=q3-q1
    low=q1-(1.5*iqr)
    upp=q3+(1.5*iqr)
    minn=df[df[i]<low].index
    maxx=df[df[i]>upp].index
    df.drop(minn,inplace=True)
    df.drop(maxx,inplace=True)
c=[x for x in df.columns.values if x!="MSRP"]
y=df["MSRP"]
x=df[c]
print(x.columns)
sc=StandardScaler()
rf=RandomForestRegressor(n_estimators=90)
#random forest regression
pipeline = Pipeline([('transformer', sc), ('estimator', rf)])
scores=cross_val_score(pipeline,x,y,cv=4)
print("the cross validation accuracy:",scores.mean())
xtr,xts,ytr,yts=train_test_split(x,y,test_size=0.30)
rf.fit(xtr,ytr)
yp=rf.predict(xts)
print("the train test split:",r2_score(yp,yts))
   
