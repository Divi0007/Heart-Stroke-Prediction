# !/usr/bin/python3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
ap=pd.read_csv("/Users/divyanshyadav/Downloads/heart_2020_cleaned.csv")
print(ap.head())
x=ap.drop('Stroke', axis=1)
y=ap['Stroke']
a=LabelEncoder()
b=LabelEncoder()
c=LabelEncoder()
x['Race']=a.fit_transform(x['Race'])
x['AgeCategory']=b.fit_transform(x['AgeCategory'])
x['GenHealth']=c.fit_transform(x['GenHealth'])
ada=RandomForestClassifier()
p=[0.2,0.3,0.4,0.5,0.6,0.7,0.8]
Score=[]
for i in p:
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=i,random_state=200)
    ada.fit(x_train,y_train)
    y_pred=ada.predict(x_test)
    h=metrics.accuracy_score(y_test, y_pred)*100
    Score.append(h)
print(Score)
plt.xlim([0,1])
plt.ylim([0,100])
plt.bar(p,Score,color ='blue',width = 0.07)
plt.xlabel("test size")
plt.ylabel("accuracy")
plt.title("accuracy")
plt.show()