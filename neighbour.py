from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import pandas as pd
ap=pd.read_csv('/Users/divyanshyadav/Downloads/heart_disease_health_indicators_BRFSS2015.csv')
# print(ap.isnull().sum())
ap=ap.dropna(axis=0)

x=ap.drop('Stroke', axis=1)
y=ap['Stroke']

# a=LabelEncoder()
# x['Gender']=a.fit_transform(x['Gender'])
# x['Dependents']=a.fit_transform(x['Dependents'])
# x['Married']=a.fit_transform(x['Married'])
# x['Education']=a.fit_transform(x['Education'])
# x['Self_Employed']=a.fit_transform(x['Self_Employed'])
# x['Property_Area']=a.fit_transform(x['Property_Area'])
# y=a.fit_transform(y)
# print(y)
scale = StandardScaler()
x = scale.fit_transform(x)
model=KNeighborsClassifier(n_neighbors=5)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
acc=metrics.accuracy_score(y_test, y_pred)
print(acc)
