import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df=pd.read_csv('diabetes.csv')
# df.info()

x= df.drop('Outcome',axis=1)
y=df['Outcome']
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=1)
###############################Logistic Regression####################################
######################################################################################
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)
accuracy_score(y_test, y_pred_lr)
print('Testing accuracy of LR model is :',accuracy_score(y_test, y_pred_lr))

dt= DecisionTreeClassifier(random_state=10)
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)
accuracy_score(y_test,y_predict)
print('Testing accuracy of DT model is :',accuracy_score(y_test, y_predict))
