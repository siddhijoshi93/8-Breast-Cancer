
"""

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

breast_cancer= pd.read_csv("C:/Users/ADMIN/Desktop/Siddhi/data.csv")
breast_cancer.columns
breast_cancer.drop('id', axis=1,inplace=True)
breast_cancer.info()
breast_cancer.describe()

#cheking missing values
breast_cancer.isna().sum()
breast_cancer.drop(['Unnamed: 32'], axis=1, inplace=True)

sns.countplot(breast_cancer.diagnosis, palette='YlGnBu')

#coorelation
plt.figure(figsize=(15,8))
sns.heatmap(breast_cancer.corr(),linewidths=.9, yticklabels=False,cmap='YlGnBu', square=False, linecolor='black')

#replacing M and B with 1 and 0
breast_cancer.diagnosis.replace(to_replace=['B','M'],value=[0,1], inplace=True)

#splitting the data
x=breast_cancer.iloc[:,1:]
y= breast_cancer.iloc[:,0]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import logisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#standar Scaler
sc=StandardScaler()
x=sc.fit_transform(x)
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)

#model fitting
acc={}
lr=LogisticRegression()
lr.fit(x_train,y_train)
acc['LogisticRegression']=accuracy_score(y_test,lr.predict(x_test))
accuracy_score(y_test,lr.predict(x_test))
#accuracy=0.98

#knn classifier
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
  acc['K Nearest Neighbors'] = accuracy_score(y_test,knn.predict(x_test))                            
   accuracy_score(y_test,knn.predict(x_test))
#accuracy=0.97                                       

#decison tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)                                            
acc['Decision Tree']=accuracy_score(y_test,dtc.predict(x_test))                                            
   accuracy_score(y_test,dtc.predict(x_test))                                              
tree.plot_tree(dtc)
#accuracy=0.95
                          
#random forest
         from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()                                            
rfc.fit(x_train,y_train)                                            
acc['Random Forest']=accuracy_score(y_test, rfc.predict(x_test))                                          
    accuracy_score(y_test, rfc.predict(x_test))                                          
#accuracy=0.97   
                                         
from sklearn.ensemble import AdaBoostClassifier 
abc=AdaBoostClassifier()                              
abc.fit(x_train,y_train)                                            
acc['Ada Boost']= accuracy_score(y_test, rfc.predict(x_test))                                           
accuracy_score(y_test, rfc.predict(x_test))                                                 
#accuarcy=0.97
key=list(acc.keys())                                            
val=[float(acc[k]) for k in key]                                            
col=sns.color_palette("YlGnBu_r",8)                                            
col=col.as_hex()                                            
                       
fig,acc=plt.subplots(figsize=(16,6))                     
acc=sns.barplot(val,key,palette="YlGnBu")                                            
acc.set_xlim(0.1,1)
for i in range(0,len(key)):                                        
 acc.text(val[i]-0.05,i,str(np.round(val[i],4)),fontdict = dict(color = col[i],fontsize = 18,ha = 'center'),weight = 'bold')                                           
                                            
   #conclusion- Logististic Regression has the highest accuracy, followed by KNN classification, random Forest and Ada Boost Classifier. Decision Tree has the least accuracy among all.                                         
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            