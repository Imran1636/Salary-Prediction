import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

dataset =pd.read_csv("DataSal.csv")
#print(dataset)

exp=dataset.iloc[:,0].values
exp = exp.reshape(len(exp),1)
Eng_sal=dataset.iloc[:,1].values
Eng_sal=Eng_sal.reshape(len(Eng_sal),1)
Med_sal = dataset.iloc[:,2].values
Med_sal = Med_sal.reshape(len(Med_sal),1)
Fin_sal = dataset.iloc[:,3].values
Fin_sal = Fin_sal.reshape(len(Fin_sal),1)
Bus_sal = dataset.iloc[:,4].values
Bus_sal = Bus_sal.reshape(len(Bus_sal),1)
Law_sal = dataset.iloc[:,5].values
Law_sal = Law_sal.reshape(len(Law_sal),1)
Teach_sal = dataset.iloc[:,6].values
Teach_sal = Teach_sal.reshape(len(Teach_sal),1)

def Engineering(exp1):
    
    exp_train,exp_test,sal_train,sal_test = train_test_split(exp,Eng_sal,test_size=1/3,random_state =0)
    regressor =LinearRegression()
    regressor.fit(exp_train,sal_train)
    #print("--------------------------")
    #print(exp_train,sal_train)
    sal_pred = regressor.predict(exp_test)
    #print(exp_test)
    #print(sal_pred)
    sal1_pred = regressor.predict([[exp1]])
    filename="Engineering.sav"
    pickle.dump(regressor,open(filename,"wb"))
    print(sal1_pred[0][0])

def Medicine(exp1):
    exp_train,exp_test,sal_train,sal_test = train_test_split(exp,Med_sal,test_size=1/3,random_state = 0)
    regressor = LinearRegression()
    regressor.fit(exp_train,sal_train)
    sal_pred = regressor.predict(exp_test)
    sal1_pred = regressor.predict([[exp1]])
    filename="Medicine.sav"
    pickle.dump(regressor,open(filename,"wb"))
    print(sal1_pred)

def Finance(exp1):
    exp_train,exp_test,sal_train,sal_test = train_test_split(exp,Fin_sal,test_size=1/3,random_state = 0)
    regressor = LinearRegression()
    regressor.fit(exp_train,sal_train)
    sal_pred = regressor.predict(exp_test)
    sal1_pred = regressor.predict([[exp1]])
    filename="Finance.sav"
    pickle.dump(regressor,open(filename,"wb"))
    print(sal1_pred)

def Business(exp1):
    exp_train,exp_test,sal_train,sal_test = train_test_split(exp,Bus_sal,test_size=1/3,random_state = 0)
    regressor = LinearRegression()
    regressor.fit(exp_train,sal_train)
    sal_pred = regressor.predict(exp_test)
    sal1_pred = regressor.predict([[exp1]])
    filename="Business.sav"
    pickle.dump(regressor,open(filename,"wb"))
    print(sal1_pred)

def Law(exp1):
    exp_train,exp_test,sal_train,sal_test = train_test_split(exp,Law_sal,test_size=1/3,random_state = 0)
    regressor = LinearRegression()
    regressor.fit(exp_train,sal_train)
    sal_pred = regressor.predict(exp_test)
    sal1_pred = regressor.predict([[exp1]])
    filename="Law.sav"
    pickle.dump(regressor,open(filename,"wb"))
    print(sal1_pred)

def Teaching(exp1):
    exp_train,exp_test,sal_train,sal_test = train_test_split(exp,Teach_sal,test_size=1/3,random_state = 0)
    regressor = LinearRegression()
    regressor.fit(exp_train,sal_train)
    sal_pred = regressor.predict(exp_test)
    sal1_pred = regressor.predict([[exp1]])
    filename="Teaching.sav"
    pickle.dump(regressor,open(filename,"wb"))
    print(sal1_pred)
