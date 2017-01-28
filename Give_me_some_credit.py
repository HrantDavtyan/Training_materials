#imports
import numpy as np
import seaborn as sns
import matplotlib as mlp
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
%matplotlib inline
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_auc_score

#import data anr rename columns
df = pd.read_csv(r'C:\Users\hrant\credit_kaggle_training.csv')
df.columns = ['ID','Default','BalanceShare','Age','#Due30-59','DRatio','MIncome','#Credits','#Due>90','#Mortgage','#Due60-89','#Dependents']

#Default is 1 if #Due>90 is >0, and 0, if the latter is also 0. 
#Thus, they are highly correlated, so we will drop #Due>90
df=df.drop('#Due>90', axis=1)

#missing value handling
#first, lets handle missing values encoded as NaN
df.info()
df['#Dependents'].isnull().values.sum() #there are 3924 NaN objects need to be filled
df['MIncome'].isnull().values.sum() #there are 29731 NaN objects need to be filled

#for the sake of simplicity, we will impute both with mean
df['#Dependents']=df['#Dependents'].fillna(df['#Dependents'].mean())
df['MIncome']=df['MIncome'].fillna(df['MIncome'].mean())

#check whether they are correctly filled
df.info()

#now, let's work on the missing values which are not encoded as NaN
#they can be filled with unusual values like -1, 99, -999 etc.
df['#Due60-89'].value_counts() #probably 98 and 96 are the encoders for missing values
df['#Due30-59'].value_counts() #same 98 and 96 here
df['#Mortgage'].value_counts() #seems we have no missing value here
df['#Credits'].value_counts() #seems we have no missing value here
df['#Dependents'].value_counts() #seems we imputed with noninteger mean
df['#Dependents']=df['#Dependents'].round(0) #so we round it up to the nearest integer

#for the rest of the variables, which have many different values
#we will use graphs, which I have not done
#here, also using graphs and your logic, you must drop or fill
#the abnormal values, like Age=0 or MIncome<0 etc. (if you have it)
#for example, we know that you cant get credit if you are <18
df['Age'][df['Age']<18].value_counts()
#so that one guy should be handled

#round the floats to 2 decimals
df=df.round(2)

#Outlier detection and removal (handle observations over 3 std dev)
#I do it for only 1 variable, you do for the rest
df['Age'][np.abs(df['Age']-df['Age'].mean())>(3*df['Age'].std())].value_counts()

#so we have seeral outliers here must drop or fill like missing values
#you can drop for simplicity but in real projects you must treat them carefully
#if they are outliers for other variables too, then drop
#if outliers only with respect to age, then fill

#now we can do some feature engineering by modifying existing variables to create new ones
df['#Due60-89'].loc[df['#Due60-89']>0]=1
df['#Due60-89'].loc[df['#Due60-89']==0]=0

df['#Due30-59'].loc[df['#Due30-59']>0]=1
df['#Due30-59'].loc[df['#Due30-59']==0]=0

#now the best part, lets do the machine learning
target = df['Default']
data = df.drop('Default', axis=1)

#decrease C to care for overfitting
logit=LogisticRegression(C=0.1)

#split the data and get probabilities
data_train, data_test, target_train, target_test = train_test_split(data,target,test_size=0.5)
logit.fit(data_train,target_train)
prediction = logit.predict_proba(data_test)

#by using the predict_proba method you get probabilities 
#of being 1 and probabilities of being 0 simultaneously
#you need to keep the probabilities of being 1 only, which is 2nd column
prediction_list=[]
for i in range(len(prediction)):
    prediction_list.append(prediction[i][1])

#lets now classify everything below 0.1 as 0
classification=[]
for i in prediction_list:
    if i<0.1:
        classification.append(0)
    else:
        classification.append(1)

#calculate the accuracy
print metrics.accuracy_score(target_test,classification)

#here you need to perform some croos validation and accuracy measurement
#I get only AUC, you do the rest
print metrics.roc_auc_score(target_test,classification)

#Let's now predict the values for final test data
#for the final test you must apply the same changes you did to the train data
ft=pd.read_csv(r'C:\Users\hrant\cs-test.csv')
ft.columns = ['ID','Default','BalanceShare','Age','#Due30-59','DRatio','MIncome','#Credits','#Due>90','#Mortgage','#Due60-89','#Dependents']
drop=['#Due>90','Default']
ft=ft.drop(drop, axis=1)

ft['#Dependents']=ft['#Dependents'].fillna(ft['#Dependents'].mean())
ft['MIncome']=ft['MIncome'].fillna(ft['MIncome'].mean())

ft['#Dependents']=ft['#Dependents'].round(0)

ft=ft.round(2)

ft['#Due60-89'].loc[ft['#Due60-89']>0]=1
ft['#Due60-89'].loc[ft['#Due60-89']==0]=0

ft['#Due30-59'].loc[ft['#Due30-59']>0]=1
ft['#Due30-59'].loc[ft['#Due30-59']==0]=0

#add here other changes if you make to the train data
#for example creating new variables

#lets make the final prediction of probabilities
final_prediction = logit.predict_proba(ft)

#Convert it to list
final_prediction_list=[]
for i in range(len(final_prediction)):
    final_prediction_list.append(final_prediction[i][1])

#lets now classify everything below 0.1 as 0
final_classification=[]
for i in final_prediction_list:
    if i<0.1:
        final_classification.append(0)
    else:
        final_classification.append(1)

submission = pd.DataFrame({"ID": ft["ID"],"Default": final_classification})
submission.to_csv('credit_submission.csv', index=False)
