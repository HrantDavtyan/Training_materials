#Ներմուծենք որոշ անհրաժեշտ գրադարաններ
import numpy as np #թվաբանական գործողությունների համար
import pandas as pd #տվյալների հետ աշխատելու համար
from pandas import DataFrame
import seaborn as sns #վիզուալիզացիայի համար
import matplotlib as mlp
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.cross_validation import train_test_split #train/test բաժանման համար

#մեքենայական ուսուցման համար
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#գնահատականների ճշգրտության ստուգման համար
from sklearn import metrics

#մոդելների կարճ անվանումների վերագրում
logit = LogisticRegression()
svm_model=SVC()
tree_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier(n_estimators=100)
gb_model = GradientBoostingClassifier()

#Տվյալների ներմուծում
df = pd.read_csv(r'C:\Users\hrant\train.csv') #փոխեք Ձեր ֆայլի հասցեով

#Առաջին 10 արժեքների դիտում
df.head(10)

#Ազատվում ենք այն անկախ փոփոխականներից, որոնք մեր կարծիքով չեն անզդում կախյալ փոփոխականի վրա
drop = ['Name','Fare','Embarked','PassengerId','Ticket','Cabin']
df=df.drop(drop, axis=1) #axis=1 ցույց է տալիս, որ drop ենք անում, նյուներ, այլ ոչ թե տողեր

#Սեռը male, female -ից դարձնում ենք 1 ու 0
df['Sex']=np.where(df['Sex']=='male',1,0)

#Age փոփոխականի բացակայող արժեքները (Missing value) լցնում ենք միջինով (mean)
df["Age"].fillna(df["Age"].mean(), inplace=True)

#Ընտրում ենք մեր կախյալ և անկախ փոփոխականները
target = df['Survived']
data = df.drop('Survived', axis=1)

#Ստեղծում ենք նոր լիստ, որի արժեքները հետագայում կօգտագործվեն որպես test_size
list=[0.2,0.25,0.3,0.35,0.4]

#ստեղծում ենք դատարկ լիստեր, որոնց հետգայում կկպցնենք յուրաքանչյուրի ճշգրտության արժեքը
logit_accuracy=[]
svm_accuracy=[]
dt_accuracy=[]
rf_accuracy=[]
gb_accuracy=[]

#օգտագործում ենք for հրամանը, որպեսզի նույն գործուղություները կիրառենք տարբեր test_size-երի համար

#լիստի ամեն մի արժեքի համար
for i in list:
    # բաժանիր տվյալները train/test հատվածների, որպես test_size ընտրելով վերոնշյալ լիստի արժեքները
    data_train, data_test, target_train, target_test = train_test_split(data,target,test_size=i)
    #ամեն բաժանման համար ստացիր լոգիստիկ դասակարգիչը՝ հիմնվելով միայն train տվյալների վրա
    logit.fit(data_train,target_train)
    #օգտագործելով ստացված դասակարգիչը կանխատեսիր՝ հիմնվելով test տվյլաների վրա
    prediction1= logit.predict(data_test)
    #համեմատիր կանխատեսումն իրականության հետ, ստացված ճշգրտության արդյունքը կպցրու համապատսխան լիստին
    logit_accuracy.append(metrics.accuracy_score(target_test,predictio1))
    #նույնը հաջորդ մոդելների համար
    svm_model.fit(data_train,target_train)
    prediction2= svm_model.predict(data_test)
    svm_accuracy.append(metrics.accuracy_score(target_test,prediction2))
    
    tree_model.fit(data_train,target_train)
    prediction3 = tree_model.predict(data_test)
    dt_accuracy.append(metrics.accuracy_score(target_test,prediction3))
    
    rf_model.fit(data_train,target_train)
    prediction4 = rf_model.predict(data_test)
    rf_accuracy.append(metrics.accuracy_score(target_test,prediction4))
    
    gb_model.fit(data_train,target_train)
    prediction5 = gb_model.predict(data_test)
    gb_accuracy.append(metrics.accuracy_score(target_test,prediction5))
    
#Ճշգրտության արդյունքները ներմուծում ենք  աղուսյակի մեջ
dataframe_acc=DataFrame(logit_accuracy,columns=['Logit'],index=list)

dataframe_acc['SVM']=svm_accuracy
dataframe_acc['DT']=dt_accuracy
dataframe_acc['RF']=rf_accuracy
dataframe_acc['GB']=gb_accuracy
dataframe_acc['NB']=nb_accuracy

#նայում ենք աղուսյակը
dataframe_acc

#Cross Validation իրականացման համար ներմուծենք համապատասխան գրադարանը
from sklearn.cross_validation import cross_val_score

#Տպենք միջին ցուցանիշը
'''Առաջին արգումենտը մոդելի անվանումն է, որի համար իրականացնում ենք Cross Validation։
Երկրորդ արգումենտը անկախ փոփոխականներն են, երրորդը՝ կախյալ։
Վերջին արգումենտը Cross Validation-ի բաժանումների քանակն է'''
print np.mean(cross_val_score(logit, data, target, cv=10))

#Ստանանք միջին ցուցանիշը բոլոր մոդելների համար և նեկայացնենք DataFrame-ի տեսքով
cv_scores=[]
models = [logit, svm_model, tree_model, rf_model, gb_model, nb_model]
for i in models:
    cv_scores.append(np.mean(cross_val_score(i, data, target, cv=10)))

dataframe_cv=DataFrame(cv_scores,columns=['CV scores'])
dataframe_cv
