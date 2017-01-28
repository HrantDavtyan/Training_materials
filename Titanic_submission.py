#Ներբեռնենք իրական Test-ը
final_test = pd.read_csv(r'C:\Users\hrant\test.csv')

#Բեռնենք տվյալները նույն ձևափոխված տեսքին
final_test=final_test.drop(drop, axis=1)
final_test['Sex']=np.where(final_test['Sex']=='male',1,0)
final_test["Age"].fillna(final_test["Age"].mean(), inplace=True)

# Կանխատեսենք իրական Test -ի հիման վրա
#final_model-ի փոխարեն գրեք այն մոդելը, որը ցանկանում եք օգտագործել
predictions = final_model.predict(final_test)

# Ստեղծենք CSV ֆայլ՝ kaggle submit անելու համար
submission = pd.DataFrame({"PassengerId": final_test["PassengerId"],"Survived": predictions})
submission.to_csv('titanic_submission.csv', index=False)
