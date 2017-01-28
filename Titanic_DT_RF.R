###############################################################################
#   Predicting Titanic Survival using Decision Trees and Random Forest in R   #
###############################################################################

#Set the working directory
setwd('/home/hrant/Downloads')

#Import the data
df <- read.csv('train.csv',stringsAsFactors = FALSE)

#View the 1st 4 rows of the data
head(df,4)

#Tabulate a column from the data
table(df$Survived)

#Summarize a variable
summary(df$Age)

#Get the proportions
prop.table(table(df$Survived))

#Get the proportions for female who survived vs not survived (that is why we use ,1)
prop.table(table(df$Sex,df$Survived),1)

#Recode the variable Sex
df$Sex[df$Sex=='female']=0
df$Sex[df$Sex=='male']=1

#Drop the variables that are not useful
#First, choose the columns to drop
to_drop <- c(3,4,9,10,11,12)

#Then use the subset function to create a subset without columns in to_drop
df<-subset(df,select=-to_drop)

#Summarize the variable Age
summary(df$Age)

#Visualize Missing Values
library(Amelia)
missmap(df, main = "Missing values vs observed")

#Impute Age with median value
#na.rm lets us to calculate the median of all values that are non-missing
df$Age[is.na(df$Age)]=median(df$Age, na.rm=TRUE)

#Train/test split with 70% train size
library(caret)
intrain<-createDataPartition(y=df$Survived,p=0.7,list=FALSE)
training<-m_train[intrain,]
testing<-m_train[-intrain,]

#Perform logit
model_logit <- glm(Survived ~.,family=binomial(link='logit'),data=training)

#Summary the fitting results
summary(model_logit)

#Predict Survival probabilities on the test dataset
proba_logit <- predict(model_logit,newdata=testing,type='response')

#Set the cut-off probability to 0.5 and get predictions
pred_logit <- ifelse(proba_logit > 0.5,1,0)

#Test the model accuracy
misClasificError <- mean(pred_logit != test$Survived)
print(paste('Accuracy',1-misClasificError))

#Plot the Confusion matrix
library(caret)
confusionMatrix(data=fitted.results, reference=test$Survived)

#Print the ROC curve
library(ROCR)
p <- predict(model, newdata=testing)
pr <- prediction(p, test$Survived)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

#Get AUC, i.e. the area under the ROC curve
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

#Perform Decision Trees
#First, import the appropriate library
library(rpart)

#Fit the model
tree_model <- rpart(Survived ~ Sex + Age + SibSp + Parch, data=train, method='class')
plot(tree_model)
text(tree_model)

#Install some packages for better analysis
install.packages('rattle')
install.packages('rpart.plot')
install.packages('RColorBrewer')

#Import installed packages
library(rattle)
library(rpart.plot)
library(RColorBrewer)

#Render the fitted tree
fancyRpartPlot(tree_model)

#Make prediction
Prediction <- predict(tree_model, test, type='class')

#To access the parameters of the decision tree
?rpart.control

#To get a more interactive tree with option to click on nodes to remove
interactive_tree <- prp(tree_model,snip=TRUE)$obj
fancyRpartPlot(interactive_tree)

#Install and import Random Forests
install.packages('randomForest')
library(randomForest)

#Fit the model
#The importance=TRUE argument allows us to inspect variable importance as we’ll see, and the ntree argument specifies how many trees we want to grow.
rf_model <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize +FamilyID2, data=train, importance=TRUE, ntree=2000)

#Plot the feature  importances
varImpPlot(tree_model)

#Create submission dataframe and write it to CSV
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit,file='submission.csv',row.names=FALSE)
