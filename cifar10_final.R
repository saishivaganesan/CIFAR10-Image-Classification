#NOTE: The script contains training the models for the top 3 methods and final model.
#For each of these three models, training and validation error rates are calculated.
#To generate final model to be run on new data, start from line 115.
#If running the final model only, run code from line 115.
#If running code for one of the top 3 models, lines 7 to 43 must be run.

# Loading the data into dataframe from given lists of images.rgb and images.lab
x.df = data.frame(matrix(unlist(images.rgb), nrow=length(images.rgb), byrow=T))
y.df = data.frame(matrix(unlist(images.lab), nrow=length(images.lab), byrow=T))
data.df = cbind(x.df, y.df)

#Renaming target column
names(data.df)[3073] = "Target"

#Train and test data split
set.seed(10)
row_count = nrow(data.df)
shuffled_rows = sample(row_count)
train = data.df[head(shuffled_rows,floor(row_count*0.8)),]
test = data.df[tail(shuffled_rows,floor(row_count*0.2)),]
train_X = train[,-3073]
train_Y = train[,3073]
test_X = test[,-3073]
test_Y = test[,3073]

#Principal Component Analysis on training data set
pr.out = prcomp(train_X, scale=TRUE)
pr.var = pr.out$sdev ^2
pve = pr.var/sum(pr.var)

#Plotting scree and cum PVE plot
plot(pve , xlab=" Principal Component ", ylab=" Proportion of Variance Explained ", ylim=c(0,1), type="b")
plot(cumsum (pve), xlab=" Principal Component ", ylab ="Cumulative Proportion of Variance Explained ", ylim=c(0,1), type="b")

#Variance explained by the first 220 Principal components in training set - 95%
sprintf("Cumulative PVE explained by first 220 components: %g", sum(pve[1:220]) * 100)

#Extracting first 220 PCA components for train set observations
train_PCA.df <- data.frame(pr.out$x[,1:220])

#Extracting first 220 PCA components for validation set observations
test_PCA.df <- predict(pr.out, newdata = test_X)
test_PCA.df <- data.frame(test_PCA.df[,1:220])

#Training linear SVM (SVC)
set.seed(10)
svmlin.fit = svm(as.factor(train_Y)~., data=train_PCA.df, kernel='linear', cost=0.1)

#Training set results
svmlin.train.predict = predict(svmlin.fit, train_PCA.df)
svmlin.train.table = table(Predicted=svmlin.train.predict, Target=train_Y)
svmlin.train.res = 0
for (i in c(1:10)){
  svmlin.train.res = svmlin.train.res + svmlin.train.table[i,i]
}
sprintf("'Linear SVC Train Accuracy is: %g", (svmlin.train.res/dim(train_PCA.df)[1])*100)

#Validation set results
svmlin.val.predict = predict(svmlin.fit, test_PCA.df)
svmlin.val.table = table(Predicted=svmlin.val.predict, Target=test_Y)
svmlin.val.res = 0
for (i in c(1:10)){
  svmlin.val.res = svmlin.val.res + svmlin.val.table[i,i]
}
sprintf("Linear SVC Validation Accuracy is: %g", (svmlin.val.res/dim(test_PCA.df)[1])*100)

#Training QDA classifier
set.seed(10)
qda.fit = qda(as.factor(train_Y)~., data=train_PCA.df)

#Training set results
qda.train.pred = predict(qda.fit, train_PCA.df)
qda.train.table = table(Predicted = qda.train.pred$class, Target=train_Y)
qda.train.res = 0
for (i in c(1:10)){
  qda.train.res = qda.train.res + qda.train.table[i,i]
}
sprintf("QDA classifier Train Accuracy is: %g", (qda.train.res/dim(train_PCA.df)[1])*100)

#Validation set results
qda.val.pred = predict(qda.fit, newdata=test_PCA.df)
qda.val.table = table(Predicted=qda.val.pred$class, Target=test_Y)
qda.val.res = 0
for (i in c(1:10)){
  qda.val.res = qda.val.res + qda.val.table[i,i]
}
sprintf("QDA classifier Validation Accuracy is: %g", (qda.val.res/dim(test_PCA.df)[1])*100)

#Training with Boosting
library(gbm)
set.seed(10)
boost.fit = gbm(formula=as.factor(train_Y)~.,data=train_PCA.df,distribution="multinomial",n.trees=4000,interaction.depth=4,shrinkage=0.005)

#Training set results
boost.train.predprob = predict(boost.fit,newdata=train_PCA.df,type="response", n.trees=4000)
boost.train.pred = apply(boost.train.predprob, 1, which.max)
boost.train.table = table(Predicted = boost.train.pred, Target = train_Y)
boost.train.res = 0
for(i in 1:10){
  boost.train.res = boost.train.res + boost.train.table[i,i]
}
sprintf("Boosting classifier Train Accuracy is: %g", (boost.train.res/dim(train_PCA.df)[1])*100)

#Validation set results
boost.val.predprob = predict(boost.fit,newdata = test_PCA.df,type="response",n.trees=2000)
boost.val.pred = apply(boost.val.predprob, 1, which.max)
boost.val.table = table(Predicted = boost.val.pred, Target = test_Y)
boost.val.res = 0
for(i in 1:10){
  boost.val.res = boost.val.res + boost.val.table[i,i]
}
sprintf("Boosting classifier Validation Accuracy is: %g", (boost.val.res/dim(test_PCA.df)[1])*100)


#Training the final model to be run on new data
#This model is trained using the complete data set
#After testing various models, we select QDA for training

# Loading the data into dataframe from given lists of images.rgb and images.lab
x.df = data.frame(matrix(unlist(images.rgb), nrow=length(images.rgb), byrow=T))
y.df = data.frame(matrix(unlist(images.lab), nrow=length(images.lab), byrow=T))
data.df = cbind(x.df, y.df)

#Renaming target column
names(data.df)[3073] = "Target"

# Target variable in full train set
full_y = data.df[,3073]

#Principal Component Analysis on complete data set
pr_full.out = prcomp(x.df, scale=TRUE)
pr_full.var = pr_full.out$sdev ^2
pve_full = pr_full.var/sum(pr_full.var)

#Variance explained by the first 220 Principal components in complete set - 95%
sprintf("Cumulative PVE explained by first 220 components: %g", sum(pve_full[1:220]) * 100)

#Extracting first 220 PCA components for full set of observations
PCA_full.df = data.frame(pr_full.out$x[,1:220])

#Training QDA on complete set
library(MASS)
set.seed(10)
qda_full.fit = qda(as.factor(full_y)~., data=PCA_full.df)

#Training set results for complete set - QDA
qda_full.pred = predict(qda_full.fit, PCA_full.df)
qda_full.table = table(Predicted = qda_full.pred$class, Target=full_y)
qda_full.res = 0
for (i in c(1:10)){
  qda_full.res = qda_full.res + qda_full.table[i,i]
}
sprintf("QDA classifier Accuracy is: %g", (qda_full.res/dim(PCA_full.df)[1])*100)

###Running QDA (best model) on given test data

# Loading the data into dataframe from lists test.images.rgb and test.images.lab
testset_x.df = data.frame(matrix(unlist(test.images.rgb), nrow=length(test.images.rgb), byrow=T))
testset_y.df = data.frame(matrix(unlist(test.images.lab), nrow=length(test.images.lab), byrow=T))
testset_data.df = cbind(testset_x.df, testset_y.df)

# Target variable
testset_y = testset_data.df[,3073]

#Extracting first 220 PCA components for test set observations
testset_PCA.df <- predict(pr_full.out, newdata = testset_x.df)
testset_PCA.df <- data.frame(testset_PCA.df[,1:220])

#Predicting on test data provided
qda.test.pred = predict(qda_full.fit, newdata=testset_PCA.df)
qda.test.table = table(Predicted=qda.test.pred$class, Target=testset_y)
qda.test.res = 0
for (i in c(1:10)){
  qda.test.res = qda.test.res + qda.test.table[i,i]
}
sprintf("QDA classifier Test Set Accuracy is: %g", (qda.test.res/dim(testset_PCA.df)[1])*100)

