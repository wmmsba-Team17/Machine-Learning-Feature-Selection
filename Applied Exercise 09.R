# Applied Exercise 09 #
#In this exercise, we will predict the number of applications received
#using the other variables in the College data set.
rm(list=ls())

library(glmnet)
library(ISLR)
library(pls)
#a)Split the data set into a training set and a test set.

set.seed(1)
indices=sample(1:nrow(College),size=0.5*nrow(College))

train <- College[indices,]
test <- College[-indices,]

xtrain <- model.matrix (Apps~.,train)[,-1]
ytrain <- train$Apps

xtest <- model.matrix (Apps~.,test)[,-1]
ytest <- test$Apps

grid <-  10 ^ seq(4, -2, length=100)

#c)Fit a ridge regression model on the training set, with λ chosen
#by cross-validation. Report the test error obtained.
ridge.cv <- cv.glmnet(xtrain,ytrain,alpha=0,thresh=1e-12,lambda = grid)
ridge.fit <- glmnet(xtrain,ytrain,alpha=0,lambda = ridge.cv$lambda.min)
ridge.pred <- predict(ridge.fit,newx=xtest )
ridge.coef <- predict(ridge.fit,type='coefficients',s=ridge.cv$lambda.min)

ridge.cv$lambda.min

ridge.coef

(Ridge_MSE <- mean( (ridge.pred-ytest)^2))

#d)Fit a lasso model on the training set, with λ chosen by crossvalidation.
#Report the test error obtained, along with the number
#of non-zero coefficient estimates.

lasso.mod <- glmnet(xtrain,ytrain,alpha=1,lambda=grid)
plot(lasso.mod)

cv.out <- cv.glmnet(xtrain, ytrain, alpha =1, lambda=grid)
plot(cv.out)

bestlam <- cv.out$lambda.min
lasso.pred <- predict(lasso.mod, s=bestlam, newx=xtest)

(MSE_lasso <- mean((lasso.pred-ytest)^2))

# Recreate model with best lambda
lasso2 <- glmnet(xtrain,ytrain,alpha=1,lambda=bestlam)

lasso.coef <- predict(lasso2 ,type="coefficients", s=bestlam)[1:18,]
lasso.coef
lasso.coef[lasso.coef!=0]

#e)Fit a PCR model on the training set, with M chosen by crossvalidation.
#Report the test error obtained, along with the value
#of M selected by cross-validation.
pcr.fit=pcr(Apps~.,data=train,scale=TRUE,validation='CV')
validationplot(pcr.fit,val.type="MSEP")

pcr.fit$ncomp

pcr.pred <- predict(pcr.fit,newdata = xtest,ncomp = 16)
(PCR_MSE <- mean( (pcr.pred-ytest)^2))

#f) Fit a PLS model on the training set, with M chosen by crossvalidation.
#Report the test error obtained, along with the value
#of M selected by cross-validation.
pls.fit=plsr(Apps~., data=train, scale=TRUE ,
             validation ="CV")

summary (pls.fit)

validationplot(pls.fit)

pls.pred=predict (pls.fit ,xtest,ncomp =10)

(PLSMSE = mean((pls.pred -ytest)^2))

#G) Compare different methods

print(c("Ridge:",Ridge_MSE))
print(c("Lasso:",MSE_lasso))
print(c("PCR", PCR_MSE))
print(c("PLS:",PLSMSE))
