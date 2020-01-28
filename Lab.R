# Setup ####

rm(list=ls())

library(glmnet)
library(leaps)
library(ISLR)
library(pls)



# Ridge ####

Hitters <- na.omit(Hitters)  #remove missing values from the data 
str(Hitters) #20 variables
x <- model.matrix(Salary~.,Hitters)[,-1]   
#model.matrix() is useful for creating x, which transforms any qualitative variables into dummy variables. 
#but it will generate intercept column, and that's why we have [,-1] here 
y <- Hitters$Salary

grid <- 10^seq(10,-2,length=100)
ridge.mod <- glmnet(x,y,alpha=0,lambda=grid)  #alpha determines what type of model is fit
#alpha=0 is the ridge penalty, alpha=1 is the lasso penalty
dim(coef(ridge.mod))  #it should be a 20*100, let's check! #gives me 100 sets of coefficient estimates 
#Selecting a good value for lambda is critical! 
ridge.mod$lambda[50]
coef(ridge.mod)[,50]
sqrt(sum(coef(ridge.mod)[-1,50]^2))    #L2 norm coefficients 

#Let's try a different lambda! 
ridge.mod$lambda[60] 
coef(ridge.mod)[,60]                 
sqrt(sum(coef(ridge.mod)[-1,60]^2))    #larger L2 norm coefficients associated with this smaller value of lambda 

pred1 <- predict(ridge.mod, s=50, type='coefficients')[1:20,]
pred2 <- predict(ridge.mod, s=60, type='coefficients')[1:20,]
sqrt(sum(pred1[-1])^2) 
sqrt(sum(pred2[-1])^2) 

set.seed(1)
train <- sample(1: nrow(x), nrow(x)/2)
test <- -train
y.test <- y[test]
ridge.mod <- glmnet(x[train,], y[train], alpha = 0, lambda = grid, thresh = 1e-12) #we can just use defualt, or to set up threshold to get a more precise solution
ridge.pred <- predict(ridge.mod, s = 4, newx = x[test,]) #make prediction on test set this time with an arbitary lambda! 
mean((ridge.pred - y.test) ^ 2)   #calculate test MSE
#Let's try just using mean of training observations instead of running regression model 
mean((mean(y[train])-y.test)^2)
#Calculate the MSE when lambda is a very large number, and beta towards zero on test set
ridge.pred <- predict(ridge.mod, s = 1e10, newx = x[test, ]) #we chould also get the same result by using a very large lambda
mean((ridge.pred - y.test)^2)
#What if lambda is zero? 
ridge.pred <- predict(ridge.mod, s = 0.01, newx = x[test, ], exact = T)
mean((ridge.pred - y.test)^2)

#Compare the coefficients created by lm() and glmnet(), and they wil be almost same 
lm(y ~ x, subset = train)
predict(ridge.mod, s = 0.01, exact = T, type = "coefficients")[1:20, ]
#Choose lambda with Cross-Validation
set.seed(1)
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 0)  #this perform 10-fold cv by default 
plot(cv.out)
bestlam <- cv.out$lambda.min
bestlam
abline(v=log(bestlam),col="purple")
#Test MSE associated with the best lambda
ridge.pred <- predict(ridge.mod, s = bestlam, newx = x[test, ])
mean((ridge.pred - y.test) ^ 2)
#Fit ridge regression model on the full data set, with the lambda chosen by cross-validation
out <- glmnet(x, y, alpha = 0)
predict(out, s = bestlam, type = 'coefficients')[1:20,]
#All 19 coefficients are non-zero. So no predictors are excluded by running ridge regression!


# Lasso ####


### Calling in data and create test and train

Hitters <-  na.omit(Hitters)
x <- model.matrix(Salary~., Hitters)[,-1]
y <-  Hitters$Salary

set.seed(1)
train <- sample(1:nrow(x), nrow(x)/2)
test <- (-train)
y.test <- y[test]


##### The Lasso

# In order to ???t a lasso model, we once again use the 
# glmnet() function; however, this time we use the argument alpha=1. 
# Other than that change, we proceed just as we did in ???tting a ridge model.

grid <- 10^seq(10,-2, length=100)
lasso.mod <- glmnet(x[train, ], y[train], alpha = 1, lambda = grid)
plot(lasso.mod)

# We now perform cross-validation
# and compute the associated test error.

set.seed(1)
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 1)
plot(cv.out)

bestlam <- cv.out$lambda.min
lasso.pred <- predict(lasso.mod, s = bestlam, newx = x[test, ])
MSE_lasso <- mean((lasso.pred - y.test)^2)

print(paste("The Lasso MSE is", round(MSE_lasso, 4)))

# This is substantially lower than the test set MSE of the null model and of least squares, 
# and very similar to the test MSE of ridge regression with ?? chosen by cross-validation. 
# However, the lasso has a substantial advantage over ridge regression in that the resulting
# coefficient estimates are sparse. Below shows how some coefficients are zero.

out <- glmnet(x,y,alpha=1,lambda=grid)
lasso.coef <-  predict(out ,type="coefficients",s=bestlam)[1:20,]
lasso.coef
lasso.coef[lasso.coef!=0]

# PCR ####

set.seed(1)

#We will now apply the PCR function to the Hitters data, in order to predict salary
#Make sure missing data has been removed
#setting validation = 'CV' causes pcr to compute the ten fold cross validation error for each possible value of m (the number of principal components used)
pcr.fit=pcr(Salary~., data=Hitters, scale= TRUE,validation='CV')

summary(pcr.fit)
#The CV score is provided for each possible number of components
#pcr reports the root mean squared error; in order to obtain the usual MSE, we must square this quantity

#plot the cross validation scores using the validationplot function
#Using val.type='MSEP' will cause the cross validation MSE to be plotted
validationplot(pcr.fit,val.type = 'MSEP')

#we see that the smallest cross validation error occurs when m = 18 
#This is barely less than m=19 which amounts to simply performing least squares
#Becaue when all of the components are used in Pcr no dimension reduction occurs
#from the plot , we also see that the cross-validation error is roughly the same when only one
#component is included in the model. This suggests that a model that uses
#just a small number of components might suffice


Hitters <-  na.omit(Hitters)
x <- model.matrix(Salary~., Hitters)[,-1]
y <-  Hitters$Salary

set.seed(1)
train <- sample(1:nrow(x), nrow(x)/2)
test <- (-train)
y.test <- y[test]


#perform pcr using train and test set
pcr.fit1 = pcr(Salary~.,data=Hitters, subset = train, scale = TRUE, validation='CV')
validationplot(pcr.fit1,val.type = 'MSEP')
summary(pcr.fit1)

pcr.pred = predict(pcr.fit1,x[test,],ncomp = 7)
mean((pcr.pred-y.test)^2)

pcr.fit2=pcr(y~x,scale=TRUE,ncomp=7)
summary(pcr.fit2)

# PLS #####

# First we need to prepare the data and set the seed
x=model.matrix(Salary~.,Hitters )[,-1]
y = Hitters$Salary[!is.na(Hitters$Salary)]
set.seed(1)

# Next we set a train/test split
train=sample (1: nrow(x), nrow(x)/2)
test=(-train)
y.test=y[test]

# Finally we can fit our PLS function and see what value to set m to.
# This is determined by seeing which comps level has the lowest CV value.

pls.fit=plsr(Salary~., data=Hitters , subset=train , scale=TRUE ,
             validation ="CV")

summary (pls.fit)

validationplot(pls.fit)

pls.pred=predict (pls.fit ,x[test ,],ncomp =3)

mean((pls.pred -y.test)^2)

pls.fit=plsr(Salary~., data=Hitters , scale=TRUE , ncomp=3)

summary (pls.fit)
