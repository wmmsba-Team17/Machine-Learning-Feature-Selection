library(glmnet)
library(ISLR)
library(pls)
#a)
rm(list=ls())

set.seed(1)
indices=sample(1:nrow(College),size=0.5*nrow(College))

train <- College[indices,]
test <- College[-indices,]

xtrain <- model.matrix (Apps~.,train)[,-1]
ytrain <- train$Apps

xtest <- model.matrix (Apps~.,test)[,-1]
ytest <- test$Apps

grid <-  10 ^ seq(4, -2, length=100)

#c)

ridge.cv <- cv.glmnet(xtrain,ytrain,alpha=0,thresh=1e-12,lambda = grid)
ridge.fit <- glmnet(xtrain,ytrain,alpha=0,lambda = ridge.cv$lambda.min)
ridge.pred <- predict(ridge.fit,newx=xtest )
ridge.coef <- predict(ridge.fit,type='coefficients',s=ridge.cv$lambda.min)

ridge.cv$lambda.min

ridge.coef

(Ridge_MSE <- mean( (ridge.pred-ytest)^2))

#d)

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

#e)
pcr.fit=pcr(Apps~.,data=train,scale=TRUE,validation='CV')
validationplot(pcr.fit,val.type="MSEP")

pcr.fit$ncomp

pcr.pred <- predict(pcr.fit,newdata = xtest,ncomp = 16)
(PCR_MSE <- mean( (pcr.pred-ytest)^2))

#f)
pls.fit=plsr(Apps~., data=train, scale=TRUE ,
             validation ="CV")

summary (pls.fit)

validationplot(pls.fit)

pls.pred=predict (pls.fit ,xtest,ncomp =10)

(PLSMSE = mean((pls.pred -ytest)^2))

#G)

print(c("Ridge:",Ridge_MSE))
print(c("Lasso:",MSE_lasso))
print(c("PCR", PCR_MSE))
print(c("PLS:",PLSMSE))
