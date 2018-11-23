################################################
########       Daniel Kaminsky         #########
########  DrivenData Name: NW_DanielK ##########
################################################

# Load packages
library(C50)
library(car)
library(caret)
library(corrplot)
library(dplyr)
library(fpp)
library(forecast)
library(foreach)
library(gbm)
library(glmnet)
library(kernlab)
library(MASS)
library(Matrix)
library(Metrics)
library(mgcv)
library(neuralnet)
library(plyr)
library(pscl) # For Zero Inflated Poisson and Negative Binomial Regressions
library(psych)
library(randomForest)
library(rattle)
library(readr)
library(rpart)

################################################
###### Exploratory Data Analysis (EDA) #########
################################################

# Load data
Train <- read.csv("D:/dengue_features_train.csv", sep = ",")
str(Train) # 'data.frame':	1456 obs. of  24 variables
head(Train)
summary(Train)

TrLabel <- read.csv("D:/dengue_labels_train.csv", sep = ",")
str(TrLabel) # 'data.frame':	1456 obs. of  4 variables
head(TrLabel)
summary(TrLabel)

# Adding the total_cases column from the TrLabel to the Train dataset
Train$total_cases <- TrLabel$total_cases
str(Train) # 'data.frame':	1456 obs. of  25 variables
head(Train)
summary(Train)

# Missing Values - Count per Variable
sapply(Train, function(Train) sum(is.na(Train)))

# Fixing Missing Values with MEDIANs. Select rows where the
# Variable Observation is NA and replace it with MEDIAN
Train$ndvi_ne[is.na(Train$ndvi_ne)==T] <- median(Train$ndvi_ne, na.rm = TRUE)
Train$ndvi_nw[is.na(Train$ndvi_nw)==T] <- median(Train$ndvi_nw, na.rm = TRUE)
Train$ndvi_se[is.na(Train$ndvi_se)==T] <- median(Train$ndvi_se, na.rm = TRUE)
Train$ndvi_sw[is.na(Train$ndvi_sw)==T] <- median(Train$ndvi_sw, na.rm = TRUE)
Train$precipitation_amt_mm[is.na(Train$precipitation_amt_mm)==T] <- median(Train$precipitation_amt_mm, na.rm = TRUE)
Train$reanalysis_air_temp_k[is.na(Train$reanalysis_air_temp_k)==T] <- median(Train$reanalysis_air_temp_k, na.rm = TRUE)
Train$reanalysis_avg_temp_k[is.na(Train$reanalysis_avg_temp_k)==T] <- median(Train$reanalysis_avg_temp_k, na.rm = TRUE)
Train$reanalysis_dew_point_temp_k[is.na(Train$reanalysis_dew_point_temp_k)==T] <- median(Train$reanalysis_dew_point_temp_k, na.rm = TRUE)
Train$reanalysis_max_air_temp_k[is.na(Train$reanalysis_max_air_temp_k)==T] <- median(Train$reanalysis_max_air_temp_k, na.rm = TRUE)
Train$reanalysis_min_air_temp_k[is.na(Train$reanalysis_min_air_temp_k)==T] <- median(Train$reanalysis_min_air_temp_k, na.rm = TRUE)
Train$reanalysis_precip_amt_kg_per_m2[is.na(Train$reanalysis_precip_amt_kg_per_m2)==T] <- median(Train$reanalysis_precip_amt_kg_per_m2, na.rm = TRUE)
Train$reanalysis_relative_humidity_percent[is.na(Train$reanalysis_relative_humidity_percent)==T] <- median(Train$reanalysis_relative_humidity_percent, na.rm = TRUE)
Train$reanalysis_sat_precip_amt_mm[is.na(Train$reanalysis_sat_precip_amt_mm)==T] <- median(Train$reanalysis_sat_precip_amt_mm, na.rm = TRUE)
Train$reanalysis_specific_humidity_g_per_kg[is.na(Train$reanalysis_specific_humidity_g_per_kg)==T] <- median(Train$reanalysis_specific_humidity_g_per_kg, na.rm = TRUE)
Train$reanalysis_tdtr_k[is.na(Train$reanalysis_tdtr_k)==T] <- median(Train$reanalysis_tdtr_k, na.rm = TRUE)
Train$station_avg_temp_c[is.na(Train$station_avg_temp_c)==T] <- median(Train$station_avg_temp_c, na.rm = TRUE)
Train$station_diur_temp_rng_c[is.na(Train$station_diur_temp_rng_c)==T] <- median(Train$station_diur_temp_rng_c, na.rm = TRUE)
Train$station_max_temp_c[is.na(Train$station_max_temp_c)==T] <- median(Train$station_max_temp_c, na.rm = TRUE)
Train$station_min_temp_c[is.na(Train$station_min_temp_c)==T] <- median(Train$station_min_temp_c, na.rm = TRUE)
Train$station_precip_mm[is.na(Train$station_precip_mm)==T] <- median(Train$station_precip_mm, na.rm = TRUE)

# Missing Values - Count per Variable
sapply(Train, function(Train) sum(is.na(Train)))

# Histogram, Q-Q and Box Plots of total_cases
par(mfrow = c(2, 2), mar = c(5.1, 6.1, 4.1, 2.1))
hist(Train$total_cases, col = "Gold", main = "Histogram - total_cases", xlab = "total_cases")
qqnorm(Train$total_cases, col = "darkblue", pch = 'o', main = "Q-Q Plot - total_cases")
qqline(Train$total_cases, col = "darkred", lty = 2, lwd = 3)
boxplot(Train$total_cases, col = "red", pch = 16,
        main = "Box Plot - total_cases")
par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1))

# Stats
quantile(Train$total_cases, c(.01, .05, .10, .25, .50,  .75, .90, .95, .99))

# Frequency Count
q99 = count(Train$total_cases>236.75)
q95 = count(Train$total_cases>81.25)
q90 = count(Train$total_cases>56)
q75 = count(Train$total_cases>28)
q99
q95
q90
q75
zeroCount = count(Train$total_cases==0)
zeroCount

# Capping Variables (Used only is some of the models)
# Train[Train$total_cases > 28, "total_cases"] = 28
# quantile(Train$total_cases, c(.01, .05, .10, .25, .50,  .75, .90, .95, .99))

# Choosing a subset of Variables 
myvars <- names(Train) %in% c("ndvi_nw","ndvi_sw","reanalysis_min_air_temp_k",
                                "reanalysis_air_temp_k","ndvi_ne","station_min_temp_c",
                                "ndvi_se","station_diur_temp_rng_c","reanalysis_tdtr_k",
                                "reanalysis_avg_temp_k","reanalysis_specific_humidity_g_per_kg",
                                "reanalysis_dew_point_temp_k","precipitation_amt_mm","station_avg_temp_c",
                                "reanalysis_max_air_temp_k","reanalysis_relative_humidity_percent")
SetForCor <- Train[myvars]

# Correlation Matrix
str(SetForCor)
jittered_x <- sapply(SetForCor, jitter)
pairs(jittered_x, names(SetForCor), col=(SetForCor$reanalysis_relative_humidity_percent)+1)
cor(SetForCor)

# Choosing another subset of Variables 
myvars2 <- names(Train) %in% c("reanalysis_air_temp_k",
                               "reanalysis_avg_temp_k",
                               "reanalysis_dew_point_temp_k",
                               "reanalysis_specific_humidity_g_per_kg")
SetForCor2 <- Train[myvars2]

# Correlation Matrix
str(SetForCor2)
jittered_x <- sapply(SetForCor2, jitter)
pairs(jittered_x, names(SetForCor2), col="blue4")
# Checking for highly correlated variables
cor(SetForCor2)

# Creating a Subset of Train (TrSet)
TrSet <- Train
str(TrSet) # data.frame':	1456 obs. of  25 variables
TrSet$year <- NULL # Removing the year column
TrSet$weekofyear <- NULL # Removing the weekofyear column
TrSet$week_start_date <- NULL # Removing the week_start_date column
str(TrSet) # data.frame':	1456 obs. of  22 variables
# Stats
quantile(TrSet$total_cases, c(.01, .05, .10, .25, .50,  .75, .90, .95, .99))

################################################
########## Modeling the Train Dataset ##########
################################################

# Model 0 - Base Model Poisson Regression
summary(Model0 <- glm(total_cases ~  ., family="poisson", data=Train)) # AIC: 11316

# Model 1 - Poisson Regression - Stepwise Variable Selection
Model1 <- glm(total_cases ~  ., family="poisson", data=TrSet)
summary(VarSelection <- step(Model1, direction="both")) # AIC: 45072

# Model 2 - Negative Binomial Regression - Stepwise Variable Selection
Model2 <- glm.nb(total_cases ~  ., data=TrSet)
summary(VarSelectionNB <- step(Model2, direction="both")) # AIC: 11534

# Decision Tree of All Variables with rattle()
# rattle()

# Decision Tree
DTree_TrSet = rpart(total_cases ~ .,data=TrSet)
summary(DTree_TrSet)
fancyRpartPlot(DTree_TrSet)

# Model 3 - Poisson with Decision Tree Variable Selection
Model3 <- glm(total_cases ~  ndvi_nw+ndvi_se+reanalysis_avg_temp_k+
                reanalysis_max_air_temp_k+reanalysis_min_air_temp_k+
                reanalysis_relative_humidity_percent, family="poisson", data=TrSet)
summary(Model3) # AIC: 47431

# Model 4 - Negative Binomial with a few Selected Variables Decision Tree
Model4 <- glm.nb(total_cases ~  ndvi_nw+ndvi_se+reanalysis_avg_temp_k+
                reanalysis_max_air_temp_k+reanalysis_min_air_temp_k+
                reanalysis_relative_humidity_percent, data=TrSet)
summary(Model4) # AIC: 11600

# Model 5 - Negative Binomial with Decision Tree Variable Selection
Model5 <- glm.nb(total_cases ~  ndvi_nw+ndvi_sw+reanalysis_min_air_temp_k+
                reanalysis_air_temp_k+ndvi_ne+station_min_temp_c+
                ndvi_se+station_diur_temp_rng_c+reanalysis_tdtr_k+
                reanalysis_avg_temp_k+reanalysis_specific_humidity_g_per_kg+
                reanalysis_dew_point_temp_k+precipitation_amt_mm+station_avg_temp_c+
                city+reanalysis_max_air_temp_k+reanalysis_relative_humidity_percent,
                data=TrSet)
summary(Model5) # AIC: 11539

# Model 6 - Negative Binomial with Decision Tree Variable Selection (2)
Model6 <- glm.nb(total_cases ~  ndvi_nw+ndvi_sw+reanalysis_min_air_temp_k+
                   reanalysis_air_temp_k+ndvi_ne+station_min_temp_c+
                   ndvi_se+station_diur_temp_rng_c+reanalysis_tdtr_k+
                   reanalysis_dew_point_temp_k+precipitation_amt_mm+station_avg_temp_c+
                   city+reanalysis_max_air_temp_k+reanalysis_relative_humidity_percent,
                 data=TrSet)
summary(Model6) # AIC: 11552

# Model 7 - Quasipoisson
Model7 <- gam(total_cases ~ 
              s(reanalysis_min_air_temp_k,k=4) + s(reanalysis_relative_humidity_percent,k=4)
              + s(precipitation_amt_mm,k=4), 
              family=quasipoisson, data=TrSet)

summary(Model7)

# Model 8 - Quasipoisson
Model8 <- gam(total_cases ~  
                s(ndvi_nw,k=4)+s(ndvi_sw,k=4)+s(reanalysis_min_air_temp_k,k=4)+
                s(reanalysis_air_temp_k,k=4)+s(ndvi_ne,k=4)+s(station_min_temp_c,k=4)+
                s(ndvi_se,k=4)+s(station_diur_temp_rng_c,k=4)+s(reanalysis_tdtr_k,k=4)+
                s(reanalysis_avg_temp_k,k=4)+s(reanalysis_specific_humidity_g_per_kg,k=4)+
                s(reanalysis_dew_point_temp_k,k=4)+s(precipitation_amt_mm,k=4)+s(station_avg_temp_c,k=4)+
                s(reanalysis_max_air_temp_k,k=4)+s(reanalysis_relative_humidity_percent,k=4), 
              family=quasipoisson, na.action=na.exclude, data=TrSet)

summary(Model8)

# Model 9 - Neural Network
Model9 <- avNNet(total_cases ~ ndvi_nw+ndvi_sw+reanalysis_min_air_temp_k+
                   reanalysis_air_temp_k+ndvi_ne+station_min_temp_c+
                   ndvi_se+station_diur_temp_rng_c+reanalysis_tdtr_k+
                   reanalysis_avg_temp_k+reanalysis_specific_humidity_g_per_kg+
                   reanalysis_dew_point_temp_k+precipitation_amt_mm+station_avg_temp_c+
                   city+reanalysis_max_air_temp_k+reanalysis_relative_humidity_percent, 
                 data=TrSet, repeats=25, size=3, decay=0.1, linout=TRUE)

summary(Model9)

# Model 10 - NN to TS Data
ts_totCases <- ts(TrSet$total_cases)
Model10 <- nnetar(ts_totCases)
f10 <- forecast(Model9,h=24)
accuracy(f10) # MAE = 4.967374
plot(f10)
f10
summary(f10)

# Model 11 - neuralnet
Model11 <- neuralnet(total_cases ~  ndvi_nw+ndvi_sw+reanalysis_min_air_temp_k+
                       reanalysis_air_temp_k+ndvi_ne+station_min_temp_c+
                       ndvi_se+station_diur_temp_rng_c+reanalysis_tdtr_k+
                       reanalysis_avg_temp_k+reanalysis_specific_humidity_g_per_kg+
                       reanalysis_dew_point_temp_k+precipitation_amt_mm+station_avg_temp_c+
                       reanalysis_max_air_temp_k+reanalysis_relative_humidity_percent,
                     TrSet, hidden = 4, lifesign = "minimal", rep=1,
                     linear.output = FALSE, threshold = 0.1, likelihood=TRUE) # AIC = 3581641.136
plot(Model11, rep="best")

# MAE
# Preparing for MAE metric
Actual <- TrSet$total_cases
Pred_Model0 <- fitted(Model0)
Pred_Model1 <- fitted(Model1)
Pred_Model2 <- fitted(Model2)
Pred_Model3 <- fitted(Model3)
Pred_Model4 <- fitted(Model4)
Pred_Model5 <- fitted(Model5)
Pred_Model6 <- fitted(Model6)
Pred_Model7 <- fitted(Model7)

# MAE library(Metrics)
mae(Actual, Pred_Model0)
mae(Actual, Pred_Model1)
mae(Actual, Pred_Model2)
mae(Actual, Pred_Model3) # 19.93653
mae(Actual, Pred_Model4) # 19.89394
mae(Actual, Pred_Model5) # 19.6816
mae(Actual, Pred_Model6) # 20.04911
mae(Actual, Pred_Model7) # 18.8961

#################################################################
## Spliting TrSet into two sets based on outcome: 80% and 20%  ##
#################################################################
index <- sample(1:nrow(TrSet), 0.80*nrow(TrSet)) 
TrainSet <- TrSet[ index,] # Train dataset 80%
CVSet <- TrSet[-index,] # Cross-Validation dataset 20%
str(TrainSet)
str(CVSet)

#############################################
### Modeling the Cross-Validation Dataset ###
#############################################

# Model 1 - Poisson Regression - Stepwise Variable Selection
Model1 <- glm(total_cases ~  ., family="poisson", data=TrainSet)
summary(VarSelection <- step(Model1, direction="both")) # AIC: 45072

# Model 2 - Negative Binomial Regression - Stepwise Variable Selection
Model2 <- glm.nb(total_cases ~  ., data=TrainSet)
summary(VarSelectionNB <- step(Model2, direction="both")) # AIC: 9202.4

# Model 3 - Poisson with Decision Tree Variable Selection
Model3 <- glm(total_cases ~  ndvi_nw+ndvi_se+reanalysis_avg_temp_k+
                reanalysis_max_air_temp_k+reanalysis_min_air_temp_k+
                reanalysis_relative_humidity_percent, family="poisson", data=TrainSet)
summary(Model3) # AIC: 38714

# Model 4 - Negative Binomial with a few Selected Variables Decision Tree
Model4 <- glm.nb(total_cases ~  ndvi_nw+ndvi_se+reanalysis_avg_temp_k+
                   reanalysis_max_air_temp_k+reanalysis_min_air_temp_k+
                   reanalysis_relative_humidity_percent, data=TrainSet)
summary(Model4) # AIC: 9252.5

# Model 5 - Negative Binomial with Decision Tree Variable Selection (2)
Model5 <- glm.nb(total_cases ~  ndvi_nw+ndvi_sw+reanalysis_min_air_temp_k+
                reanalysis_air_temp_k+ndvi_ne+station_min_temp_c+
                ndvi_se+station_diur_temp_rng_c+reanalysis_tdtr_k+
                reanalysis_avg_temp_k+reanalysis_specific_humidity_g_per_kg+
                reanalysis_dew_point_temp_k+precipitation_amt_mm+station_avg_temp_c+
                city+reanalysis_max_air_temp_k+reanalysis_relative_humidity_percent
                , data=TrainSet)
summary(Model5) # AIC: 9218.2

# Model 6 - Negative Binomial with Decision Tree Variable Selection (2)
Model6 <- glm.nb(total_cases ~  ndvi_nw+ndvi_sw+reanalysis_min_air_temp_k+
                   reanalysis_air_temp_k+ndvi_ne+station_min_temp_c+
                   ndvi_se+station_diur_temp_rng_c+reanalysis_tdtr_k+
                   reanalysis_dew_point_temp_k+precipitation_amt_mm+station_avg_temp_c+
                   city+reanalysis_max_air_temp_k+reanalysis_relative_humidity_percent,
                 data=TrainSet)
summary(Model6)

# Model 7 - Quasipoisson
Model7 <- gam(total_cases ~ 
                s(reanalysis_min_air_temp_k,k=4) + s(reanalysis_relative_humidity_percent,k=4)
              + s(precipitation_amt_mm,k=4), 
              family=quasipoisson, data=TrainSet)

summary(Model7)
coefficients(Model6)

# Model 8 - Quasipoisson
Model8 <- gam(total_cases ~  
                s(ndvi_nw,k=4)+s(ndvi_sw,k=4)+s(reanalysis_min_air_temp_k,k=4)+
                s(reanalysis_air_temp_k,k=4)+s(ndvi_ne,k=4)+s(station_min_temp_c,k=4)+
                s(ndvi_se,k=4)+s(station_diur_temp_rng_c,k=4)+s(reanalysis_tdtr_k,k=4)+
                s(reanalysis_avg_temp_k,k=4)+s(reanalysis_specific_humidity_g_per_kg,k=4)+
                s(reanalysis_dew_point_temp_k,k=4)+s(precipitation_amt_mm,k=4)+s(station_avg_temp_c,k=4)+
                s(reanalysis_max_air_temp_k,k=4)+s(reanalysis_relative_humidity_percent,k=4), 
              family=quasipoisson, na.action=na.exclude, data=TrainSet)

summary(Model8)
coefficients(Model8)

# Model 9 - MLR
TrainSet_LN <- TrainSet
TrainSet_LN$LN_T_Cases <- log(TrainSet_LN$total_cases)
TrainSet_LN$total_cases <- NULL # Removing the TrainSet_LN$total_cases column
head(TrainSet_LN)
# Histogram, Q-Q and Box Plots of LN_T_Cases
par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1))
hist(TrainSet_LN$LN_T_Cases, col = "Gold", main = "Histogram - LN_T_Cases", xlab = "LN_T_Cases")


Model9 <- lm(LN_T_Cases ~ ndvi_nw+ndvi_sw+reanalysis_min_air_temp_k+
               reanalysis_air_temp_k+ndvi_ne+station_min_temp_c+
               ndvi_se+station_diur_temp_rng_c+reanalysis_tdtr_k+
               reanalysis_avg_temp_k+reanalysis_specific_humidity_g_per_kg+
               reanalysis_dew_point_temp_k+precipitation_amt_mm+station_avg_temp_c+
               city+reanalysis_max_air_temp_k+reanalysis_relative_humidity_percent, data=TrainSet_LN)

summary(Model9)

# MAE
# Preparing for MAE metric
Actual <- TrainSet$total_cases
Pred_Model1 <- fitted(Model1)
Pred_Model2 <- fitted(Model2)
Pred_Model3 <- fitted(Model3)
Pred_Model4 <- fitted(Model4)
Pred_Model5 <- fitted(Model5)
Pred_Model6 <- fitted(Model6)
Pred_Model7 <- fitted(Model7)

# MAE library(Metrics)
mae(Actual, Pred_Model1)
mae(Actual, Pred_Model2) # 20.22577
mae(Actual, Pred_Model3) # 19.93653
mae(Actual, Pred_Model4) # 19.89394 TrainSet MAE = 20.39186
mae(Actual, Pred_Model5) # 19.51632
mae(Actual, Pred_Model6) # 20.04911
mae(Actual, Pred_Model7) # 18.8961

###############################################################################
### Testing that the Model didn't have errors and Manually testing the MAE  ###
###############################################################################

# Regression Model 4 NB TrainSet Dataset
TrainSet$LN_CASES <-	(  -74.545938
                     +TrainSet$ndvi_nw *	1.831695
                     +TrainSet$ndvi_se * -0.894609
                     +TrainSet$reanalysis_avg_temp_k * 0.223450
                     +TrainSet$reanalysis_max_air_temp_k * -0.159401
                     +TrainSet$reanalysis_min_air_temp_k * 0.198055
                     +TrainSet$reanalysis_relative_humidity_percent * 0.004328)

# Checking TrainSet
head(TrainSet)
tail(TrainSet)

# Transforming to Probabilities by Exponentiating the LN_CASES variable
#with the natural exponent ("e")
TrainSet$P_CASES <- (exp(TrainSet$LN_CASES))

# MAE by Hand Model 4
mean(abs(TrainSet$total_cases-TrainSet$P_CASES)) # MAE = 20.39186 vs. 20.39078

#############################################
### Scoring the Cross-Validation Dataset ###
#############################################

# Regression Model 4 NB CVSet Dataset
CVSet$LN_CASES <-	(  -74.545938
                     +CVSet$ndvi_nw *	1.831695
                     +CVSet$ndvi_se * -0.894609
                     +CVSet$reanalysis_avg_temp_k * 0.223450
                     +CVSet$reanalysis_max_air_temp_k * -0.159401
                     +CVSet$reanalysis_min_air_temp_k * 0.198055
                     +CVSet$reanalysis_relative_humidity_percent * 0.004328)

# Checking CVSet
head(CVSet)
tail(CVSet)

# Transforming to Probabilities by Exponentiating the LN_CASES variable
#with the natural exponent ("e")
CVSet$P_CASES <- (exp(CVSet$LN_CASES))
CVSet$P_CASES <- (round(CVSet$P_CASES,0))

# MAE by Hand Model 4
mean(abs(CVSet$total_cases-CVSet$P_CASES)) # MAE = 19.14705

# Regression Model 5 Poisson CVSet Dataset
CVSet$LN_CASES <-	(  2.347e+02
                     +CVSet$ndvi_nw *	1.344e+00
                     +CVSet$ndvi_sw *	4.776e-01
                     +CVSet$reanalysis_min_air_temp_k *	1.297e-02
                     +CVSet$reanalysis_air_temp_k *	3.532e-01
                     +CVSet$ndvi_ne *	2.436e-01
                     +CVSet$station_min_temp_c *	1.974e-02
                     +CVSet$ndvi_se *	-3.289e-01
                     +CVSet$station_diur_temp_rng_c *	2.880e-02
                     +CVSet$reanalysis_tdtr_k *	-2.286e-01
                     +CVSet$reanalysis_avg_temp_k *	4.963e-03
                     +CVSet$reanalysis_specific_humidity_g_per_kg *	1.049e+00
                     +CVSet$reanalysis_dew_point_temp_k *	-1.404e+00
                     +CVSet$precipitation_amt_mm *	-9.947e-04
                     +CVSet$station_avg_temp_c *	6.804e-02
                     +outer(CVSet$city == 'sj', 1*(1.795e+00))
                     +CVSet$reanalysis_max_air_temp_k * 1.442e-01
                     +CVSet$reanalysis_relative_humidity_percent * 9.700e-02)

# Checking CVSet
head(CVSet)
tail(CVSet)

# Transforming to Probabilities by Exponentiating the LN_CASES variable
#with the natural exponent ("e")
CVSet$P_CASES <- (exp(CVSet$LN_CASES))
CVSet$P_CASES <- (round(CVSet$P_CASES,0))

# Use summary() to obtain and present descriptive statistics from mydata.
str(CVSet)
summary(CVSet)
head(CVSet)
tail(CVSet)

# CAPPED total_cases (75%) Regression Model 5 Poisson CVSet Dataset
CVSet$LN_CASES <-	(  4.020e+02
                     +CVSet$ndvi_nw *	6.662e-01 
                     +CVSet$ndvi_sw *	2.475e-01
                     +CVSet$reanalysis_min_air_temp_k *	8.086e-02
                     +CVSet$reanalysis_air_temp_k *	8.644e-03
                     +CVSet$ndvi_ne *	1.352e-01
                     +CVSet$station_min_temp_c *	6.597e-02
                     +CVSet$ndvi_se *	-1.349e+00
                     +CVSet$station_diur_temp_rng_c *	7.856e-02
                     +CVSet$reanalysis_tdtr_k *	-2.509e-02
                     +CVSet$reanalysis_avg_temp_k *	-2.014e-01
                     +CVSet$reanalysis_specific_humidity_g_per_kg *	1.612e+00
                     +CVSet$reanalysis_dew_point_temp_k *	-1.282e+00
                     +CVSet$precipitation_amt_mm *	-5.093e-04
                     +CVSet$station_avg_temp_c *	-1.402e-01
                     +outer(CVSet$city == 'sj', 1*(3.458e-01))
                     +CVSet$reanalysis_max_air_temp_k * -2.637e-02
                     +CVSet$reanalysis_relative_humidity_percent * -5.427e-02)

# Checking CVSet
head(CVSet)
tail(CVSet)

# Transforming to Probabilities by Exponentiating the LN_CASES variable
#with the natural exponent ("e")
CVSet$P_CASES <- (exp(CVSet$LN_CASES))
CVSet$P_CASES <- (round(CVSet$P_CASES,0))

# Use summary() to obtain and present descriptive statistics from mydata.
str(CVSet)
summary(CVSet)
head(CVSet)
tail(CVSet)

# Regression Model 8 MLR CVSet Dataset
CVSet$LN_CASES <-	(  3896.81817
                     +CVSet$ndvi_nw *	41.57172
                     +CVSet$ndvi_sw *	-3.22431
                     +CVSet$reanalysis_min_air_temp_k *	-0.19476
                     +CVSet$reanalysis_air_temp_k *	9.35024
                     +CVSet$ndvi_ne *	3.07160
                     +CVSet$station_min_temp_c *	-0.03614
                     +CVSet$ndvi_se *	-6.17212
                     +CVSet$station_diur_temp_rng_c *	-0.76806
                     +CVSet$reanalysis_tdtr_k *	-1.69542
                     +CVSet$reanalysis_avg_temp_k *	-5.10192
                     +CVSet$reanalysis_specific_humidity_g_per_kg *	18.21666
                     +CVSet$reanalysis_dew_point_temp_k *	-21.14427
                     +CVSet$precipitation_amt_mm *	-0.02307
                     +CVSet$station_avg_temp_c *	3.71628
                     +outer(CVSet$city == 'sj', 1*(37.71388))
                     +CVSet$reanalysis_max_air_temp_k * 2.12227
                     +CVSet$reanalysis_relative_humidity_percent * 1.22275)

# Checking CVSet
head(CVSet)
tail(CVSet)

# Transforming to Probabilities by Exponentiating the LN_CASES variable
#with the natural exponent ("e")
CVSet$P_CASES <- round(exp(CVSet$LN_CASES))

################################################
#### Predicting with the Test Dataset      #####
################################################

# Load Data
Test <- read.csv("D:/dengue_features_test.csv", sep = ",")
str(Test) # 'data.frame':	416 obs. of  24 variables
head(Test)
summary(Test)

### Test Dataset Cleansing (Same as what was done to the Train dataset) ###
# Missing Values - Count per Variable
sapply(Test, function(Test) sum(is.na(Test)))

# Fixing Missing Values with MEDIANs. Select rows where the
# Variable Observation is NA and replace it with MEDIAN
Test$ndvi_ne[is.na(Test$ndvi_ne)==T] <- median(Train$ndvi_ne, na.rm = TRUE)
Test$ndvi_nw[is.na(Test$ndvi_nw)==T] <- median(Train$ndvi_nw, na.rm = TRUE)
Test$ndvi_se[is.na(Test$ndvi_se)==T] <- median(Train$ndvi_se, na.rm = TRUE)
Test$ndvi_sw[is.na(Test$ndvi_sw)==T] <- median(Train$ndvi_sw, na.rm = TRUE)
Test$precipitation_amt_mm[is.na(Test$precipitation_amt_mm)==T] <- median(Train$precipitation_amt_mm, na.rm = TRUE)
Test$reanalysis_air_temp_k[is.na(Test$reanalysis_air_temp_k)==T] <- median(Train$reanalysis_air_temp_k, na.rm = TRUE)
Test$reanalysis_avg_temp_k[is.na(Test$reanalysis_avg_temp_k)==T] <- median(Train$reanalysis_avg_temp_k, na.rm = TRUE)
Test$reanalysis_dew_point_temp_k[is.na(Test$reanalysis_dew_point_temp_k)==T] <- median(Train$reanalysis_dew_point_temp_k, na.rm = TRUE)
Test$reanalysis_max_air_temp_k[is.na(Test$reanalysis_max_air_temp_k)==T] <- median(Train$reanalysis_max_air_temp_k, na.rm = TRUE)
Test$reanalysis_min_air_temp_k[is.na(Test$reanalysis_min_air_temp_k)==T] <- median(Train$reanalysis_min_air_temp_k, na.rm = TRUE)
Test$reanalysis_precip_amt_kg_per_m2[is.na(Test$reanalysis_precip_amt_kg_per_m2)==T] <- median(Train$reanalysis_precip_amt_kg_per_m2, na.rm = TRUE)
Test$reanalysis_relative_humidity_percent[is.na(Test$reanalysis_relative_humidity_percent)==T] <- median(Train$reanalysis_relative_humidity_percent, na.rm = TRUE)
Test$reanalysis_sat_precip_amt_mm[is.na(Test$reanalysis_sat_precip_amt_mm)==T] <- median(Train$reanalysis_sat_precip_amt_mm, na.rm = TRUE)
Test$reanalysis_specific_humidity_g_per_kg[is.na(Test$reanalysis_specific_humidity_g_per_kg)==T] <- median(Train$reanalysis_specific_humidity_g_per_kg, na.rm = TRUE)
Test$reanalysis_tdtr_k[is.na(Test$reanalysis_tdtr_k)==T] <- median(Train$reanalysis_tdtr_k, na.rm = TRUE)
Test$station_avg_temp_c[is.na(Test$station_avg_temp_c)==T] <- median(Train$station_avg_temp_c, na.rm = TRUE)
Test$station_diur_temp_rng_c[is.na(Test$station_diur_temp_rng_c)==T] <- median(Train$station_diur_temp_rng_c, na.rm = TRUE)
Test$station_max_temp_c[is.na(Test$station_max_temp_c)==T] <- median(Train$station_max_temp_c, na.rm = TRUE)
Test$station_min_temp_c[is.na(Test$station_min_temp_c)==T] <- median(Train$station_min_temp_c, na.rm = TRUE)
Test$station_precip_mm[is.na(Test$station_precip_mm)==T] <- median(Train$station_precip_mm, na.rm = TRUE)

# Missing Values - Count per Variable
sapply(Test, function(Test) sum(is.na(Test)))

# Regression Model 4 NB Test Dataset
Test$LN_total_cases <-	(  -72.965476
                     +Test$ndvi_nw *	1.580389
                     +Test$ndvi_se * -0.898199
                     +Test$reanalysis_avg_temp_k * 0.144996
                     +Test$reanalysis_max_air_temp_k * -0.112075
                     +Test$reanalysis_min_air_temp_k * 0.224370
                     +Test$reanalysis_relative_humidity_percent * 0.001856)

# Checking Test
head(Test)
tail(Test)

# Transforming to Probabilities by Exponentiating the LN_total_cases variable
#with the natural exponent ("e")
Test$total_cases <- (exp(Test$LN_total_cases))
Test$total_cases <- (round(Test$total_cases,0))

# Preparing the data for writing to csv using the format from the submission format file
OutputTest1 <- read.csv("D:/submission_format.csv", check.names=FALSE)
head(OutputTest1)
str(OutputTest1)
# Creating the Output file
OutputTest1$total_cases <- Test$total_cases
head(OutputTest1)
tail(OutputTest1)
str(OutputTest1)

### Write OutputTest1 to CSV ###
write.csv(OutputTest1, 
          file = "D:/OutputTest1.csv", 
          row.names = FALSE)

# Regression Model 5
Test$LN_total_cases <-	(  2.980e+02
                           +Test$ndvi_nw *	1.391e+00
                           +Test$ndvi_sw *	5.975e-01
                           +Test$reanalysis_min_air_temp_k *	2.617e-02
                           +Test$reanalysis_air_temp_k *	4.985e-01
                           +Test$ndvi_ne *	6.610e-01
                           +Test$station_min_temp_c *	1.721e-02
                           +Test$ndvi_se *	-1.533e+00
                           +Test$station_diur_temp_rng_c *	1.917e-03
                           +Test$reanalysis_tdtr_k *	-6.671e-02
                           +Test$reanalysis_avg_temp_k *	-2.105e-01
                           +Test$reanalysis_specific_humidity_g_per_kg *	1.377e+00
                           +Test$reanalysis_dew_point_temp_k *	-1.433e+00
                           +Test$precipitation_amt_mm *	-1.055e-03
                           +Test$station_avg_temp_c *	-3.197e-02
                           +outer(Test$city == 'sj', 1*(1.420e+00))
                           +Test$reanalysis_max_air_temp_k * 2.292e-02
                           +Test$reanalysis_relative_humidity_percent * 5.023e-02)

# Checking Test
head(Test)
tail(Test)

# Transforming to Probabilities by Exponentiating the LN_total_cases variable
#with the natural exponent ("e")
Test$total_cases <- round(exp(Test$LN_total_cases))

# Preparing the data for writing to csv using the format from the submission format file
OutputTest2 <- read.csv("D:/submission_format.csv", check.names=FALSE)
head(OutputTest2)
str(OutputTest2)
# Creating the Output file
OutputTest2$total_cases <- Test$total_cases
head(OutputTest2)
tail(OutputTest2)
str(OutputTest2)

### Write OutputTest2 to CSV ###
write.csv(OutputTest2, 
          file = "D:/OutputTest2.csv", 
          row.names = FALSE)

# Regression Model 6
Test$LN_total_cases <-	(  -9.617e+01
                          +Test$ndvi_nw *	1.445e+00
                          +Test$ndvi_sw *	7.443e-01
                          +Test$reanalysis_min_air_temp_k *	-2.597e-02
                          +Test$reanalysis_air_temp_k *	6.452e-01
                          +Test$ndvi_ne *	6.471e-01
                          +Test$station_min_temp_c *	4.533e-03
                          +Test$ndvi_se *	-1.642e+00
                          +Test$station_diur_temp_rng_c *	1.038e-03
                          +Test$reanalysis_tdtr_k *	-1.265e-01
                          +Test$reanalysis_dew_point_temp_k *	-3.392e-01
                          +Test$precipitation_amt_mm *	-6.717e-04
                          +Test$station_avg_temp_c *	-3.199e-02
                          +outer(Test$city == 'sj', 1*(1.267e+00))
                          +Test$reanalysis_max_air_temp_k * 2.006e-02
                          +Test$reanalysis_relative_humidity_percent * 1.048e-01)

# Checking Test
head(Test)
tail(Test)

# Transforming to Probabilities by Exponentiating the LN_total_cases variable
#with the natural exponent ("e")
Test$total_cases <- round(exp(Test$LN_total_cases))

# Preparing the data for writing to csv using the format from the submission format file
OutputTest3 <- read.csv("D:/submission_format.csv", check.names=FALSE)
head(OutputTest3)
str(OutputTest3)
# Creating the Output file
OutputTest3$total_cases <- Test$total_cases
head(OutputTest3)
tail(OutputTest3)
str(OutputTest3)

### Write OutputTest3 to CSV ###
write.csv(OutputTest3, 
          file = "D:/OutputTest3.csv", 
          row.names = FALSE)

# CAPPED total_cases (75%) Regression Model 5
Test$LN_total_cases <-	(  3.675e+02
                          +Test$ndvi_nw *	7.937e-01
                          +Test$ndvi_sw *	5.976e-02
                          +Test$reanalysis_min_air_temp_k *	2.572e-02
                          +Test$reanalysis_air_temp_k *	1.346e-01
                          +Test$ndvi_ne *	2.173e-01
                          +Test$station_min_temp_c *	4.185e-02
                          +Test$ndvi_se *	-1.022e+00
                          +Test$station_diur_temp_rng_c *	3.844e-02
                          +Test$reanalysis_tdtr_k *	-3.923e-02
                          +Test$reanalysis_avg_temp_k *	9.762e-02
                          +Test$reanalysis_specific_humidity_g_per_kg *	1.500e+00
                          +Test$reanalysis_dew_point_temp_k *	-1.533e+00
                          +Test$precipitation_amt_mm *	-5.687e-04
                          +Test$station_avg_temp_c *	-1.165e-01
                          +outer(Test$city == 'sj', 1*(8.070e-01))
                          +Test$reanalysis_max_air_temp_k * -5.012e-02
                          +Test$reanalysis_relative_humidity_percent * 3.004e-02)

# Checking Test
head(Test)
tail(Test)

# Transforming to Probabilities by Exponentiating the LN_total_cases variable
#with the natural exponent ("e")
Test$total_cases <- round(exp(Test$LN_total_cases))

# Preparing the data for writing to csv using the format from the submission format file
OutputTest4 <- read.csv("D:/submission_format.csv", check.names=FALSE)
head(OutputTest4)
str(OutputTest4)
# Creating the Output file
OutputTest4$total_cases <- Test$total_cases
head(OutputTest4)
tail(OutputTest4)
str(OutputTest4)
summary(OutputTest4)

### Write OutputTest4 to CSV ###
write.csv(OutputTest4, 
          file = "D:/OutputTest4.csv", 
          row.names = FALSE)




