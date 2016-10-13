###############################################################################################
#                                                                                             #
# 					                                                                          #
# This script tries to analyze lending club data and predict loan outcomes using H2O          #
# Author - Rohit Kapa                                                                         #
#                                                                                             #
###############################################################################################
# The following program explores some of the basic utilities the h2o package and builds 
# advanced machine learning models available with this package

# Clearing the variables in existing sas session
rm(list = ls())

# The following packages are good to have along with h2o
# Installing packages required along with h2o
install.packages("RCurl")
install.packages("bitops")
install.packages("rjson")
install.packages("statmod")

# Base package required
# install.packages("tools")

# Installing h2o package
install.packages("h2o")

# Loading package h2o
library(h2o)

# h2o being a JAVA virtual machine would be required to be initialized
# The following commands would start h2o in background and allow us to use its commands in 
# R environemnt

# Depending upon the source of data, the initialization command would vary


##---local mode ---
# In this mode, the file is present on a local drive

# Starting with default settings
h2o.init()

# Starting with user defined settings
# Below command has an added option of maximum memory size
# This would limit h2o to use only 2Gb of the RAM for its operations
h2o.init(ip = 'localhost', port = 54321, max_mem_size = '2g')

# To obtain the Info about the connection, run the below command
h2o.clusterInfo()

# The h2o virtual machine can be shutted down by using the below command
# However, we would not be able to use the package command any further.
# Hence it is better to run this command when we would close the session.
h2o.shutdown()

## --- cluster mode ----
# In this mode, we can connect R to other clusters like Hadoop Big Data cluster.
# The below command initialized h2o to run on 1 node utilizing 6Gb of Ram.
# The command is commented as we are not connecting R to big data in this project
# hadoop jar h2odriver.jar -nodes 1 -mapperXmx 6g -  output hdfsOutputDirName




# Data can be imported from local machine / Big data cluster into R.
# This import does not actually imports the data. It actually creates a pointer to this data
# It is called import as an environment type of file is created in R which establishes the 
# connection with h2o JVM and hence to the actual file

## To import data
#To import small loan data file from H2O's package:
loanpath = "C:/Users/rohit/Documents/RClassWork/Project1/LoanStats3c.csv"
loan.hex = h2o.importFile(path = loanpath, destination_frame = "loan.hex")


#To import an entire folder of files as one data
#object:
loanpath = "C:/Users/rohit/Documents/RClassWork/Project1"
loan.hex = h2o.importFile(path = loanpath,destination_frame = "loan.hex")
head(loan.hex,10)
nrow(loan.hex)

#To import from HDFS and connect to H2O in R using the
#IP and port of an H2O instance running on your Hadoop cluster:
# h2o.init(ip= <IPAddress>, port =54321, nthreads = -1)
# pathToData = "hdfs://mr-0xd6.h2oai.loc/datasets/loan.csv"
# loan.hex = h2o.importFile(path = pathToData,destination_frame = "loan.hex")

## To export data from R to h2o frame
loanpath = "C:/ your path"
loan.hex = h2o.uploadFile(path = loanpath, destination_frame = "loan.hex")


### h2o Utilities

#In order to convert a H2O based data frame like we created above to R based data frame we can use below command
as.data.frame()
loan.R = as.data.frame(loan.hex)

#In order to convert R based data frame to H2O based data frame
as.h2o()
loan.hex = as.h2o(loan.R,destination_frame = 'loan.hex')

#In order to rename the H2O data frame with other name, also ls() gives list of all H2O based objects
loan.hex <- h2o.assign(data = loan.hex, key = "new_loan.hex")
h2o.ls()

#To see the data structure of any object we can check through below code
str()
str(loan.hex)

#To convert any data frame to a matrix format
loan.M = as.matrix(loan.hex)

# Check class of BankData.hex
class(loan.hex)

# Some of the basic R functions run easily on h20 frames
View(loan.hex)
head(loan.hex)

# Summary of absolute values for all the columns
summary(h2o.abs(loan.hex))
summary(loan.hex, exact_quantiles = TRUE)

## To obtain column names of a h2o frame, use either one of the below 2 commands
colnames(loan.hex)
names(loan.hex)


# Check if any column in h2o frame contains continuous data
h2o.anyFactor(loan.hex)

# Binding 2 data sets into one
bind.hex = h2o.cbind(loan.hex, loan.hex)
# View command below will take a huge amount of time as this h20 frame is very large
View(bind.hex)

#To find the maximum minimum for a variable
min(loan.hex$funded_amnt)
max(loan.hex$funded_amnt)

#To check the quantile based distribution at every 10 percentile
loan.qs <- quantile(loan.hex$funded_amnt, probs = (1:10)/10)
loan.qs

# Checking the histogram of some of the numeric variables
h2o.hist(loan.hex$loan_amnt)
# Data in loan_amt seems to be normal with a few outliers

# Standerdizing the loan_amnt variable
z_loan_amnt = (loan.hex$loan_amnt - mean(loan.hex$loan_amnt)) / sd(loan.hex$loan_amnt, na.rm = T)

# Checking for outliers
outliers_loan_amnt = z_loan_amnt[z_loan_amnt < -3 | z_loan_amnt > 3]
outliers_loan_amnt

# no value in z_loan_amt  is below -3 or greater than 3.
# Thus the seems to be no outliers in this column

summary(loan.hex$loan_amnt)

# Checking the columns having missing data
cbind(colnames(loan.hex), h2o.nacnt(loan.hex))

# We would remove the columns which have a large number of missing values

h2o.nacnt(loan.hex)
names(loan.hex[h2o.nacnt(loan.hex)>(nrow(loan.hex)*0.25)])
names(loan.hex[h2o.nacnt(loan.hex)<(nrow(loan.hex)*0.25)])

# Removing the columns which have an excess of missing data
loan.hex = loan.hex[h2o.nacnt(loan.hex)<(nrow(loan.hex)*0.25)]

colnames(loan.hex)

# Creating a vector of columns used for modeling
req_cols = c("id","addr_state",	"annual_inc",	"application_type",	"total_pymnt", "avg_cur_bal",	"collections_12_mths_ex_med",	"delinq_2yrs",	"dti",	"emp_length",	"emp_title",	"funded_amnt",	"home_ownership",	"inq_last_6mths",	"installment",	"int_rate",	"loan_amnt",	"mort_acc",	"open_acc",	"pub_rec",	"purpose",	"revol_bal",	"revol_util",	"term",	"tot_hi_cred_lim",	"total_acc",	"total_bal_ex_mort",	"verification_status",	"loan_status")

# Creating h2o frame having only the useful columns
loan2.hex = loan.hex[colnames(loan.hex) %in% req_cols]


summary(loan2.hex$loan_status)

# Setting up the target column to have binary values.
# The binary values would be 1 for a bad loan and 0 for a good loan
# The goal of modeling is to predict a bad loan
# If the loan is charged off or default, it would be classified as a bad loan
loan2.hex = loan2.hex[!(loan2.hex$loan_status %in% c("Current", "In Grace Period", "Late (16-30 days)")), ]
loan2.hex = loan2.hex[!is.na(loan2.hex$id),]
loan2.hex$bad_loan = loan2.hex$loan_status %in% c("Charged Off", "Default")
loan2.hex$bad_loan = as.factor(loan2.hex$bad_loan)


sum(!is.na(loan2.hex$id))

nrow(loan2.hex)


# Splitting the data set in training and validation
s <- h2o.runif(loan2.hex)
## Create training set with threshold of 0.8
loan.train <- loan2.hex[s <= 0.8,]
##Assign name to training set
loan.train <- h2o.assign(loan.train, "loan.train")
## Create test set with threshold to filter values greater than 0.8
loan.test <- loan2.hex[s > 0.8,]
## Assign name to test set
loan.test <- h2o.assign(loan.test, "loan.test")
## Combine results of test & training sets, then display result
nrow(loan.train) + nrow(loan.test)

nrow(loan2.hex) ## Matches the full set



# Splits data in prostate data frame with a ratio of 0.75
loan.split <- h2o.splitFrame(data = loan2.hex ,ratios = 0.75)
# Creates training set from 1st data set in split
loan.train <- loan.split[[1]]
# Creates testing set from 2st data set in split
loan.test <- loan.split[[2]]

# Modeling 
# Setting up the targer variable
target_loan = 'bad_loan'

# The predictor variable are as follows
explanatory_loan = c("addr_state","annual_inc","application_type","avg_cur_bal","collections_12_mths_ex_med",	"delinq_2yrs",	"dti",	"emp_length",	"emp_title",	"funded_amnt",	"home_ownership",	"inq_last_6mths",	"installment",	"int_rate",	"loan_amnt",	"mort_acc",	"open_acc",	"pub_rec",	"purpose",	"revol_bal",	"revol_util",	"term",	"tot_hi_cred_lim",	"total_acc",	"total_bal_ex_mort",	"verification_status")

# Building a gradient boosting model
gbm_model_loan <- h2o.gbm(x = explanatory_loan, y = target_loan, training_frame = loan.train, validation_frame = loan.test, model_id = 'gbm_model_loan' ,balance_classes = T, learn_rate = 0.05, score_each_iteration = T, ntrees = 100)

# Performance charactaristics of the model
h2o.performance(gbm_model_loan, newdata = loan.train)

# The area under the curve for the gbm model
h2o.auc(gbm_model_loan)

# Area under the curve for gbm model using validation data
h2o.auc(gbm_model_loan,valid = T)
#gbm_score <- plot_scoring(model = gbm_model_loan)

# The most importnat variables can be found using the varimp command
h2o.varimp(gbm_model_loan)

# Various metrics of gbm model
gbm_model_loan@model$scoring_history
gbm_model_loan@model$training_metrics


# Building a general linear model
glm_model_loan <- h2o.glm(x = explanatory_loan, y = target_loan, training_frame = loan.train, model_id = 'glm_model_loan', family = 'binomial',nfolds=10,alpha = 0.5)

# Performance of glm model
h2o.performance(glm_model_loan, loan.train)

# AUC og glm model
h2o.auc(glm_model_loan)

# Metrics for glm model
glm_model_loan@model$cross_validation_metrics

# Obtaining a confusion matrix
h2o.confusionMatrix(glm_model_loan, loan.train)


# Predictions of a model can be done using the following command
# This would give 3 columns as output.
# Column 1 is the prediction, column 2 is probability(p), column 3 is 1 - p
h2o.predict(object = glm_model_loan,newdata = loan.test$bad_loan)



# Deep learning model using h2o
# Takes huge amount of time to run with default settings.
deeplearn_model_loan <- h2o.deeplearning(x = explanatory_loan, y = target_loan, training_frame = loan.train,validation_frame = loan.test)


# Model Assessment
# Calculating the total revenue made by the loans repayment.
# Taking into account the charged off or unpaid loans
loan.train$Total_revenue = loan.train$total_pymnt - loan.train$loan_amnt
colnames(loan.train)

loan.test$Total_revenue = loan.test$total_pymnt - loan.test$loan_amnt

# Adding predictions to the loan.train h2o frame
# Predictions of a model can be done using the following command
# Gradient Boosting Model
loan.train$gbm_predictions = h2o.predict(object = gbm_model_loan,newdata = loan.train)[,1]
Revenue_gbm = h2o.group_by(data = loan.train, by = c("bad_loan", "gbm_predictions"), sum("Total_revenue"))
Revenue_gbm.R = as.data.frame(Revenue_gbm)

head(Revenue_gbm.R,10)
# Cost of false Negative : -75101833
Revenue_gbm.R[Revenue_gbm.R$bad_loan == 1 & Revenue_gbm.R$gbm_predictions == 0, 3]

# General Linear Model
loan.train$glm_predictions = h2o.predict(object = glm_model_loan,newdata = loan.train)[,1]
Revenue_glm = h2o.group_by(data = loan.train, by = c("bad_loan", "glm_predictions"), sum("Total_revenue"))
Revenue_glm.R = as.data.frame(Revenue_glm)

head(Revenue_glm.R,10)
# Cost of false Negative : -96894628
Revenue_glm.R[Revenue_glm.R$bad_loan == 1 & Revenue_glm.R$glm_predictions == 0, 3]



# Performing similar calculation for the validation data
loan.test$Total_revenue = loan.test$total_pymnt - loan.test$loan_amnt
colnames(loan.test)

# Adding predictions to the loan.test h2o frame
# Predictions of a model can be done using the following command
# Gradient Boostin Model
loan.test$gbm_predictions = h2o.predict(object = gbm_model_loan,newdata = loan.test)[,1]
Revenue_gbm = h2o.group_by(data = loan.test, by = c("bad_loan", "gbm_predictions"), sum("Total_revenue"))
Revenue_gbm.R = as.data.frame(Revenue_gbm)

head(Revenue_gbm.R,10)
# Cost of false Negative : -26753378
Revenue_gbm.R[Revenue_gbm.R$bad_loan == 1 & Revenue_gbm.R$gbm_predictions == 0, 3]

# General Linear Model
loan.test$glm_predictions = h2o.predict(object = glm_model_loan,newdata = loan.test)[,1]
Revenue_glm = h2o.group_by(data = loan.test, by = c("bad_loan", "glm_predictions"), sum("Total_revenue"))
Revenue_glm.R = as.data.frame(Revenue_glm)

head(Revenue_glm.R,10)
# Cost of false Negative : -31982462
Revenue_glm.R[Revenue_glm.R$bad_loan == 1 & Revenue_glm.R$glm_predictions == 0, 3]

# Our model should have as less number of false negatives as possible
# A bad loan if classified as a good loan can cause more loss and hence the cost of false 
# negative should be as low as possible

# From the analysis it can be seen that the cost of false negative is less in Model 1 
# or the gbm model in both the training and the validation data.
# Hence it can be chosen as the superior model