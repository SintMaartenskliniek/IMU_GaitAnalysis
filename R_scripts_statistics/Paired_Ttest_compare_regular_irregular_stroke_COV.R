## ---------------------------
##
## Script name: 
##
## Purpose of script:
##
## Author: Dr. Katrijn Smulders
##
## Date Created: 2023-05-09
##
## 
## ---------------------------
##
## Notes:
##   
##
## ---------------------------

## load up the packages we will need:


library(dplyr)
library(lme4)
library(jtools)
library(readxl)

## open data set, assumed the .xlsx file is available from the working directory
filename = "./R_dataset.xlsx"
gaitdata <- read_excel(filename)

## change ID (included condition of the walk) to ID only
gaitdata$SubjectID = substr(gaitdata$SubjectID,1,10)

# rename columns
names(gaitdata)[names(gaitdata) == "Stride velocity sensor"] <- "vel_sensor"
names(gaitdata)[names(gaitdata) == "Stride velocity OMCS"] <- "vel_omcs"
names(gaitdata)[names(gaitdata) == "Stride length sensor"] <- "sl_sensor"
names(gaitdata)[names(gaitdata) == "Stride length OMCS"] <- "sl_omcs"
names(gaitdata)[names(gaitdata) == "Stride time sensor"] <- "st_sensor"
names(gaitdata)[names(gaitdata) == "Stride time OMCS"] <- "st_omcs"

# remove strides if there is no vicon OR no sensor data to enable averaging
gaitdata[complete.cases(gaitdata), ]

# define parameters for Coefficient of variation analysis
params <- c('st', 'sl')

# Define relevant trial conditions for this analysis
trialtype1 = 'GRAIL stroke regular'
trialtype2 = 'GRAIL stroke irregular'
data= filter(gaitdata, Trialtype == trialtype1 | Trialtype == trialtype2)
data= filter(data, SubjectID != "900_CVA_03") # Participant did not perform irregular walking condition

for (param in params){
 
    print(param)
  
    new = paste(param, '_sensor', sep="")
    ref = paste(param, '_omcs', sep="")
    ID = "SubjectID"
    trialtype="Trialtype"
  
    data_sub <- (data[,c(ID,trialtype, new, ref)])
    colnames(data_sub) <- c("id","trialtype","new","ref")
    
    # calculate mean and sd per participant (i.e. averaging over strides per participant) 
    sdsensor = aggregate(new ~ data_sub$id + trialtype, data=data_sub, FUN=sd)
    sdomcs = aggregate(ref ~ data_sub$id + trialtype, data=data_sub, FUN=sd)
    meansensor = aggregate(new ~ data_sub$id + trialtype, data=data_sub, FUN=mean)
    meanomcs = aggregate(ref ~ data_sub$id + trialtype, data=data_sub, FUN=mean)
    
    # Create new dataframe with all cov data per participant (mean and sd)
    covdata = bind_cols(sdsensor, meansensor$new, sdomcs$ref, meanomcs$ref)
    # rename column names in this dataframe
    covdata = rename(covdata,  "sd_sensor"= new, "mean_sensor"="...4", "sd_omcs" = "...5", "mean_omcs"="...6")  
    
    # Calculate cov per subjects
    covdata$cov_sensor = 100*covdata$sd_sensor/covdata$mean_sensor # coefficient of variation in percentages
    covdata$cov_omcs = 100*covdata$sd_omcs/covdata$mean_omcs # coefficient of variation in percentages
    
    # Calculate the difference between methods (sensor vs omcs) of cov per subject
    covdata$method_diff = covdata$cov_sensor-covdata$cov_omcs
    
  
    # Paired t-test to compare the conditions (regular vs irregular)
    res <- t.test(method_diff ~ trialtype, data = covdata, paired = TRUE)
    print(res)
    # Show summary of COV of the parameter (x) per condition
    print(aggregate(covdata$method_diff, list(covdata$trialtype), mean))
    
}
