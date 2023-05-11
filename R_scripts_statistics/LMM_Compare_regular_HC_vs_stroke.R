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
names(gaitdata)[names(gaitdata) == "Initial contact sensor"] <- "ic_sensor"
names(gaitdata)[names(gaitdata) == "Initial contact OMCS"] <- "ic_omcs"
names(gaitdata)[names(gaitdata) == "Terminal contact sensor"] <- "tc_sensor"
names(gaitdata)[names(gaitdata) == "Terminal contact OMCS"] <- "tc_omcs"

gaitdata$diff_sl = gaitdata$sl_sensor-gaitdata$sl_omcs




## COMPARE HC VS STROKE IN REGULAR WALKING

params <- c('ic', 'tc', 'st', 'sl', 'vel')

for (param in params)
{
  print(param)  
  
  # Set comparison to HC vs stroke in regular walking trials
  trialtype1 = 'GRAIL healthy regular'
  trialtype2 = 'GRAIL stroke regular'
  data= filter(gaitdata, Trialtype == trialtype1 | Trialtype == trialtype2)
  
  new = paste(param, '_sensor', sep="")
  ref = paste(param, '_omcs', sep="")
  ID = "SubjectID"
  trialtype = 'Trialtype'
  
  data_sub <- (data[,c(ID,new,ref, trialtype)])
  colnames(data_sub) <- c("id","new","ref", 'type')
  data_sub$trialtype = as.factor(data_sub$type)
  
  # Calculate the difference per stride
  data_sub$Diff_M <- data_sub$new-data_sub$ref
  
  # Get means and sd for different methods and the mean difference with sd
  data_sub %>%
    group_by(trialtype) %>%
    summarize(mean_sensor = mean(new),
              sd_sensor = sd(new),
              mean_OMCS = mean(ref, na.rm=TRUE),
              sd_OMCS = sd(ref, na.rm=TRUE),
              mean_diff = mean(Diff_M),
              sd_diff = sd(Diff_M))
  

  
  # run comparison between groups (here: trialtypes) in linear mixed model with ID as random effect
  lm1 = lmer(Diff_M ~ trialtype + (1|id), data=data_sub)
  print(summ(lm1, confint = TRUE, digits = 3))
  

} 

