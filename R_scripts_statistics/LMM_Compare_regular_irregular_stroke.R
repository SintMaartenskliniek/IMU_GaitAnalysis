## ---------------------------
##
## Script name: LMM_Compare_regular_irregular_HC
##
## Purpose of script: Run a linear mixed model to compare gait metrics derived from imu's with those derived from optical motion capture systems. 
## Comparison of regular vs irregular walking in healthy subjects as described in Ensink et al. [input doi]
##
## Author: Dr. Katrijn Smulders
##
## Date Created: 2023-05-09
##
## 
## ---------------------------
##
## Notes: This data compares the error in estimating gait metrics using imu's between regular and irregular walking in healhty subjects. 
## Irregular walking was triggered using stepping stones on an instrumented treadmill. STatistcal analysis is done on a stride-by-stride basis.
## A linear model was applied to account for within subject factor (adeed as random factor). 
## See article for details. 
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


## Rename columns
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

## Set reference category to healthy
gaitdata$Trialtype[gaitdata$Trialtype=="GRAIL stroke regular"] = "aGRAIL stroke regular"
gaitdata$Trialtype[gaitdata$Trialtype=="GRAIL stroke irregular"] = "bGRAIL stroke irregular" 


## COMPARE REGULAR VS IRREGULAR IN HC
params <- c('ic', 'tc', 'st', 'sl', 'vel')


for (param in params)
{
  print(param)  
  trialtype1 = 'aGRAIL stroke regular'
  trialtype2 = 'bGRAIL stroke irregular'
  data= filter(gaitdata, Trialtype == trialtype1 | Trialtype == trialtype2)
    
  new = paste(param, '_sensor', sep="")
  ref = paste(param, '_omcs', sep="")
  ID = "SubjectID"
  trialtype = 'Trialtype'
    
  # Subset data
  data_sub <- (data[,c(ID,new,ref, trialtype)])
  colnames(data_sub) <- c("id","new","ref", 'trialtype')
  data_sub$trialtype = as.factor(data_sub$trialtype)
  
  # calculate the difference between methods per stride
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
    
   
    # Run comparison between regular and irregular (trialtype) as repeated measures with within subject factor is ID
    lmm = lmer(Diff_M ~ trialtype + (1|id), data=data_sub)
    qqnorm(resid(lmm))
    qqline(resid(lmm))
    
    print('linear mixed model to compare regular with irregular in stroke')
    # Show output of linear mixed models
    print(summ(lmm, confint = TRUE, digits = 3))
        
} 
    
    