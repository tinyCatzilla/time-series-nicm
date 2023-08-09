# Clear workspace variables
rm(list=ls(all=TRUE))

# I've installed these packages in aiiih shared, but note that it is not in the default R library path as I don't have write access to that.
# install.packages("data.table", lib="/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library", repos = "https://cran.r-project.org")
# install.packages("bit64", lib="/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library", repos = "https://cran.r-project.org")

library(data.table, lib.loc="~/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library")
library(bit, lib.loc="~/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library")
library(bit64, lib.loc="~/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library")
print(paste0("Start prep_data ",format(Sys.time(), "%H:%M")))
################################################################################
# This script reshapes data from long to wide
# It also removes rare data if necessary
# labels[which((nicm==1)|(icm==1)|(amyl==1)|(sarcoido==1)|(hcm==1)|(myocarditis==1)|(dilated==1)]
################################################################################

##################################### flags ####################################
# this switches from time normalized to intervals from admission or intervals
# at specific times of the day (e.g. every 4 hrs or 6am and 6pm).
# T = intervals from admission
time_int <- 7 #time interval by days
flag_useProcs <- FALSE #/* whether to use procedures */
flag_useLabs <- FALSE #/* whether to use labs */
flag_normalize <- FALSE #/* whether to normalize result values */
data_thresh <- 0.01 # threshold if variable only appears <% of patients
flag_MakeTimeInt <- TRUE #flag to make time intervals. code throws errors without this.
flag_MakeConsistTime <- TRUE #flag to make consistent time intervals
flag_MakeNA <- FALSE #flag to make missingness indicators
# flag_MakeInt <- TRUE #flag to make interval between tests
flag_rmrare <- TRUE
flag_LOCF <- TRUE #flag to perform LOCF (last observation carried forward)
flag_echo <- FALSE #flag to use echo as a second index event
save_img <- "/data/aiiih/projects/ts_nicm/data/tmp.RData"
save_path <- "/data/aiiih/projects/ts_nicm/data"

################################## read data ###################################
#load labels
labels <- fread("/data/aiiih/projects/ts_nicm/data/nicm_labels_clean.csv") 
labels <- labels[,c("K_PAT_KEY","mri_dt","nicm","icm","amyl","sarcoido","hcm","myocarditis","dilated"),with=F]
names(labels) <- tolower(names(labels)); names(labels) <- c('pid', 'mri_dt',"nicm","icm","amyl","sarcoido","hcm","myocarditis","dilated");
labels <- labels[order(pid, mri_dt,nicm,icm,amyl,sarcoido,hcm,myocarditis,dilated)]

#load diagnosis
dx <- fread("/data/aiiih/projects/ts_nicm/data/dx_nicm.csv",na.strings=c("","NA","Unknown","UNKNOWN"))
names(dx) <- tolower(names(dx)); names(dx) <- c('pid','eid','var_dt','icd9','var','dx');
dx <- dx[,c('pid','var_dt','var'),with=F]

#load procedures
procs <- fread("/data/aiiih/projects/ts_nicm/data/proc_nicm.csv",na.strings=c("","NA","Unknown","UNKNOWN"))
names(procs) <- tolower(names(procs)); names(procs) <-  c('pid','eid','enc_dt','var_dt','var')
procs[,var_dt:=as.Date(var_dt)]
procs[,val:=1] 
procs <- procs[,c('pid','var_dt','var','val'),with=F]

#load labs
labs <- fread("/data/aiiih/projects/ts_nicm/data/labs_nicm.csv",na.strings=c("","NA","Unknown","UNKNOWN"))
names(labs) <- tolower(names(labs)); names(labs) <-  c('pid','var_id','var','loinc_cd','var_dt','val')
labs[,var_dt:=as.Date(var_dt)]
labs <- labs[,c('pid','var_dt','var','val'),with=F]

#load echo
echo <- fread("/data/aiiih/projects/ts_nicm/data/common_ids.csv",na.strings=c("","NA","Unknown","UNKNOWN"))
echo[,var_dt:=as.Date(echo_dt)]
echo <- echo[,c('pid','echo_dt'),with=F]
print(paste0("Finished Reading Data at ",format(Sys.time(), "%H:%M")))
  
############################### preprocess data ################################
if (flag_useLabs){
  ### change lab val to numeric
  if (exists("labs")) {
    labs[,val:=gsub(">","",val)]
    labs[,val:=gsub("<","",val)]
    labs[,val:=gsub("=","",val)]
    labs[,val:=gsub("L","",val)]
    labs[,val:=gsub("NC","",val)]
    labs[,val:=gsub("%","",val)]
    labs[,val:=gsub("FIO2","",val)]
    labs[,val:=gsub("O2","",val)]
    labs[,val:=gsub("CC","",val)]
    labs[,val:=as.numeric(val)]
    labs <- labs[!is.na(val)]

    ### normalize labs
    if (flag_normalize) {
      labs[,avg:=mean(val,na.rm=T),by=var] # get mean of each lab
      labs[,std:=sd(val,na.rm=T),by=var] # get sd
      labs[,val:=(val-avg)/std] # normalize
      # garbage collection
      labs <- labs[,-c('avg','std'),with=F]
      print(paste0("Finished lab value normalization at ",format(Sys.time(), "%H:%M")))
    }
  }
}

### split dx codes into individual ones
if (exists("dx")) {
  dx[,paste0('v',1:10):=tstrsplit(var,',',fixed=TRUE)]
  dx[,var:=NULL]
  for (v in paste0('v',1:10)) {
    dx[,(v):=gsub(' ','',get(v))]
  }
  dx <- melt(dx, id.vars = c("pid",'var_dt'), measure.vars = paste0('v',1:10))
  dx <- dx[!is.na(value)]
  dx[,uid:=paste0(pid,var_dt,value)]
  dx <- dx[!duplicated(uid)][,-c('uid','variable'),with=F]
  names(dx)[3] <- 'var'
  dx[,val:=1]
}

print(paste0("Finished value preprocessing at ",format(Sys.time(), "%H:%M")))

# Cast columns
dx[, var_dt := as.Date(var_dt)]
dx[, pid := as.character(pid)]
labels[, pid := as.character(pid)]

if flag_echo{
  echo[, pid := as.character(pid)]
}


### combine data together
if (flag_useProcs & flag_useLabs) {
  var_data <- rbind(dx, labs, procs)
} else if (flag_useProcs) {
  var_data <- rbind(dx, procs)
} else if (flag_useLabs) {
  var_data <- rbind(dx, labs)
} else {
  var_data <- dx
}

var_data <- var_data[pid %in% labels[,pid]]

# Take the most recent MRI date for each patient
labels <- labels[, .SD[which.max(mri_dt)], by = pid]

if flag_echo{
  # Take the most recent echo date for each patient
  echo <- echo[, .SD[which.max(echo_dt)], by = pid]
  # Merge labels with echo data on pid and keep both mri_dt and echo_dt
  labels <- merge(labels, echo[, .(pid, echo_dt)], by = 'pid')

  # Calculate the difference in years between mri_dt and echo_dt
  labels[, date_diff := as.numeric(difftime(echo_dt, mri_dt, units = "weeks"))/52.25]

  # Filter rows where the difference is <= 3 years
  labels <- labels[date_diff <= 3]

  # Drop the date_diff column
  labels[, date_diff := NULL]
}


# Merge labels and var_data
dt1 <- merge(labels, var_data, by = 'pid')

if flag_echo{
  # Drop mri_dt
  dt1[, mri_dt := NULL]

  # Rename echo_dt to mri_dt
  names(dt1)[names(dt1) == 'echo_dt'] <- 'mri_dt'

  rm(echo)
}

fwrite(labels, file.path(save_path, "labels_sani.csv"))
# fwrite(dx, file.path(save_path, "dx_sani.csv"))

rm(labels, dx, labs, procs, var_data)


############################### standardize time ###############################
if (flag_MakeTimeInt) { 
  ### standardize time by intervals relative to admin 
  dt1[,dtime:=as.numeric(difftime(var_dt,mri_dt,units="days"))]
  #/* remove data that are out of scope */
  dt1 <- dt1[(dtime>= -182) & (dtime < 0)]
  #/* normalize time */
  dt1[,dtime:=floor(dtime/time_int)*time_int]
 
  print(paste0("Finished standardizing time at ",format(Sys.time(), "%H:%M")))
}

temp <- dcast(dt1[,c("pid","dtime","var","val"),with=F],
                pid+dtime~var, value.var="val", last, fill=NA) #
  temp <- temp[order(pid,dtime)]
  print(paste0("num vars: ",ncol(temp), ", num pats: ", uniqueN(temp[,pid])))

# remove rare data elements
if (flag_rmrare) {
  nPats <- dt1[,uniqueN(pid)]
  dt1[,pvar:=uniqueN(pid)/nPats,by=var]
  dt1 <- dt1[pvar>data_thresh]
  # garbage collection
  dt1[,pvar:=NULL]; rm(nPats)
  print(paste0("Finished removing rare data at ",data_thresh, " at ",format(Sys.time(), "%H:%M")))
}
var_names <- unique(dt1[,var])


############################### reshape to wide ################################
if (!exists("dt1w")) {
  #"nicm","icm","amyl","sarcoido","hcm","myocarditis","dilated"
  dt1w <- dcast(dt1[,c("pid","dtime","var","val"),with=F],
                pid+dtime~var, value.var="val", last, fill=NA) #
  dt1w <- dt1w[order(pid,dtime)]
  print(paste0("Finished reshaping ",format(Sys.time(), "%H:%M")))
  print(paste0("num vars: ",ncol(dt1w), ", num pats: ", uniqueN(dt1w[,pid])))
######################## processing various indicators #########################
}

if (flag_MakeConsistTime) {
  # Find the earliest observation time for each patient
  earliest_time <- dt1[, min(dtime), by = pid]
  
  # Create a consistent timeline for each patient from their earliest time to 7 days prior
  consistent_time <- lapply(seq_len(nrow(earliest_time)), function(i) {
    data.table(pid = earliest_time[i, pid], 
               dtime = seq(from = earliest_time[i, V1], to = -7, by = time_int))
  })
  consistent_time <- rbindlist(consistent_time)
  
  # Merge the original data with this consistent timeline, filling in any missing entries
  dt1w <- merge(dt1w, consistent_time, by = c("pid", "dtime"), all = TRUE)
  
  # Order by pid and time
  dt1w <- dt1w[order(pid, dtime)]
  print(var_names)
  print(paste0("Finished setting consistent data intervals at ", format(Sys.time(), "%H:%M")))
}

### add missingness indicators
if (flag_MakeNA) {
  dt1w_NA <- data.table(is.na(dt1w[,-c("pid","dtime"),with=F]))
  dt1w_NA[dt1w_NA==T] <- 1
  names(dt1w_NA) <- paste0(names(dt1w_NA),"_NA")
  dt1w <- cbind(dt1w, dt1w_NA)
  rm(dt1w_NA)

  print(paste0("Finished adding missing values at ",format(Sys.time(), "%H:%M")))
}

if (flag_LOCF) {
  # for any NA values, fill with 0
  dt1w[is.na(dt1w)] <- 0
  fwrite(dt1w, file.path(save_path, "check.csv"))
  for (kVar in var_names) {
    dt1w[, (kVar) := {
      locf_vector <- get(kVar)
      idx_first_one <- which(locf_vector == 1)[1]   
      if (!is.na(idx_first_one)) {
        locf_vector[idx_first_one:length(locf_vector)] <- 1
      }
      locf_vector
    }, by = pid]
  }
  print(paste0("Finished adding data intervals at ",format(Sys.time(), "%H:%M")))
}


### normalize labs
if (flag_normalize & flag_useLabs) {
  for (kVar in var_names) {
    dt1w[,avg:=mean(get(kVar),na.rm=T),by=kVar] # get mean of each lab
    dt1w[,std:=sd(get(kVar),na.rm=T),by=kVar] # get sd
    dt1w[,(kVar):=(get(kVar)-avg)/std] # normalize
    dt1w <- dt1w[,-c('avg','std'),with=F]
  }
  print(paste0("Finished lab value normalization at ",format(Sys.time(), "%H:%M")))
}



######################## save file #########################
fwrite(dt1w, file.path(save_path, "nicm_combined.csv"))
save.image(file=save_img)
print(paste0("Finished running prep_data.R at ",format(Sys.time(), "%H:%M")))