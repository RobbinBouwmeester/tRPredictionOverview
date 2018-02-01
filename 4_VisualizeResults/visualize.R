# Copyright 2017 Robbin Bouwmeester
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This code is used to analyze the output of multiple machine learning
# algorithms used for LC retention time prediction. The code generates
# the plots used in the manuscript.
#
# Library versions:
# 
#  R version 3.3.1 (2016-06-21)
#  RColorBrewer_1.1-2 
#  corrplot_0.84      
#  ggplot2_2.2.1 
#
#
# This project was made possible by MASSTRPLAN. MASSTRPLAN received funding 
# from the Marie Sklodowska-Curie EU Framework for Research and Innovation 
# Horizon 2020, under Grant Agreement No. 675132.

library(ggplot2)
library(corrplot)

# Set working directory if needed
#setwd("")

# Output plots to pdf and or eps
plot_pdf <- F
plot_eps <- T

# Output dir
output_dir <- "figs/"

#########################################################################################
#                                                                                       #
# Plot the learning curves                                                              #
# Start with the aggregated learning curves (i.e. multiple learning curves in one plot) #
#                                                                                       #
#########################################################################################

# Function to plot the learning curve.
#
# Inputs:
#   df - dataframe consisting of the columns "experiment" (indicating the dataset) 
#        "num_train" (indicating the number of training examples) "reDistributpeat" (indicating a possible repeat of learning) 
#        "algo" (indicating the algorithm) "tr" (indicating the experimentally measured retention time)  
#        "pred" (indicating the prediction)
#
#   ylim - vector indicating the limits c([from,to])
#
#   legend_loc - vector with location of the legend c([relative location x, relative location y])
#
#   title - string with the title
#
#   xlab - string with the x label
#
#   ylab - string with the y label
#
#   percentage - boolean indicating if percentages are used (value will be multiplied by 100)

plot_learning_curves <- function(df,ylim=c(-0.5,1.0),legend_loc=c(0.95,0.75),
                                 title="",
                                 xlab="Number of initial training instances",
                                 ylab="Average error relative to total elution time (%)",
                                 percentage=F){
  # Percentages need to be multiplied by 100 %
  if (percentage) {
    p<-ggplot(df, aes(x=as.factor(num_train), y=perf*100, fill = algo)) +
      geom_boxplot(position=position_dodge(0.6),width=0.65,outlier.size = 1.5) +
      stat_summary(fun.y = mean, geom="point",colour="darkred", size=1,position=position_dodge(0.6))
  }
  else {
    p<-ggplot(df, aes(x=as.factor(num_train), y=perf, fill = algo)) +
      geom_boxplot(position=position_dodge(0.6),width=0.65,outlier.size = 1.5) +
      stat_summary(fun.y = mean, geom="point",colour="darkred", size=1,position=position_dodge(0.6))
  }
  
  # Apply a theme and put labels+legend in
  p <- p + scale_fill_brewer("") + theme_classic() + 
    theme(plot.background = element_blank(),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()) +
    theme(axis.line.x = element_line(color="black", size = 0.5),
          axis.line.y = element_line(color="black", size = 0.5),
          axis.line = element_line(color="black", size = 0.5),
          legend.justification=c(1,0), legend.position=legend_loc,legend.text=element_text(size=12))+ 
    ylim(ylim) 
  p <- p + labs(title = title,
                x = xlab, 
                y = ylab)
  return(p)
  
}

# Read the data for the lcurves with the 151 feature set
summary_lcurves <- read.csv("data/predictions_algo_sum_big_lcurve_name.csv")

# For annotation of algorithms a factor is needed instead of a vector with strings
summary_lcurves$algo <- factor(summary_lcurves$algo, levels = c("SVR","BRR","LASSO","AB","ANN","RF","GB"))

# Write to pdf?
if (plot_pdf){
  pdf(paste0(output_dir,"lcurve_me_bigfeat.pdf"),width=7.5,height=7.5)
}

# Write to eps?
if (plot_eps){
  postscript(paste0(output_dir,"lcurve_me_bigfeat.eps"), width = 7.5, height = 7.5, horizontal = FALSE, 
           onefile = FALSE, paper = "special", colormodel = "cmyk", 
           family = "Times")
}

# Plot the learning curve with the performance measure median absolute error
print(plot_learning_curves(summary_lcurves[summary_lcurves$perf_type == "me",],ylim=c(0,30.0),ylab="Median error relative to total elution time (%)",percentage=T))

# Close dev if figure was written
if (plot_pdf | plot_eps){
  dev.off()
}

# Write to pdf?
if (plot_pdf){
  pdf(paste0(output_dir,"lcurve_mae_bigfeat.pdf"),width=7.5,height=7.5)
}

# Write to eps?
if (plot_eps){
  postscript(paste0(output_dir,"lcurve_mae_bigfeat.eps"), width = 7.5, height = 7.5, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Plot the learning curve with the performance measure mean absolute error
print(plot_learning_curves(summary_lcurves[summary_lcurves$perf_type == "mae",],ylim=c(0,30.0),ylab="Average error relative to total elution time (%)",percentage=T))

# Close dev if figure was written<
if (plot_pdf | plot_eps){
  dev.off()
}

# Write to pdf?
if (plot_pdf){
  pdf(paste0(output_dir,"lcurve_cor_bigfeat.pdf"),width=7.5,height=7.5)
}

# Write to eps?
if (plot_eps){
  postscript(paste0(output_dir,"lcurve_cor_bigfeat.eps"), width = 7.5, height = 7.5, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Plot the learning curve with the performance measure correlation
print(plot_learning_curves(summary_lcurves[summary_lcurves$perf_type == "correlation",],legend_loc=c(1,0),ylab="Pearson correlation (predicted and experimental retention time)"))

# Close dev if figure was written
if (plot_pdf | plot_eps){
  dev.off()
}

# Read the data for the lcurves with the 9 feature set
summary_lcurves <- read.csv("data/predictions_algo_sum_small_lcurve_name.csv")

# For annotation of algorithms a factor is needed instead of a vector with strings
summary_lcurves$algo <- factor(summary_lcurves$algo, levels = c("SVR","BRR","LASSO","AB","ANN","RF","GB"))

# Write to pdf?
if (plot_pdf){
  pdf(paste0(output_dir,"lcurve_me_smallfeat.pdf"),width=7.5,height=7.5)
}

# Write to eps?
if (plot_eps){
  postscript(paste0(output_dir,"lcurve_me_smallfeat.eps"), width = 7.5, height = 7.5, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Plot the learning curve with the performance measure median absolute error
print(plot_learning_curves(summary_lcurves[summary_lcurves$perf_type == "me",],ylim=c(0,30.0),ylab="Median error relative to total elution time (%)",percentage=T))

# Close dev if figure was written
if (plot_pdf | plot_eps){
  dev.off()
}

# Write to pdf?
if (plot_pdf){
  pdf(paste0(output_dir,"lcurve_me_bigfeat.pdf"),width=7.5,height=7.5)
}

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"lcurve_me_bigfeat.eps"), width = 7.5, height = 7.5, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Plot the learning curve with the performance measure mean absolute error
print(plot_learning_curves(summary_lcurves[summary_lcurves$perf_type == "mae",],ylim=c(0,30.0),ylab="Average error relative to total elution time (%)",percentage=T))

# Close dev if figure was written
if (plot_pdf | plot_eps){
  dev.off()
}

# Write to pdf?
if (plot_pdf){
  pdf(paste0(output_dir,"lcurve_cor_smallfeat.pdf"),width=7.5,height=7.5)
}

# Write to eps?
if (plot_eps){
  postscript(paste0(output_dir,"lcurve_cor_smallfeat.eps"), width = 7.5, height = 7.5, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Plot the learning curve with the performance measure correlation
print(plot_learning_curves(summary_lcurves[summary_lcurves$perf_type == "correlation",],legend_loc=c(1,0),ylab="Pearson correlation (predicted and experimental retention time)"))

# Close dev if figure was written
if (plot_pdf | plot_eps){
  dev.off()
}

#########################################################################################
#                                                                                       #
# Plot the learning curves                                                              #
# Seperated learning curves for different datasets (i.e. learning curve per dataset)    #
#                                                                                       #
#########################################################################################

# Read the data for the lcurves with the 151 feature set
summary_lcurves <- read.csv("data/predictions_algo_sum_big_lcurve_no_norm_name.csv")

# Extract the median absolute error
temp_summary_lcurves <- summary_lcurves[summary_lcurves$perf_type == "me",]

# For the 10 repeats of creating the learning curve (different training/test sets) get the mean value
mean_perf_algo <- aggregate(perf~experiment+num_train+algo, data=temp_summary_lcurves, mean)

# Function to plot the learning curve.
#
# Inputs:
#   mean_perf_algo - dataframe consisting of the columns "experiment" (indicating the dataset) 
#        "num_train" (indicating the number of training examples) "algo" (indicating the algorithm) 
#        "perf" (indicating the performance for a previously selected measure)
plot_algo_lcurve_exp <- function(mean_perf_algo){
  require(RColorBrewer)
  
  # Get a list of all algorithms
  unique_algo <- unique(mean_perf_algo$algo)
  # Get a list of all datasets
  unique_exp <- unique(mean_perf_algo$experiment)
  
  # Match the algorithms to a color
  matchcols = data.frame(unique_algo=unique_algo,
                         color= c("black","grey","blue","lightblue","magenta","red","green"),
                         ltype=1:length(unique_algo))
  
  # Iterate over the datasets
  for (j in unique_exp){
    # Make sure we have enough training examples
    if (max(mean_perf_algo[mean_perf_algo$experiment == j,]$num_train) > 159){
      # Make an initial plot without any learning curve  
      plot(0,0,ylim=c(min(mean_perf_algo[mean_perf_algo$experiment == j,]$perf)-0.01,max(mean_perf_algo[mean_perf_algo$experiment == j,]$perf)+0.01),
           xlim=c(0,max(mean_perf_algo[mean_perf_algo$experiment == j,]$num_train)),
           main=j,xlab="Training examples",ylab="Median absolute error (s)")
      # Fill the initialized plot by iterating over the different algorithms
      for (i in unique_algo){
        # Extract the algorithm
        temp_mean_perf_algo <- mean_perf_algo[mean_perf_algo$experiment == j & mean_perf_algo$algo == i,]
        
        # Plot the line
        lines(temp_mean_perf_algo$num_train,temp_mean_perf_algo$perf,type="l",
              col=matchcols$color[matchcols$unique_algo==i],
              lty=3)
        
        # Add points to the line for every step
        points(temp_mean_perf_algo$num_train,temp_mean_perf_algo$perf,
               col=matchcols$color[matchcols$unique_algo==i],
               pch=matchcols$ltype[matchcols$unique_algo==i])
      }
    }
    
  }
  # Reset any plotting settings so the legend can be placed in the bottom right
  reset()
  
  # Plot the legend
  legend("bottomright", 
         legend=unique_algo,
         lty=3,
         pch=matchcols$ltype,
         col=matchcols$color)
}

# Reset function for plotting settings
reset <- function() {
  par(mfrow=c(1, 1), oma=rep(2, 4), mar=rep(2, 4), new=TRUE)
  plot(0:1, 0:1, type="n", xlab="", ylab="", axes=FALSE)
}

# Write to eps?
if (plot_eps){
  postscript(paste0(output_dir,"lcurve_per_set_me.eps"), width = 7, height = 9, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Set plotting settings
par(mar=c(5, 5, 1.5, 0.0) + 0.1)
par(mfrow=c(3,2))

# Retrieve the lcurve median absolute error and plot
temp_summary_lcurves <- summary_lcurves[summary_lcurves$perf_type == "me",]
mean_perf_algo <- aggregate(perf~experiment+num_train+algo, data=temp_summary_lcurves, mean)
plot_algo_lcurve_exp(mean_perf_algo)

# Close dev if figure was written
if (plot_pdf | plot_eps){
  dev.off()
}

# Write to eps?
if (plot_eps){
  postscript("lcurve_per_set_mae.eps", width = 10, height = 10, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Set plotting settings
par(mar=c(5, 5, 1.5, 0.0) + 0.1)
par(mfrow=c(3,2))

# Retrieve the lcurve mean absolute error and plot
temp_summary_lcurves <- summary_lcurves[summary_lcurves$perf_type == "mae",]
mean_perf_algo <- aggregate(perf~experiment+num_train+algo, data=temp_summary_lcurves, median)
plot_algo_lcurve_exp(mean_perf_algo)

# Close dev if figure was written
if (plot_pdf | plot_eps){
  dev.off()
}

# Write eps?
if (plot_eps){
  postscript("lcurve_per_set_cor.eps", width = 10, height = 10, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Set plotting settings
par(mar=c(5, 5, 1.5, 0.0) + 0.1)
par(mfrow=c(3,2))

# Retrieve the correlation and plot
temp_summary_lcurves <- summary_lcurves[summary_lcurves$perf_type == "correlation",]
mean_perf_algo <- aggregate(perf~experiment+num_train+algo, data=temp_summary_lcurves, median)
plot_algo_lcurve_exp(mean_perf_algo)

# Close dev if figure was written
if (plot_pdf | plot_eps){
  dev.off()
}



#########################################################################################
#                                                                                       #
# Plot the CV results - big feature set                                                 #
#                                                                                       #
#########################################################################################

#####################################################
#                                                   #
# Plot the correlation of errors with each other    #
#                                                   #
#####################################################

# Read the CV results for the 151 feature set
cv_preds <- read.csv("data/predictions_algo_verbose_big_cv_name.csv")

# One plot per plotting window
par(mfrow=c(1,1))

# Dataframe init with the SVM as initial start
error_df = data.frame(rem=matrix(nrow = length(cv_preds[cv_preds$algo=="SVR","tr"])))
for (a in c("SVR","GB","BRR","AB","LASSO","RF","ANN")){
  error_df[a] = cv_preds[cv_preds$algo==a,"tr"]-cv_preds[cv_preds$algo==a,"pred"]
}

# Remove the first column because it will contain "NA"
error_df$rem <- NULL

if (plot_eps){
  postscript("corrplot_errors.eps", width = 7.5, height = 7.5, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Correlation plot between the different algorithms
corrplot(cor(error_df,method="spearman"),method="number",cl.lim=c(0,1.0),tl.col="black")

if (plot_pdf | plot_eps){
  dev.off()
}

###########################################################################
#                                                                         #
# Plot the error per dataset for each algorithm for a pairwise comparison #
#                                                                         #
###########################################################################

# Get the experiments from the CV set
unique_experi <- unique(cv_preds$experiment)

# Get the algos from the CV set
unique_algo <- unique(cv_preds$algo)

# Initialize vectors that will hold the performance measures per dataset and algorithm
t_algo_cor <- c()
t_algo_mae <- c()
t_algo_me <- c()

# Iterate over the experiments and algorithms and use different performance measures 
for (e in unique_experi){
  for (a in unique_algo){
    temp_cv_preds <- cv_preds[cv_preds$experiment == e & cv_preds$algo == a,]
    t_algo_mae <- c(t_algo_mae,sum(abs(temp_cv_preds$tr-temp_cv_preds$pred),na.rm=T)/length(temp_cv_preds$tr))
    t_algo_cor <- c(t_algo_cor,cor(temp_cv_preds$tr,temp_cv_preds$pred))
    t_algo_me <- c(t_algo_me,median(abs(temp_cv_preds$tr-temp_cv_preds$pred),na.rm=T))
    
  }
}

# Make sure the performance measures are put in a matrix with the correct algorithm and experiment name
perf_mae <- data.frame(matrix(t_algo_mae,ncol=length(unique_algo),byrow=T))
colnames(perf_mae) <- unique_algo
rownames(perf_mae) <- unique_experi

perf_me <- data.frame(matrix(t_algo_me,ncol=length(unique_algo),byrow=T))
colnames(perf_me) <- unique_algo
rownames(perf_me) <- unique_experi

perf_me_big <- perf_me

perf_cor <- data.frame(matrix(t_algo_cor,ncol=length(unique_algo),byrow=T))
colnames(perf_cor) <- unique_algo
rownames(perf_cor) <- unique_experi

# Write to eps?
if (plot_eps){
  postscript(paste0(output_dir,"pairwise_scatter_mae.eps"), width = 8, height = 32, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Two columns with nine rows plotting window
par(mfrow=c(9,2))

# Vector that will hold previously analyzed combinations
analyzed <- c()

# Iterate over all algorithms twice an make combinations between them to pairwise compare the mean absolute error
for (a in c("SVR","GB","BRR","AB","RF","ANN")){
  for (b in c("SVR","GB","BRR","AB","RF","ANN")){
    # Check if combination was already plotted (if not make a plot)
    if (a != b & !(paste(a,b,sep="_") %in% analyzed) & !(paste(b,a,sep="_") %in% analyzed)){
      analyzed <- c(analyzed,paste(a,b,sep="_"))
      plot(perf_mae[,a],perf_mae[,b],xlab=a,ylab=b,pch=10,ylim=c(0,500),xlim=c(0,500),cex=0.5)
      abline(0,1,lty=2)
    }
  }
}

# Close dev if figure was written
if (plot_pdf | plot_eps){
  dev.off()
}

# Write to eps?
if (plot_eps){
  postscript(paste0(output_dir,"pairwise_scatter_me.eps"), width = 8, height = 32, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Two columns with nine rows plotting window
par(mfrow=c(9,2))

# Vector that will hold previously analyzed combinations
analyzed <- c()

# Iterate over all algorithms twice an make combinations between them to pairwise compare the median absolute error
for (a in c("SVR","GB","BRR","AB","RF","ANN")){
  for (b in c("SVR","GB","BRR","AB","RF","ANN")){
    # Check if combination was already plotted (if not make a plot)
    if (a != b & !(paste(a,b,sep="_") %in% analyzed) & !(paste(b,a,sep="_") %in% analyzed)){
      analyzed <- c(analyzed,paste(a,b,sep="_"))
      plot(perf_me[,a],perf_me[,b],xlab=a,ylab=b,pch=10,ylim=c(0,500),xlim=c(0,500),cex=0.5)
      abline(0,1,lty=2)
    }
  }
}

# Close dev if figure was written
if (plot_pdf | plot_eps){
  dev.off()
}

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"pairwise_scatter_error.eps"), width = 8, height = 32, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Two columns with nine rows plotting window
par(mfrow=c(9,2))

# Vector that will hold previously analyzed combinations
analyzed <- c()

# Iterate over all algorithms twice an make combinations between them to pairwise compare the correlation between observed and experimental
for (a in c("SVR","GB","BRR","AB","RF","ANN")){
  for (b in c("SVR","GB","BRR","AB","RF","ANN")){
    # Check if combination was already plotted (if not make a plot)
    if (a != b & !(paste(a,b,sep="_") %in% analyzed) & !(paste(b,a,sep="_") %in% analyzed)){
      analyzed <- c(analyzed,paste(a,b,sep="_"))
      plot(perf_cor[,a],perf_cor[,b],xlab=a,ylab=b,pch=10,ylim=c(0,1),xlim=c(0,1),cex=0.5)
      abline(0,1,lty=2)
    }
  }
}

# Close dev if figure was written
if (plot_pdf | plot_eps){
  dev.off()
}

############################################################
#                                                          #
# Get the best and worst performing algorithms per dataset #
#                                                          #
############################################################

# One plot per plotting window
par(mfrow=c(1,1))

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"highest_bars_me.eps"), width = 3, height = 4.5, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Reset the margins
par(mar=c(7, 4, 4, 2) + 0.1)

# Get and plot the count for the highest median error (worst) over all datasets
barplot(table(colnames(perf_me)[apply(perf_me,1,which.max)]),ylab="Highest median error on dataset (#)", las=3)

# Close dev if figure was plotted
if (plot_pdf | plot_eps){
  dev.off()
}

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"lowest_bars_me.eps"), width = 3, height = 4.5, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Reset the margins
par(mar=c(7, 4, 4, 2) + 0.1)

# Get and plot the count for the lowest median error (best) over all datasets
barplot(table(colnames(perf_me)[apply(perf_me,1,which.min)]),ylab="Lowest median error on dataset (#)",ylim=c(0,13), las=3)

# Close dev if figure was plotted
if (plot_pdf | plot_eps){
  dev.off()
}

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"rank_bars_me.eps"), width = 3, height = 4.5, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Reset the margins
par(mar=c(7, 4, 4, 2) + 0.1)

# Get and plot the count for the mean ranking in terms of median error (lower mean rank is better) over all datasets
barplot(apply(apply(perf_me,1,rank),1,mean),ylab="Mean rank of median error",ylim=c(0,5),las=3)

# Close dev if figure was plotted
if (plot_pdf | plot_eps){
  dev.off()
}

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"highest_bars_mae.eps"), width = 6, height = 6, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Get and plot the count for the highest mean error (worst) over all datasets
barplot(table(colnames(perf_mae)[apply(perf_mae,1,which.max)]),ylab="Highest mean error on dataset (#)", las=3)

# Close dev if figure was plotted
if (plot_pdf | plot_eps){
  dev.off()
}

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"lowest_bars_mae.eps"), width = 6, height = 6, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Get and plot the count for the lowest mean error (best) over all datasets
barplot(table(colnames(perf_mae)[apply(perf_mae,1,which.min)]),ylab="Lowest mean error on dataset (#)", las=3)

# Close dev if figure was plotted
if (plot_pdf | plot_eps){
  dev.off()
}

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"highest_bars_cor.eps"), width = 6, height = 6, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Get and plot the count for the highest correlation (best) between predictions and experimental observations over all datasets
barplot(table(colnames(perf_cor)[apply(perf_cor,1,which.max)]),ylab="Highest correlation on dataset (#)", las=3)

# Close dev if figure was written
if (plot_pdf | plot_eps){
  dev.off()
}

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"lowest_bars_cor.eps"), width = 6, height = 6, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Get and plot the count for the lowest correlation (worst) between predictions and experimental observations over all datasets
barplot(table(colnames(perf_cor)[apply(perf_cor,1,which.min)]),ylab="Lowest correlation on dataset (#)", las=3)

# Close dev if figure was plotted
if (plot_pdf | plot_eps){
  dev.off()
}

######################################################
#                                                    #
# Make a pairwise comparison between XGBoost and SVM #
#                                                    #
######################################################

# Get the maximum retention time per experiment
max_rt <- tapply(cv_preds$tr, cv_preds$experiment, max)[rownames(perf_me)]

# Plot the median absolute error and mean absolute error per dataset for XGBoost and SVM
par(mfrow=c(1,2))

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"xgb_vs_svm_me.eps"), width = 6, height = 6, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Scatterplot that plots the normalized median absolute error for each dataset between SVM (x-axis) and XGBoost (y-axis)
plot(perf_me$SVR/max_rt,perf_me$GB/max_rt,pch=10,xlim=c(0,320),ylim=c(0,320),xlab="SVM (median absolute error(s))",ylab="XGBoost (median absolute error (s))")

# Straight line; indicating no difference in performance between the algorithms
abline(0,1)

# Close dev if figure was plotted
if (plot_pdf | plot_eps){
  dev.off()
}

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"xgb_vs_svm_mae.eps"), width = 6, height = 6, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Scatterplot that plots the normalized mean absolute error for each dataset between SVM (x-axis) and XGBoost (y-axis)
plot(perf_mae$SVR/max_rt,perf_mae$GB/max_rt,pch=10,xlim=c(0,320),ylim=c(0,320),xlab="SVM (mean absolute error(s))",ylab="XGBoost (mean absolute error (s))")
abline(0,1)

# Close dev if figure was plotted
if (plot_pdf | plot_eps){
  dev.off()
}

# Calculate the number of molecules per dataset, division by 7  is performed because there are 7 algorithms
perf_me <- cbind(perf_me,Freq=c((table(cv_preds$experiment)/7)[rownames(perf_me)]))
perf_mae <- cbind(perf_mae,Freq=c((table(cv_preds$experiment)/7)[rownames(perf_mae)]))

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"xgb_vs_svm_me_size.eps"), width = 6, height = 6, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Plot the difference between SVM and XGBoost vs the number of training examples in terms of their median absolute error per dataset
par(mfrow=c(1,2))
plot((perf_me$SVR-perf_me$GB),perf_me$Freq,pch=10,
     xlab="xgb - SVM (difference in median error)",ylab="Total number of examples")
abline(v=0,lty=2)
abline(h=100,lty=2)

# Close dev if figure was plotted
if (plot_pdf | plot_eps){
  dev.off()
}

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"xgb_vs_svm_mae_size.eps"), width = 6, height = 6, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Plot the difference between SVM and XGBoost vs the number of training examples in terms of their mean absolute error per dataset
plot((perf_mae$SVR-perf_mae$GB),perf_mae$Freq,pch=10,
     xlab="xgb - SVM (difference in mean error)",ylab="")
abline(v=0,lty=2)
abline(h=100,lty=2)

# Close dev if figures were plotted
if (plot_pdf | plot_eps){
  dev.off()
}

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"xgb_vs_svm_me_size_normalized.eps"), width = 6, height = 6, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Plot the difference between SVM and XGBoost vs the number of training examples in terms of their normalized median absolute error per dataset
plot(((perf_me$SVR-perf_me$GB)/max_rt)*100,perf_me$Freq,pch=10,
     xlab="GB - SVM (difference in relative median error (%))",ylab="Total number of examples",xlim=c(-2,6))
abline(v=0,lty=2)
abline(h=100,lty=2)

# Close dev if figure was written
if (plot_pdf | plot_eps){
  dev.off()
}

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"xgb_vs_svm_mae_size_normalized.eps"), width = 6, height = 6, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Plot the difference between SVM and XGBoost vs the number of training examples in terms of their normalized mean absolute error per dataset
plot(((perf_mae$SVR-perf_mae$GB)/max_rt)*100,perf_mae$Freq,pch=10,
     xlab="xgb - SVM (difference in relative mean error (%))",ylab="",xlim=c(-2,6))
abline(v=0,lty=2)
abline(h=100,lty=2)

if (plot_pdf | plot_eps){
  dev.off()
}

#######################################################################
#                                                                     #
# Make a pairwise comparison between XGBoost and the other algorithms #
#                                                                     #
#######################################################################

# Plot settings
par(mfrow=c(1,2))
par(mar=c(7, 3, 4, 2) + 0.1)

# Vector that holds combinations that have already been analyzed (counters A -> B and B -> A comparisons)
analyzed <- c()

# Get the maximum retention time per experiment based on the last identification
max_rt <- tapply(cv_preds$tr, cv_preds$experiment, max)[rownames(perf_me)]

# Subtract the difference in median error with the best performing other algorithm (this can be the same algorithm as compared with!)
diff_best <- perf_me[,c("GB")]-apply(perf_me[,c("GB","AB","ANN","BRR","SVR","RF","LASSO")],1,min)

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"xgb_vs_all.eps"), width = 6, height = 6, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Get the best performing algorithms per dataset and normalize their error, write to original dataframe
perf_me$diff_algo <- diff_best
perf_me$diff_algo_order <- (diff_best)/max_rt

# Make a barplot with the normalized median error of the best algorithms compared to GB
barplot((perf_me$diff_algo[order(perf_me$diff_algo_order)]/max_rt[order(perf_me$diff_algo_order)])*100,
        names=rownames(perf_me)[order(perf_me$diff_algo_order)],las=2,cex.names=0.8,
        main=paste("Best out of seven - GB","\n Average increase (%):",round(mean((perf_me$diff_algo[order(perf_me$diff_algo_order)]/max_rt[order(perf_me$diff_algo_order)])*100),2)),
        ylim=c(0,8),ylab="")
print((perf_me$diff_algo[order(perf_me$diff_algo_order)]/max_rt[order(perf_me$diff_algo_order)])*100)
mtext(text="Difference in median error \n relative to the total elution time  (%)", side=2, line=3, las=0) 

# Close dev if figure was written
if (plot_pdf | plot_eps){
  dev.off()
}

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"svm_vs_all.eps"), width = 6, height = 6, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Subtract the difference in median error with the best performing other algorithm (this can be the same algorithm as compared with!)
diff_best <- perf_me[,c("SVR")]-apply(perf_me[,c("GB","AB","ANN","BRR","SVR","RF","LASSO")],1,min)

# Get the best performing algorithms per dataset and normalize their error, write to original dataframe
perf_me$diff_algo <- diff_best
perf_me$diff_algo_order <- (diff_best)/max_rt

# Make a barplot with the normalized median error of the best algorithms compared to SVM
barplot((perf_me$diff_algo[order(perf_me$diff_algo_order)]/max_rt[order(perf_me$diff_algo_order)])*100,
        names=rownames(perf_me)[order(perf_me$diff_algo_order)],las=2,cex.names=0.8,
        main=paste("Best out of seven - SVR","\n Average increase (%):",round(mean((perf_me$diff_algo[order(perf_me$diff_algo_order)]/max_rt[order(perf_me$diff_algo_order)])*100),2)),
        ylim=c(0,8),ylab="")
print((perf_me$diff_algo[order(perf_me$diff_algo_order)]/max_rt[order(perf_me$diff_algo_order)])*100)

# Close dev if figure was written
if (plot_pdf | plot_eps){
  dev.off()
}

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"xgb_pairwise.eps"), width = 8, height = 12, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Plot settings
par(mfrow=c(4,2))
par(mar=c(5, 4, 0.5, 2) + 0.1)

# Vector that holds combinations that have already been analyzed (counters A -> B and B -> A comparisons)
analyzed <- c()

# Get the maximum retention time per experiment based on the last identification
max_rt <- tapply(cv_preds$tr, cv_preds$experiment, max)[rownames(perf_me)]
par(mfrow=c(3,2))

# Iterate over all algorithms, but a is always GB; then make a pairwise comparison
for (a in c("GB")){
  for (b in c("GB","AB","ANN","BRR","SVR","RF","LASSO")){
    # Check if already analyzed, if not do the analysis
    if (a != b & !(paste(a,b,sep="_") %in% analyzed) & !(paste(b,a,sep="_") %in% analyzed)){
      par(mar=c(8, 5, 4, 0) + 0.1)
      # These are on the sides of plotting window and should have different margins
      if (b %in% c("ANN","SVR","LASSO")) {
        par(mar=c(8, 2, 4, 2) + 0.1)
      }
      
      # Perform a pairwise comparison of the median absolute error between algorithm a and b
      perf_me$diff_algo <- perf_me[,b]-perf_me[,a]
      perf_me$diff_algo_order <- (perf_me[,b]-perf_me[,a])/max_rt
      barplot((perf_me$diff_algo[order(perf_me$diff_algo_order)]/max_rt[order(perf_me$diff_algo_order)])*100,
              names=rownames(perf_me)[order(perf_me$diff_algo_order)],las=2,cex.names=0.8,
              main=paste(b," - ",a,"\n Average increase (%):",round(mean((perf_me$diff_algo[order(perf_me$diff_algo_order)]/max_rt[order(perf_me$diff_algo_order)])*100),2)),
              ylim=c(-5,10),ylab="Difference in median error \n relative to the total elution time  (%)")
      print((perf_me$diff_algo[order(perf_me$diff_algo_order)]/max_rt[order(perf_me$diff_algo_order)])*100)
      analyzed <- c(analyzed,paste(a,b,sep="_"))
    }
  }
}

# Close dev if figure was written
if (plot_pdf | plot_eps){
  dev.off()
}

# Write eps?
if (plot_eps){
  postscript(paste0(output_dir,"remaining_pairwise.eps"), width = 8, height = 12, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Create plot window with 5 rows and 3 columns
par(mfrow=c(5,3))

# Iterate over all algorithms; then make a pairwise comparison
for (a in c("AB","SVR","BRR","LASSO","RF","ANN")){
  for (b in c("GB","SVR","ANN","BRR","AB","RF","LASSO")){
    # Check if already analyzed, if not do the analysis
    if (a != b & !(paste(a,b,sep="_") %in% analyzed) & !(paste(b,a,sep="_") %in% analyzed)){
      par(mar=c(8, 2, 4, 2) + 0.1)
      
      # Change some of the margins if plots are on the edges of the plotting window
      if (a == "AB" && b == "SVR"){
        par(mar=c(8, 5, 4, 2) + 0.1)
        
      }
      if (a == "AB" && b == "RF"){
        par(mar=c(8, 5, 4, 2) + 0.1)
      }
      if (a == "SVR" && b == "BRR"){
        par(mar=c(8, 5, 4, 2) + 0.1)
      }
      if (a == "BRR" && b == "ANN"){
        par(mar=c(8, 5, 4, 2) + 0.1)
      }
      if (a == "LASSO" && b == "ANN"){
        par(mar=c(8, 5, 4, 2) + 0.1)
      }
      
      # Get the difference in absolute median error and normalize using the maximum retention time
      perf_me$diff_algo <- perf_me[,b]-perf_me[,a]
      perf_me$diff_algo_order <- (perf_me[,b]-perf_me[,a])/max_rt
      
      # Get the performance
      perf <- (perf_me$diff_algo[order(perf_me$diff_algo_order)]/max_rt[order(perf_me$diff_algo_order)])*100
      
      # Plot the performance difference
      barplot(perf,
              names=rownames(perf_me)[order(perf_me$diff_algo)],las=2,cex.names=0.7,
              main=paste(b," - ",a,"\n Average increase (%):",round(mean(perf),3)),
              ylim=c(-10,10),ylab="Difference median error relative\n to the total elution time (%)")
      
      # Add to analyze so we do not analyze it again
      analyzed <- c(analyzed,paste(a,b,sep="_"))
    }
  }
}

# Close dev if figure was written
if (plot_pdf | plot_eps){
  dev.off()
}

#########################################################################################
#                                                                                       #
# Plot the CV results - small feature set                                               #
#                                                                                       #
#########################################################################################

#####################################################
#                                                   #
# Plot the correlation of errors with each other    #
#                                                   #
#####################################################

# Read the CV results for the 151 feature set
cv_preds <- read.csv("data/predictions_algo_verbose_small_cv_name.csv")

par(mfrow=c(1,1))

# Dataframe init with the SVM as initial start
error_df = data.frame(rem=matrix(nrow = length(cv_preds[cv_preds$algo=="SVR","tr"])))
for (a in c("SVR","GB","BRR","AB","LASSO","RF","ANN")){
  error_df[a] = cv_preds[cv_preds$algo==a,"tr"]-cv_preds[cv_preds$algo==a,"pred"]
}

# Remove the first column because it will contain "NA"
error_df$rem <- NULL

# Correlation plot between the different algorithms
corrplot(cor(error_df,method="spearman"),method="number",cl.lim=c(0,1.0),tl.col="black")

###########################################################################
#                                                                         #
# Plot the error per dataset for each algorithm for a pairwise comparison #
#                                                                         #
###########################################################################

# Initialize vectors that will hold the performance measures per dataset and algorithm
t_algo_cor <- c()
t_algo_mae <- c()
t_algo_me <- c()

# Iterate over the experiments and algorithms and use different performance measures 
for (e in unique_experi){
  for (a in unique_algo){
    temp_cv_preds <- cv_preds[cv_preds$experiment == e & cv_preds$algo == a,]
    t_algo_mae <- c(t_algo_mae,sum(abs(temp_cv_preds$tr-temp_cv_preds$pred),na.rm=T)/length(temp_cv_preds$tr))
    t_algo_cor <- c(t_algo_cor,cor(temp_cv_preds$tr,temp_cv_preds$pred))
    t_algo_me <- c(t_algo_me,median(abs(temp_cv_preds$tr-temp_cv_preds$pred),na.rm=T))
    
  }
}

# Make sure the performance measures are put in a matrix with the correct algorithm and experiment name
perf_mae <- data.frame(matrix(t_algo_mae,ncol=length(unique_algo),byrow=T))
colnames(perf_mae) <- unique_algo
rownames(perf_mae) <- unique_experi

perf_me <- data.frame(matrix(t_algo_me,ncol=length(unique_algo),byrow=T))
colnames(perf_me) <- unique_algo
rownames(perf_me) <- unique_experi
perf_me_small <- perf_me

perf_cor <- data.frame(matrix(t_algo_cor,ncol=length(unique_algo),byrow=T))
colnames(perf_cor) <- unique_algo
rownames(perf_cor) <- unique_experi

# Set plotting window with 3 rows and 2 columns
par(mfrow=c(3,2))

# Vector that will hold previously analyzed combinations of algorithms
analyzed <- c()

# Iterate over all combinations of algorithms and do a pairwise comparison
for (a in c("SVR","GB","BRR","AB","RF","ANN")){
  for (b in c("SVR","GB","BRR","AB","RF","ANN")){
    # If not analyzed yet; make a pairwise comparison between the algorithms
    if (a != b & !(paste(a,b,sep="_") %in% analyzed) & !(paste(b,a,sep="_") %in% analyzed)){
      analyzed <- c(analyzed,paste(a,b,sep="_"))
      
      # Make a pairwise comparison in terms of mean absolute error
      plot(perf_mae[,a],perf_mae[,b],xlab=a,ylab=b,pch=10,ylim=c(0,500),xlim=c(0,500),cex=0.5)
      abline(0,1,lty=2)
    }
  }
}

# Set plotting window with 3 rows and 2 columns
par(mfrow=c(3,2))

# Vector that will hold previously analyzed combinations of algorithms
analyzed <- c()

# Iterate over all combinations of algorithms and do a pairwise comparison
for (a in c("SVR","GB","BRR","AB","RF","ANN")){
  for (b in c("SVR","GB","BRR","AB","RF","ANN")){
    # If not analyzed yet; make a pairwise comparison between the algorithms
    if (a != b & !(paste(a,b,sep="_") %in% analyzed) & !(paste(b,a,sep="_") %in% analyzed)){
      analyzed <- c(analyzed,paste(a,b,sep="_"))
      
      # Make a pairwise comparison in terms of median absolute error
      plot(perf_me[,a],perf_me[,b],xlab=a,ylab=b,pch=10,ylim=c(0,500),xlim=c(0,500),cex=0.5)
      abline(0,1,lty=2)
    }
  }
}

# Set plotting window with 3 rows and 2 columns
par(mfrow=c(3,2))

# Vector that will hold previously analyzed combinations of algorithms
analyzed <- c()

# Iterate over all combinations of algorithms and do a pairwise comparison
for (a in c("SVR","GB","BRR","AB","RF","ANN")){
  for (b in c("SVR","GB","BRR","AB","RF","ANN")){
    # If not analyzed yet; make a pairwise comparison between the algorithms
    if (a != b & !(paste(a,b,sep="_") %in% analyzed) & !(paste(b,a,sep="_") %in% analyzed)){
      analyzed <- c(analyzed,paste(a,b,sep="_"))
      
      # Make a pairwise comparison in terms of correlation between predicted and observed retention times
      plot(perf_cor[,a],perf_cor[,b],xlab=a,ylab=b,pch=10,ylim=c(0,1),xlim=c(0,1),cex=0.5)
      abline(0,1,lty=2)
    }
  }
}

############################################################
#                                                          #
# Get the best and worst performing algorithms per dataset #
#                                                          #
############################################################

# Plot the lowest and highest errors per dataset
par(mfrow=c(1,1))

# Median absolute error
barplot(table(colnames(perf_me)[apply(perf_me,1,which.max)]),ylab="Highest median error on dataset (#)", las=3)
barplot(table(colnames(perf_me)[apply(perf_me,1,which.min)]),ylab="Lowest median error on dataset (#)", las=3)

# Mean absolute error
barplot(table(colnames(perf_mae)[apply(perf_mae,1,which.max)]),ylab="Highest mean error on dataset (#)", las=3)
barplot(table(colnames(perf_mae)[apply(perf_mae,1,which.min)]),ylab="Lowest mean error on dataset (#)", las=3)

# Correlation between predicted and observed retention times
barplot(table(colnames(perf_cor)[apply(perf_cor,1,which.max)]),ylab="Highest correlation on dataset (#)", las=3)
barplot(table(colnames(perf_cor)[apply(perf_cor,1,which.min)]),ylab="Lowest correlation on dataset (#)", las=3)

######################################################
#                                                    #
# Make a pairwise comparison between XGBoost and SVM #
#                                                    #
######################################################

# Plot the median absolute error and mean absolute error per dataset for XGBoost and SVM
par(mfrow=c(1,2))
plot(perf_me$SVR,perf_me$GB,pch=10,xlim=c(0,320),ylim=c(0,320),xlab="SVR (median absolute error(s))",ylab="GB (median absolute error (s))")
abline(0,1)
plot(perf_mae$SVR,perf_mae$GB,pch=10,xlim=c(0,320),ylim=c(0,320),xlab="SVR (mean absolute error(s))",ylab="GB (mean absolute error (s))")
abline(0,1)

# Calculate the number of molecules per dataset, division by 7  is performed because there are 7 algorithms
perf_me <- cbind(perf_me,Freq=c((table(cv_preds$experiment)/7)[rownames(perf_me)]))
perf_mae <- cbind(perf_mae,Freq=c((table(cv_preds$experiment)/7)[rownames(perf_mae)]))

# Plot the difference in SVM and XGBoost vs the number of training examples
par(mfrow=c(1,2))
plot((perf_me$SVR-perf_me$GB),perf_me$Freq,pch=10,
     xlab="xgb - SVM (difference in median error)",ylab="Total number of examples")
abline(v=0,lty=2)
abline(h=100,lty=2)

# Plot the difference in SVM and XGBoost vs the number of training examples
plot((perf_mae$SVR-perf_mae$GB),perf_mae$Freq,pch=10,
     xlab="xgb - SVM (difference in mean error)",ylab="")
abline(v=0,lty=2)
abline(h=100,lty=2)


#######################################################################
#                                                                     #
# Make a pairwise comparison between XGBoost and the other algorithms #
#                                                                     #
#######################################################################

# Plot eps?
if (plot_eps){
  postscript(paste0(output_dir,"pairwise_comparison_small.eps"), width = 14, height = 21, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Plot settings
par(mfrow=c(6,4))
par(mar=c(5, 4, 0.5, 2) + 0.1)

# Vector that holds combinations that have already been analyzed (counters A -> B and B -> A comparisons)
analyzed <- c()

# Get the maximum retention time per experiment based on the last identification
max_rt <- tapply(cv_preds$tr, cv_preds$experiment, max)[rownames(perf_me)]

# Iterate over all combinations of algorithms and do a pairwise comparison
for (a in c("AB","SVR","BRR","LASSO","RF","ANN")){
  for (b in c("GB","SVR","ANN","BRR","AB","RF","LASSO")){
    # If not analyzed yet; make a pairwise comparison between the algorithms
    if (a != b & !(paste(a,b,sep="_") %in% analyzed) & !(paste(b,a,sep="_") %in% analyzed)){
      # Change margins for algorithms at the sides...
      par(mar=c(8, 2, 4, 2) + 0.1)
      if (a == "AB" && b == "GB"){
        par(mar=c(8, 5, 4, 2) + 0.1)
        
      }
      if (a == "AB" && b == "RF"){
        par(mar=c(8, 5, 4, 2) + 0.1)
      }
      if (a == "SVR" && b == "BRR"){
        par(mar=c(8, 5, 4, 2) + 0.1)
      }
      if (a == "BRR" && b == "ANN"){
        par(mar=c(8, 5, 4, 2) + 0.1)
      }
      if (a == "LASSO" && b == "ANN"){
        par(mar=c(8, 5, 4, 2) + 0.1)
      }
      
      if (a == "ANN" && b == "GB"){
        par(mar=c(8, 5, 4, 2) + 0.1)
      }
      
      # Calculate the median error and normalized median error
      perf_me$diff_algo <- perf_me[,b]-perf_me[,a]
      perf_me$diff_algo_order <- (perf_me[,b]-perf_me[,a])/max_rt
      
      perf <- (perf_me$diff_algo[order(perf_me$diff_algo_order)]/max_rt[order(perf_me$diff_algo_order)])*100
      
      # Plot the pairwise comparison
      barplot(perf,
              names=rownames(perf_me)[order(perf_me$diff_algo)],las=2,cex.names=0.7,
              main=paste(b," - ",a,"\n Average increase (%):",round(mean(perf),3)),
              ylim=c(-10,10),ylab="Difference median error relative\n to the total elution time (%)")
      
      # Add pair to analyzed combinations
      analyzed <- c(analyzed,paste(a,b,sep="_"))
    }
  }
}

# Close dev if figure was plotted
if (plot_eps){
  dev.off()
}

# Set plotting window to 1 row with two columns
par(mfrow=c(1,2))

# Scatter plot with the median an mean errors of SVR and GB as coordinates
plot(perf_me$SVR,perf_me$GB,pch=10,xlim=c(0,300),ylim=c(0,300),xlab="SVM (median absolute error(s))",ylab="XGBoost (mean absolute error (s))")
abline(0,1)
plot(perf_mae$SVR,perf_mae$GB,pch=10,xlim=c(0,300),ylim=c(0,300),xlab="SVM (mean absolute error(s))",ylab="XGBoost (mean absolute error (s))")
abline(0,1)

# Plot eps?
if (plot_eps){
  postscript(paste0(output_dir,"xgb_comparison_big_small_feature_set.eps"), width = 6, height = 6, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Make a pairwise comparison in the median absolute error between GB with eleven features and 151 features
perf_diff_big_small <- (((perf_me_small$GB[unique_experi]-perf_me_big$GB[unique_experi]))/max_rt[unique_experi])*100
par(mfrow=c(1,1))
barplot(perf_diff_big_small[order(perf_diff_big_small)],
        names=unique_experi[order(perf_diff_big_small)],las=2,cex.names=0.7,
        main=paste("GB 11 features - GB 151 features \n (ME improvement: ",round(mean(perf_diff_big_small),2)," %)",sep=""),
        ylab="Difference median error relative to the total elution time (%)")

# Close dev if figure was plotted
if (plot_pdf | plot_eps){
  dev.off()
}

#######################################################################################################
#                                                                                                     #
# Plot the preds vs. exp. measured mainly focused on molecules themselves and lose the dataset level  #
#                                                                                                     #
#######################################################################################################

# Read the CV results for the 151 feature set
cv_preds <- read.csv("data/predictions_algo_verbose_big_cv_name.csv")

# Get the experiments from the CV set
unique_experi <- unique(cv_preds$experiment)

# Get the algos from the CV set
unique_algo <- unique(cv_preds$algo)

# Get rid of ridiculous low predictions
cv_preds$pred[cv_preds$pred < -1e+04] <- 0

# Iterate over the algorithms and plot predictions vs. experimentally measured rt in a scatter plot and a density plot
for (algo in unique_algo){
  plot(cv_preds$tr[cv_preds$algo== algo],cv_preds$pred[cv_preds$algo==algo],cex=0.1,main=algo)
  plot(density((cv_preds$tr[cv_preds$algo== algo]-cv_preds$pred[cv_preds$algo==algo])/max_rt[cv_preds$experiment],adjust=10),cex=0.1,main=algo,xlim=c(-1,1))
}

# Iterate over the algorithms and plot them in the same density plot
plot(density((cv_preds$tr[cv_preds$algo== algo]-cv_preds$pred[cv_preds$algo==algo])/max_rt[cv_preds$experiment],n=50000),cex=0.1,main=algo,xlim=c(-0.25,0.25),ylim=c(0,11))
for (algo in unique_algo){
  lines(density((cv_preds$tr[cv_preds$algo== algo]-cv_preds$pred[cv_preds$algo==algo])/max_rt[cv_preds$experiment],n=50000),cex=0.1,main=algo,xlim=c(-1,1))
}

# Function to calculate the error for a given fraction of molecules that fall under a specified threhsold.
#
# Inputs:
#   errors - dataframe consisting of the columns "experiment" (indicating the dataset) 
#   "num_train" (indicating the number of training examples) "algo" (indicating the algorithm) 
#   "perf" (indicating the performance for a previously selected measure)
#   range_threshold_div - range of thresholds used for stepping between the fractions
# Returns:
#   vector containing the fractions for given tresholds
#
plot_threshold_c <- function(errors,range_threshold_div=10){
  ret_vec = c()
  errors <- errors[order(errors)]
  div_steps = length(errors)/range_threshold_div
  for (t in seq(div_steps+1,length(errors),div_steps)){
    ret_vec <- c(ret_vec,mean(errors[t-div_steps:t]))
  }
  return(ret_vec*100)
}  

# Maximum rt measured for a dataset
max_rt <- tapply(cv_preds$tr, cv_preds$experiment, max)[rownames(perf_me)]

# Create an empty plot to plot the threshold curves
plot(1,xlim=c(0,10),ylim=c(0,5),col="white")

# Iterate over the algorithms to create the threshold curves
for (algo in unique_algo){
  print(plot_threshold_c((abs(cv_preds$tr[cv_preds$algo== algo]-cv_preds$pred[cv_preds$algo== algo])/max_rt[cv_preds$experiment[cv_preds$algo== algo]])))
  lines(seq(1,9,1),plot_threshold_c((abs(cv_preds$tr[cv_preds$algo== algo]-cv_preds$pred[cv_preds$algo== algo])/max_rt[cv_preds$experiment[cv_preds$algo== algo]])))
}

###########################################################################
#                                                                         #
# Plot threshold curves for blending predictions of different algorithms  #
#                                                                         #
###########################################################################


# Plot eps?
if (plot_eps){
  postscript(paste0(output_dir,"blending_perf.eps"), width = 6, height = 6, horizontal = FALSE, 
             onefile = FALSE, paper = "special", colormodel = "cmyk", 
             family = "Times")
}

# Set a limit in % error
max_error <- 20

# Define the different colors of the algorithms
col_sep <- c("black","grey","magenta","green","red","blue","brown","orange")

# Define the initial plot that will contain the threshold curve
plot(1,xlim=c(0.0,max_error),ylim=c(0,100.0),ylab="Metabolites under threshold (%)",xlab="Error threshold relative to the total elution time (%)",col="white")

# Start with the first algorithm
i <- 1

# Initialize a vector of blended predictions; length will be the same as for example SVR
cv_preds$blend <- rep(0,sum(cv_preds$algo == "SVR"))
cv_preds_mat <- c()
algo_perf <- c()

# Iterate over the algorithms
for (algo in unique_algo){
  # If the algorithm is part of these three, make sure we can blend them later on
  if (algo %in% c("SVR","GB","ANN")){
    cv_preds_mat <- c(cv_preds_mat,cv_preds$pred[cv_preds$algo== algo])
  }
  
  # Calculate the (overall) error and normalize
  algo_perf <- c(algo_perf,mean(abs(cv_preds$tr[cv_preds$algo==algo]-cv_preds$pred[cv_preds$algo==algo])/max_rt[cv_preds$experiment[cv_preds$algo==algo]]))
  
  # Calculate the (individual per molecule) errors and normalize
  error <- (cv_preds$tr[cv_preds$algo== algo]-cv_preds$pred[cv_preds$algo== algo])/max_rt[cv_preds$experiment[cv_preds$algo== algo]]
  error <- abs(error * 100)
  
  # Iterate over different error thresholds and determine the fractions...
  new_err <- c()
  for (j in seq(0,max_error,0.1)) {
    new_err <- c(new_err,sum(error < j)/length(error))
  }
  
  # Plot the error fraction
  lines(seq(0,max_error,0.1),new_err*100,col=col_sep[i],lty=i,lwd=1.5)
  
  # Colour of the next algorithm...
  i <- i + 1
}

# To do the blending make a matrix where each row is an individual molecule
cv_preds_mat <- matrix(cv_preds_mat,ncol=3)

# Calculate the (overall) error and normalize, for the blended version
algo_perf <- c(algo_perf,mean(abs(apply(cv_preds_mat,1,mean)-cv_preds$tr[cv_preds$algo== "SVR"])/as.numeric(max_rt[cv_preds$experiment[cv_preds$algo=="SVR"]])))
error <- (apply(cv_preds_mat,1,mean)-cv_preds$tr[cv_preds$algo== "SVR"])/as.numeric(max_rt[cv_preds$experiment[cv_preds$algo=="SVR"]])
error <- abs(error * 100)

# Iterate over different error thresholds and determine the fractions...
new_err <- c()
for (j in seq(0,max_error,0.1)) {
  new_err <- c(new_err,sum(error < j)/length(error))
}

# Plot the error fraction
lines(seq(0,max_error,0.1),new_err*100,col="orange",lty=8,lwd=1.5)

# Plot a legend
legend(10,60,c(as.character(unique_algo),"Blended"),lty=1:8,col=col_sep)

# Close dev if figure was plotted
if (plot_eps){
 dev.off() 
}

# Initialize a vector of blended predictions and a matrix that will hold all predictions
blended_perf <- c()
perf_mat <- c()

# Do this analysis per dataset, so iterate over the experiments
for (experi in unique_experi){
  # Get the predictions for a specific dataset
  cv_preds_experi <- cv_preds[cv_preds$experiment == experi,]
  cv_preds_mat <- c()
  
  # Iterate over all algorithms
  for (algo in unique_algo){
    # If the algorithm is part of one of these three; save for later use of blending
    if (algo %in% c("SVR","GB","ANN")){
      cv_preds_mat <- c(cv_preds_mat,cv_preds_experi$pred[cv_preds_experi$algo== algo])
    }
    # Add the mean absolute error to the total performance
    perf_mat <- c(perf_mat,mean(abs(cv_preds_experi$pred[cv_preds_experi$algo== algo]-cv_preds_experi$tr[cv_preds_experi$algo== algo])))
  }
  # Calculate the mean absolute error for the blended predictions
  cv_preds_mat <- matrix(cv_preds_mat,ncol=3)
  blended_perf <- (mean(abs(apply(cv_preds_mat,1,mean)-cv_preds_experi$tr[cv_preds_experi$algo== "SVR"])))
  perf_mat <- c(perf_mat,blended_perf)
}

# Add the blended algorithm
unique_algo_blended <- c(as.character(unique_algo),"Blended")

# Make a matrix out of the predictions
perf_mat <- matrix(perf_mat,ncol=8,byrow=TRUE)
row.names(perf_mat) <- unique_experi
colnames(perf_mat) <- unique_algo_blended

# Calculate the average rank of the mean absolute error per dataset
apply(apply(perf_mat, 1, rank),1,mean)

# Initialize a vector of blended predictions and a matrix that will hold all predictions
blended_perf <- c()
perf_mat <- c()

# Do this analysis per dataset, so iterate over the experiments
for (experi in unique_experi){
  # Get the predictions for a specific dataset
  cv_preds_experi <- cv_preds[cv_preds$experiment == experi,]
  cv_preds_mat <- c()
  
  # Iterate over all algorithms
  for (algo in unique_algo){
    # If the algorithm is part of one of these three; save for later use of blending
    if (algo %in% c("SVR","GB","ANN")){
      cv_preds_mat <- c(cv_preds_mat,cv_preds_experi$pred[cv_preds_experi$algo== algo])
    }
    # Add the median absolute error to the total performance
    perf_mat <- c(perf_mat,median(abs(cv_preds_experi$pred[cv_preds_experi$algo== algo]-cv_preds_experi$tr[cv_preds_experi$algo== algo])))
  }
  # Calculate the mean absolute error for the blended predictions
  cv_preds_mat <- matrix(cv_preds_mat,ncol=3)
  blended_perf <- (median(abs(apply(cv_preds_mat,1,mean)-cv_preds_experi$tr[cv_preds_experi$algo== "SVR"])))
  perf_mat <- c(perf_mat,blended_perf)
}

# Add the blended algorithm
unique_algo_blended <- c(as.character(unique_algo),"Blended")

# Make a matrix out of the predictions
perf_mat <- matrix(perf_mat,ncol=8,byrow=TRUE)
row.names(perf_mat) <- unique_experi
colnames(perf_mat) <- unique_algo_blended

# Calculate the average rank of the mean absolute error per dataset
apply(apply(perf_mat, 1, rank),1,mean)