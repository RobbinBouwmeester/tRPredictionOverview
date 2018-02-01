# Set a working directory if needed
#setwd("")

postscript("figs/mw_distribution.eps", width = 7, height = 7, horizontal = FALSE, 
           onefile = FALSE, paper = "special", colormodel = "cmyk", 
           family = "Times")

data <- read.csv("data/concat_df.csv")
hist(data$MolWt,breaks=200,xlab="Molecular weight (Da)",main="Distribution of molecular weight\n for the compiled dataset")
diff_data <- unique(data$system)

mat_d <- c()
for (d in diff_data) {
  mat_d <- c(mat_d,c(d,length(data[data$system == d,]$MolWt),max(data[data$system == d,]$MolWt),min(data[data$system == d,]$MolWt),max(data[data$system == d,]$MolLogP),min(data[data$system == d,]$MolLogP)))
}

dev.off()