"""
Copyright 2017 Robbin Bouwmeester

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This code is used to train retention time predictors and evaluate the
effect of different training set sizes.

Library versions:

Python 2.7.13
xgboost.__version__ = '0.6'
sklearn.__version__ = '0.19.0'
scipy.__version__ = '0.19.1'
numpy.__version__ = '1.13.3'
pandas.__version__ = '0.20.3'

This project was made possible by MASSTRPLAN. MASSTRPLAN received funding 
from the Marie Sklodowska-Curie EU Framework for Research and Innovation 
Horizon 2020, under Grant Agreement No. 675132.
"""

__author__ = "Robbin Bouwmeester"
__copyright__ = "Copyright 2017"
__credits__ = ["Robbin Bouwmeester","Prof. Lennart Martens","Sven Degroeve"]
__license__ = "Apache License, Version 2.0"
__version__ = "1.0"
__maintainer__ = "Robbin Bouwmeester"
__email__ = "Robbin.bouwmeester@ugent.be"

# Native imports 
import subprocess
from random import shuffle
import random

# Internal imports
from train import train_func
from apply import apply_models

# Machine learning imports
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

# Data imports
import pandas as pd

# Global variable that holds additive strings for filenames (e.g. learning curve)
adds=["_r1","_r1","_r1","_r1","_r1","_r1","_r1","_r1",
	  "_r2","_r2","_r2","_r2","_r2","_r2","_r2","_r2",
	  "_r3","_r3","_r3","_r3","_r3","_r3","_r3","_r3",
	  "_r4","_r4","_r4","_r4","_r4","_r4","_r4","_r4",
	  "_r5","_r5","_r5","_r5","_r5","_r5","_r5","_r5",
	  "_r6","_r6","_r6","_r6","_r6","_r6","_r6","_r6",
	  "_r7","_r7","_r7","_r7","_r7","_r7","_r7","_r7",
	  "_r8","_r8","_r8","_r8","_r8","_r8","_r8","_r8",
	  "_r9","_r9","_r9","_r9","_r9","_r9","_r9","_r9",
	  "_r10","_r10","_r10","_r10","_r10","_r10","_r10","_r10"]

# Grid to walk through in terms of training sizes; so will go from 20 to 160 with increments of 20 for 10 times
n_all = [20,40,60,80,100,120,140,160,
		 20,40,60,80,100,120,140,160,
		 20,40,60,80,100,120,140,160,
		 20,40,60,80,100,120,140,160,
		 20,40,60,80,100,120,140,160,
		 20,40,60,80,100,120,140,160,
		 20,40,60,80,100,120,140,160,
		 20,40,60,80,100,120,140,160,
		 20,40,60,80,100,120,140,160,
		 20,40,60,80,100,120,140,160]

def remove_low_std(X,std_val=0.01):
	"""
	Function that marks columns in the feature matrix based on their standard deviation.

    Parameters
    ----------
    X : pandas DF
        feature matrix
	std_val : float
		minimal standard deviation for one feature to not be marked
		
    Returns
    -------
    list
		feature names that do not meet the standard deviation requirement
    """
	rem_f = []
	
	# Get the standard deviation per column
	std_dist = X.std(axis=0)
	
	# If below filter; add to list
	rem_f.extend(list(std_dist.index[std_dist<std_val]))
	return(rem_f)

def remove_high_cor(X,upp_cor=0.98,low_cor=-0.98):
	"""
	Function that marks columns in the feature matrix based on their Pearson correlation.
	One of the highly correlation columns is taken by random.

    Parameters
    ----------
    X : pandas DF
        feature matrix
	upp_cor : float
		upper limit Pearson correlation
	low_cor : float
		lower limit Pearson correlation
	
    Returns
    -------
    list
		feature names that do not meet the correlation requirement
    """
	rem_f = []
	keep_f = []
	
	# Calculate correlations between all columns
	new_m = X.corr()
	new_m = list(new_m.values)
	
	# Iterate over the correlations, if it does not meed the requirement; remove
	for i in range(len(new_m)):
		for j in range(len(new_m[i])):
			# Do not remove yourself...
			if i == j: continue

			if new_m[i][j] > upp_cor or new_m[i][j] < low_cor:
				# Test if the feature ws not previously selected for keeping...
				if X.columns[j] not in keep_f:
					rem_f.append(X.columns[j])
					keep_f.append(X.columns[i])
	return(rem_f)

def sel_features(infile,verbose=True,remove_std=True,remove_cor=True,std_val=0.01,upp_cor=0.99,low_cor=-0.99,ignore_cols=["system","IDENTIFIER","time"]):
	"""
	Function used for feature selection based on standard deviation within a feature and correlation between features

    Parameters
    ----------
    infile : pandas DF
        feature matrix
	verbose : boolean
		print information while executing
	remove_std : boolean
		flag; filter of features based on standard deviation
	remove_cor : boolean
		flag; filter of features based on correlation
	std_val : float
		value for filter based on standard deviation
	upp_cor : float
		upper limit for filtering on correlation
	low_cor : float
		lower limit for filtering on correlation
	ignore_cols : list
		ignore these columns for filtering
		
	Returns
    -------
    pandas DF
		dataframe with columns removed
	pandas series
		series with the column names that were retained
    """
	
	# If we need to filter on columns...
	if remove_std or remove_cor:
		rem_f = []
		
		# Apply filtering
		if remove_std: rem_f.extend(remove_low_std(infile,std_val=std_val))
		if remove_cor: rem_f.extend(remove_high_cor(infile,))
	
		rem_f = list(set(rem_f))
		
		# Remove the ignore columns from the to remove list
		[rem_f.remove(x) for x in rem_f if x in ignore_cols]
		
		if verbose: print "Removing the following features: %s" % rem_f
		
		# Replace original dataframe with filtered features
		infile.drop(rem_f, axis=1, inplace=True)
	return(infile,infile.columns)

def get_sets(infile):
	"""
	Function that maps a dataset name to a sliced version of a dataframe

    Parameters
    ----------
    infile : pandas DF
        dataframe containing the features and objective values
		
    Returns
    -------
    dict
		dictionary that maps between datasets and a sliced version of a dataframe
    """
	sets_dict = {}
	
	# Get the unique datasets
	unique_systems = list(set(infile["system"]))
	
	# Iterate over the datasets and get slices
	for us in unique_systems:
		sets_dict[us] = infile[infile["system"]==us]

	return(sets_dict)

def remove_models(k,n):
	"""
	System call to remove pickled model files.

    Parameters
    ----------
    k : str
        dataset name coupled to the model pickle
	n : str
		unique name coupled to the model pickle
		
    Returns
    -------
    
    """
	cmd = "rm -rf mods_l1/%s_*%s.pickle" % (k,n)
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
	p.communicate()

def scale_cols(selected_set):
	"""
	Scale features of a matrix

    Parameters
    ----------
    selected_set : pandas DF
        dataframe containing the features and objective values
		
    Returns
    -------
    pandas DF
		dataframe with scaled features
    """
	
	# Drop the features that we need to keep for learning
	X = selected_set.drop(["time","IDENTIFIER","system"],axis=1)
	
	# Scale features
	X_scaled = pd.DataFrame(scale(X))
	X_scaled.columns = X.columns
	X_scaled.index = X.index
	
	# Set scaled values to original dataframe
	for c in X_scaled.columns:
		selected_set[c] = X_scaled[c]
	return(selected_set)

def main(infilen="train/retmetfeatures_removed_duplicates.csv",scale=True):
	"""
	Main function for training the retention time predictor.

    Parameters
    ----------
    infilen : str
        csv file that is used for training
	feat_filename : str
		txt file that contains the features that need to be used for training
	scale : boolean
		flag indicating if features need to be scaled
		
    Returns
    -------

    """
	
	global adds
	global n_all

	infile = pd.read_csv(infilen)
	
	# Try to read the indicated feat filename, if it does not exist; create one	
	try:
		keep_f = [x.strip() for x in open("features/selected_features_big.txt").readlines()]
		infile = infile[keep_f]
	except IOError:
		infile,keep_f = sel_features(infile)
		outfile = open("features/selected_features_big.txt","w")
		outfile.write("\n".join(list(keep_f)))
		outfile.close()
	
	# Get all the slices and unique datasets
	sets = get_sets(infile)

	# Iterate over the datasets
	for k in sets.keys():
		selected_set = sets[k]
		select_index = range(len(selected_set.index))
		for ind in range(len(n_all)):
			# How many instances should we use?
			n = n_all[ind]
			
			# Check if dataset meets number of instances requirements
			if len(selected_set.index) < 159: continue
			if scale: selected_set = scale_cols(selected_set)
			
			# Shuffle the indexes for the first step; after not any more because we want to add to the original sets
			if n == 20: shuffle(select_index)
			
			# Define train and test sets; test is after full train is selected
			train = selected_set.iloc[select_index[0:n],]
			test = selected_set.iloc[select_index[161:],]
			
			# Define CV for hyperparameter optimization
			cv = KFold(len(train.index),n_folds=10)

			print "Training L1 %s,%s,%s" % (k,n,adds[ind])
			
			# Train regression model
			preds_own = train_func(train,names=[k,k,k,k,k,k,k],adds=[n,n,n,n,n,n,n,n],cv=cv)

			print "Applying L1 %s,%s,%s" % (k,n,adds[ind])
			
			# Apply trained regression models
			preds_l1_train,skipped_train = apply_models(train.drop(["time","IDENTIFIER","system"],axis=1),known_rt=train["time"],row_identifiers=train["IDENTIFIER"],skip_cont=[k])
			preds_l1_test,skipped_test = apply_models(test.drop(["time","IDENTIFIER","system"],axis=1),known_rt=test["time"],row_identifiers=test["IDENTIFIER"])			
		
			preds_l1_train = pd.concat([preds_l1_train.reset_index(drop=True), preds_own], axis=1)
			
			# Write results
			outfilel1 = open("test_preds/%s_preds_%s%s.csv" % (k,n,adds[ind]),"w")
			outfilel1train = open("test_preds/%s_preds_train_%s%s.csv" % (k,n,adds[ind]),"w")

			preds_l1_test.to_csv(outfilel1,index=False)
			preds_l1_train.to_csv(outfilel1train,index=False)

			outfilel1.close()
			outfilel1train.close()

			remove_models(k,n)


if __name__ == "__main__":
	random.seed(42)
	main()
