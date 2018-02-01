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

This software was be used to prepare retention time prediction for plotting.
Predictions from CV and randomly picked train/test folds are aggregated
and evaluated.

Library versions:

Python 2.7.13
scipy.__version__ = '0.19.1'
pandas.__version__ = '0.20.3'
numpy.__version__ = '1.14.0'

This project was made possible by MASSTRPLAN. MASSTRPLAN received funding 
from the Marie Sklodowska-Curie EU Framework for Research and Innovation 
Horizon 2020, under Grant Agreement No. 675132.
"""

__author__ = "Robbin Bouwmeester"
__copyright__ = "Copyright 2017"
__credits__ = ["Robbin Bouwmeester","Prof. Lennart Martens","dr. Sven Degroeve"]
__license__ = "Apache License, Version 2.0"
__version__ = "1.0"
__maintainer__ = "Robbin Bouwmeester"
__email__ = "Robbin.bouwmeester@ugent.be"
__status__ = "Ready to run"

# Native imports
from os import listdir
from os.path import isfile, join

# Data processing related imports
import pandas
from scipy.stats import pearsonr
import numpy

def get_max_experi(directory):
	"""
	Function to get the maximum retention time per experiment.

    Parameters
    ----------
    directory : str
		directory with all the datasets to analyze

    Returns
    -------
    dict
    	The maximum retention time is the value and experiment name the key
    """
	experi_max = {}
	for filename in listdir(directory):
		# Do not touch the binarized backup file
		if filename == "backup": continue
		# Ignore any file with the .csv extension
		if ".csv" not in filename: continue

		# Search for this key str to get the experiment name
		exp = filename.split("_preds_")[0]
		infile = pandas.read_csv(directory+filename)
		max_rt = float(max(infile["time"]))

		# If it is already present, just make sure this is the max (can occur for CV folds that are put into seperate files)
		if experi_max.has_key(exp):
			if max_rt > experi_max[exp]:
				experi_max[exp] = max_rt
		else:
			experi_max[exp] = max_rt
	return(experi_max)

def get_extrapol(directory):
	"""
	Function to see what the minimum and maximum retention time is for a specific fold (used to see if extrapolation is a problem)

    Parameters
    ----------
     directory : str
		directory with all the datasets to analyze

		
    Returns
    -------
	dictionary
		Values is a list of two values (minimum and maximum rt) and keys are the experiment name with fold number
    """
	min_max_train = {}
	for filename in listdir(directory):
		if "_train_" not in filename: continue
		exp = filename.split("_preds_")[0]
		rep = filename.split("_train_")[1].replace(".csv","")
		infile = pandas.read_csv(directory+filename)
		max_rt = float(max(infile["time"]))
		min_rt = float(min(infile["time"]))
		min_max_train[exp+"_"+rep] = [min_rt,max_rt]
	return(min_max_train)


def main(outfile_sum_name="predictions_algo_sum.csv",
		  outfile_name="predictions_algo_verbose.csv",
		  directory="test_preds/",
		  normalize=True):
	"""
	Main function to prepare the predicted retention times for furth down-stream analysis

    Parameters
    ----------
    outfile_sum_name : str
        the outfile names used for concatenating results with a specified evaluation metric
    outfile_name : str
        the outfile names used for writing the prediction results of individual molecules
	directory : str
		the directory that contains the predictions that need to be analyzed
	normalize : bool
        flag to indicate if normalization should be applied, based on the maximum rt
		
    Returns
    -------

    """

	outfile = open(outfile_name,"w")
	outfile.write("experiment,identifier,num_train,repeat,algo,tr,pred\n")

	outfile_sum = open(outfile_sum_name,"w")
	outfile_sum.write("experiment,num_train,repeat,algo,perf_type,perf\n")
	
	experi_max = get_max_experi(directory=directory)
	extrapol = get_extrapol(directory=directory)

	total = 0
	extra = 0
	for filename in listdir(directory):
		# Do not analyze CV preds from train
		if "train" in filename: continue
		# Do not touch the binarized backup file
		if filename == "backup": continue
		# Ignore csv files
		if ".csv" not in filename: continue

		# Parse information from the file about the predictions and experiment
		exp = filename.split("_preds_")[0]
		rep = filename.split("_preds_")[1].replace(".csv","")
		num_train = filename.split("_preds_")[1].split("_")[0]
		r = filename.split("_")[-1].replace(".csv","")

		infile = pandas.read_csv(directory+filename)

		# Go over every column in the output file
		for c in infile.columns:
			if c == "IDENTIFIER": 
				identifiers = list(infile[c])
			elif c == "time": 
				tr_list = list(infile[c])
			else:
				# Something has gone wrong here... Debug; should not happen in a normal run!
				if exp not in c:
					print "!!! execution failed for: !!!"
					print exp
					print c
					continue

				preds_algo_extrapol = []
				tr_list_extrapol = []
				
				preds_algo = list(infile[c])
				algo = c.split("_")[-1].rstrip('1234567890.')

				for tr,pred,ident in zip(tr_list,preds_algo,identifiers):
					total += 1
					outfile.write("%s,%s,%s,%s,%s,%s,%s\n" % (exp,ident,num_train,r,algo,tr,pred))

				# Calculate the evaluation metrics
				cor_pred = pearsonr(tr_list,preds_algo)[0]
				mae_pred = sum(map(abs,[a-b for a,b in zip(tr_list,preds_algo)]))/len(tr_list)
				me_pred = numpy.median(map(abs,[a-b for a,b in zip(tr_list,preds_algo)]))

				if normalize:
					outfile_sum.write("%s,%s,%s,%s,correlation,%s\n" % (exp,num_train,r,algo,cor_pred))
					outfile_sum.write("%s,%s,%s,%s,mae,%s\n" % (exp,num_train,r,algo,mae_pred/experi_max[exp]))
					outfile_sum.write("%s,%s,%s,%s,me,%s\n" % (exp,num_train,r,algo,me_pred/experi_max[exp]))
				else:
					outfile_sum.write("%s,%s,%s,%s,correlation,%s\n" % (exp,num_train,r,algo,cor_pred))
					outfile_sum.write("%s,%s,%s,%s,mae,%s\n" % (exp,num_train,r,algo,mae_pred))
					outfile_sum.write("%s,%s,%s,%s,me,%s\n" % (exp,num_train,r,algo,me_pred))

	outfile.close()
	outfile_sum.close()



if __name__ == "__main__":
	main(directory="big_feat/cv/",outfile_sum_name="predictions_algo_sum_big_cv.csv",outfile_name="predictions_algo_verbose_big_cv.csv")
	main(directory="big_feat/lcurve/",outfile_sum_name="predictions_algo_sum_big_lcurve.csv",outfile_name="predictions_algo_verbose_big_lcurve.csv")
	main(directory="big_feat/lcurve/",outfile_sum_name="predictions_algo_sum_big_lcurve_no_norm.csv",outfile_name="predictions_algo_verbose_big_lcurve_no_norm.csv",normalize=False)
	main(directory="small_feat/cv/",outfile_sum_name="predictions_algo_sum_small_cv.csv",outfile_name="predictions_algo_verbose_small_cv.csv")
	main(directory="small_feat/lcurve/",outfile_sum_name="predictions_algo_sum_small_lcurve.csv",outfile_name="predictions_algo_verbose_small_lcurve.csv")
