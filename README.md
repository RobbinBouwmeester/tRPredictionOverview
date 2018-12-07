# tRPredictionOverview
Evaluation of multiple machine learning algorithms, number of instances and different features on a vast amount of different data.

# Contact

Robbin.bouwmeester@ugent.be

# File and folder structure
	./

Contains 4 different type of directories with all the data, scripts, models and figures used in the manuscript. The folders are numbered in the order they should be executed to replicate all the experiments.

##################################################################################

	/1_MolDescriptors/

The folders and data needed to calculate the molecular descriptors (features).

	/1_MolDescriptors/rawData/

Folder that contains the raw data from the Aicheler paper, MoNA and PredRet.
	
	/1_MolDescriptors/smilesExtracted/

Folder that contains the extracted SMILES from the Aicheler paper, MoNA and PredRet.
	
	/1_MolDescriptors/smilesToFeatures/

Folder that contains the extracted features from the SMILES.

	/1_MolDescriptors/getf.py 
	
Python script that is used to parse the SMILES and calculate the feature using RDKit

##################################################################################

	/2_TrainModels/

The folders, data and scripts used for fitting the regression models.

	/2_TrainModels/features/

Folder with text files that contain the features used in the models (e.g. the used big and small feature set)

	/2_TrainModels/mods/

Folder where the pickles of the regression models are written.

	/2_TrainModels/test_preds/

Folder where the results from fitting and predicting on a test set are written.
	
	/2_TrainModels/train/

Folder that contains the training data used to fit and evaluate the models.

	/2_TrainModels/initial_train_CV.py

Main python script used to fit and evaluate the regression models. Using CV.

	/2_TrainModels/initial_train_LCURVE.py

Main python script used to fit and evaluate the regression models. Using learning curves.

	/2_TrainModels/train.py

Python script used to train the regression models using different machine learning algorithms.

	/2_TrainModels/apply.py

Python script used to apply the trained the regression models (pickled).

    /2_TrainModels_ANN_depth/
 
The folders and data to test different depths of the ANN.

    /2_TrainModels_molwt/
    
The folders and data to test a baseline model based on mol wt.

    /2_TrainModels_svr_force_kernel/
    
The folders and data to test different SVR kernels.

    /2_TrainModels_time/
    
Get the calculation time for each algorithm.
    
##################################################################################

	/3_ProcessResults/

Folder containing the data and results needed for the evaluation of the different algorithms/features/train sizes.

	/3_ProcessResults/small_feat/

Folder containing the outputs from the regression models using the small feature set.

	/3_ProcessResults/big_feat/

Folder containing the outputs from the regression models using the big feature set.

	/3_ProcessResults/processed/

Folder containing the processed files that are ready for plotting.

	/3_ProcessResults/prepare_plot_files.py

Python script used to make the output ready for plotting.

##################################################################################

	/4_VisualizeResults/

Folder containing the data, code and figures used in all evaluations of the different algorithms/features/train sizes.

	/4_VisualizeResults/data/

Folder containing the data that is used for plotting.
	
	/4_VisualizeResults/figs/

Folder containing the eps figures used for evaluation.

	/4_VisualizeResults/plotMW.R

R script used for plotting the distribution of molecular weight in the aggregated dataset.

	/4_VisualizeResults/visualize.R

R script used for plotting all the results.

    /4_VisualizeResults/simulate_errors_overlap_and_algo_timing.ipynb
    
IPython notebook that analyzes an induced error rate.
