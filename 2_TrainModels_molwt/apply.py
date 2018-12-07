"""
This code is used to apply regression models for retention time prediction

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

from os import listdir
from os.path import isfile, join
import pickle
import pandas
from sklearn.preprocessing import maxabs_scale

def apply_models(X,outfile="",model_path="mods/",known_rt=[],row_identifiers=[],skip_cont=[]):
	"""
	Function that applies pickled regression models to a feature matrix

    Parameters
    ----------
    X : pandas DF
        feature matrix
	outfile : str
		minimal standard deviation for one feature to not be marked
	model_path : str
		path for the model pickles
	known_rt : list
		if the function is applied to a feature matrix with known rts this can be indicated with a list of floats
	row_identifiers : list
		list of strings with identifiers for each row
	skip_cont : list
		list of strings with names of models to skip
		
    Returns
    -------
    pandas DF
		predictions from the applied regression models
	list
		list of skipped pickled models
    """
	model_fn = [f for f in listdir(model_path) if isfile(join(model_path, f))]
	preds = []
	t_preds = []
	skipped = []

	if len(row_identifiers) > 0: preds.append(list(row_identifiers))
	if len(known_rt) > 0: preds.append(list(known_rt))

	cnames = []
	if len(row_identifiers)  > 0: cnames.append("IDENTIFIER")
	if len(known_rt)  > 0: cnames.append("time")
	
	# Iterate over the pickled models and make predictions
	for f in model_fn:
		con = False
		
		# Need to skip this model?
		for skip in skip_cont:
			compare = f.split("_")
			compare.pop()
			compare = "_".join(compare)
			if skip == compare:
				skipped.append(f.replace(".pickle",""))
				con = True
		if con: continue
		
		# Open the model
		try:
			with open(join(model_path, f)) as model_f:
				model = pickle.load(model_f)
		except ValueError:
			continue
		
		# Try to make predictions
		try: 
			temp_preds = model.predict(X)
		except:
			continue
		preds.append(temp_preds)
		cnames.append(f.replace(".pickle",""))

	preds = zip(*preds)
	preds = map(list,preds)

	preds = pandas.DataFrame(preds)

	preds.columns = cnames

	return(preds,skipped)