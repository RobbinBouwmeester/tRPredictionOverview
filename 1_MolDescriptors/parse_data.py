"""
This code is used to extract molecular descriptors using rdkit or CDKDesc from molecule SMILES.

Library versions:

Python 2.7.13
rdkit.__version__ = '2017.09.1'
"""

__author__ = "Robbin Bouwmeester"
__copyright__ = "Copyright 2017"
__credits__ = ["Robbin Bouwmeester","Prof. Lennart Martens","Sven Degroeve"]
__license__ = "Apache License, Version 2.0"
__version__ = "1.0"
__maintainer__ = "Robbin Bouwmeester"
__email__ = "Robbin.bouwmeester@ugent.be"

# Native imports
import csv
from collections import Counter
from subprocess import Popen
from subprocess import PIPE
from os import remove

# Rdkit imports
from rdkit import Chem
from rdkit.Chem import Descriptors

def rdkit_descriptors(mol):
	"""
	Get the molecular descriptors by iterating over all possible descriptors.

    Parameters
    ----------
    mol : object
        An rdkit molecules object.
		
    Returns
    -------
    dict
		Dictionary that contains the molecular descriptor names as keys and their respective values 
    """
	ret_dict = {}
	
	# Iterate over all descriptors, get their functions (func) and apply to the molecule object
	for name,func in Descriptors.descList:
		ret_dict[name] = func(mol)
	return(ret_dict)

def cdk_descriptors(mol,temp_f_smiles_name="tempsmiles.smi",temp_f_cdk_name="tempcdk.txt"):
		"""
	Get the molecular descriptors in cdk

    Parameters
    ----------
    mol : object
        An rdkit molecules object.
	temp_f_smiles_name : str
		Where to temporarily store the smiles
	temp_f_cdk_name : str
		Where to temporarily store the cdk file
		
    Returns
    -------
    dict
		Dictionary that contains the molecular descriptor names as keys and their respective values 
    """
	ret_dict = {}
	
	# Get the SMILES from the rdkit object
	smiles = Chem.MolToSmiles(mol,1)
	
	# Write SMILES to a temp file
	temp_f_smiles = open(temp_f_smiles_name,"w")
	temp_f_smiles.write("%s temp" % smiles)
	temp_f_smiles.close()
	
	# Get the features, descriptors indicate the main class of descriptors
	ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="topological"))
	ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="geometric"))
	ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="constitutional"))
	ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="electronic"))
	ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="hybrid"))
	
	# Remove the temporary files
	remove(temp_f_smiles_name)
	remove(temp_f_cdk_name)

	return(ret_dict)


def call_cdk(infile="",outfile="",descriptors=""):
	"""
	Make a system call to cdk to get descriptors

    Parameters
    ----------
    infile : str
        String indicating the filename for the input SMILES
	outfile : str
		String indicating the filename for the output values
	descriptors : str
        Main class of molecular descriptors to retrieve
		
    Returns
    -------
    dict
		Dictionary that contains the molecular descriptor names as keys and their respective values 
    """
	cmd = "java -jar CDKDescUI-1.4.6.jar -b %s -a -t %s -o %s" % (infile,descriptors,outfile)
	p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
	out = p.communicate()
	return(parse_cdk_file(outfile))

def parse_cdk_file(file):
	"""
	Parse the cdk output files

    Parameters
    ----------
    file : str
        String indicating the filename for the output of cdk
		
    Returns
    -------
    dict
		Dictionary that contains the molecular descriptor names as keys and their respective values 
    """
	cdk_file = open(file).readlines()
	cols = cdk_file[0].strip().split()[1:]
	feats = cdk_file[1].strip().split()[1:]
	return(dict(zip(cols, feats)))

def getf(mol,progs=["rdkit"]):
	"""
	Main function that makes the call to rdkit or cdk

    Parameters
    ----------
    mol : object
        An rdkit molecules object.
	progs : list
		List containing the tools to use for calculating molecular descriptors. Use "rdkit" and or "cdk"
		
    Returns
    -------
    dict
		Dictionary that contains the molecular descriptor names as keys and their respective values 
    """
	ret_dict = {}
	if "rdkit" in progs: ret_dict["rdkit"] = rdkit_descriptors(mol)
	if "cdk" in progs: ret_dict["cdk"] = cdk_descriptors(mol)
	return(ret_dict)

def parse_predret(infile_f="data-predret.csv",outfile_f="predret_parsed.csv"):
	"""
	Function to parse the predret database and retrieve the string that describes the molecule structure

    Parameters
    ----------
    infile_f : str
        Infilename
	outfile_f : str
		Outfilename
		
    Returns
    -------

    """
	infile = open(infile_f)
	outfile = open(outfile_f,"w")
	
	# Skip the first line that contains the header; incompatible with python 3
	infile.next()
	
	# Get the molecule structure
	for line in infile:
		split_line = list(csv.reader([line]))[0]
		if "InChI=" not in split_line[6]: continue
		outfile.write("%s\t%s\t%s\t%s\n" % (split_line[2],split_line[1],split_line[6],float(split_line[4])*60.0))
	outfile.close()

def parse_kohlbacher(infile_f="paper_dataset.csv",outfile_f="kohlbacher_parsed.csv"):
	"""
	Function to parse the data from the Aicheler and Kohlbacher paper; retrieve the string that describes the molecule structure

    Parameters
    ----------
    infile_f : str
        Infilename
	outfile_f : str
		Outfilename
		
    Returns
    -------

    """
	infile = open(infile_f)
	outfile = open(outfile_f,"w")
	
	# Skip the first line that contains the header; incompatible with python 3
	infile.next()

	for line in infile:
		line = line.replace("\"","").strip()
		split_line = line.split(",")
		outfile.write("%s\t%s\t%s\t%s\n" % ("kohlbacher",split_line[5],split_line[-1],split_line[4]))
	outfile.close()


def parse_mona(infile_f="monaRT.csv",outfile_f="mona_parsed.csv"):
	"""
	Function to parse the data from the MONA database; retrieve the string that describes the molecule structure

    Parameters
    ----------
    infile_f : str
        Infilename
	outfile_f : str
		Outfilename
		
    Returns
    -------

    """
	infile = open(infile_f)
	outfile = open(outfile_f,"w")
	
	for line in infile:
		line = line.strip()
		split_line = line.split("\t")
		outfile.write("%s\t%s\t%s\t%s\n" % (split_line[0],split_line[1],split_line[2],split_line[3]))
	outfile.close()

def concat_files(infiles,outfile_f="concat_train.csv"):
	"""
	Function to concatenate the input files and check for duplicates, if there are duplicates; annotate them

    Parameters
    ----------
    infiles : list
        List with all the infiles
	outfile_f : str
		Outfilename
		
    Returns
    -------

    """
	analyzed_inchi = []
	
	# Go through the files and extract the string that describes the molecule
	for fname in infiles:
		# Do not have to close due to with...
		with open(fname) as infile:
			for line in infile:
				analyzed_inchi.append(line.split("\t")[2])
	
	# Check the occurence number
	dup_inchi = [val for val,count in Counter(analyzed_inchi).items() if count > 1]

	# Write concatenated results to a new file
	# Do not have to close due to with...
	with open(outfile_f, 'w') as outfile:
		for fname in infiles:
			with open(fname) as infile:
				for line in infile:
					if line.split("\t")[2] in dup_inchi: outfile.write(line.rstrip()+"\t%s\n" % ("yes"))
					else: outfile.write(line.rstrip()+"\t%s\n" % ("no"))


def get_features(infile_name="concat_train.csv",outfile_name="retmetfeatures_new.csv",outfile_name2="retmetfeatures_new_noduplications.csv"):
	"""
	Function to get molecular descriptors (features) for all molecules, write the results to two files. 
	One containing all molecules and the other only unique molecules.

    Parameters
    ----------
    infile_name : str
        The concatenated results containing SMILES
	outfile_name : str
		Outfilename
	outfile_name2 : str
		Outfilename
	
    Returns
    -------

    """
	
	# Initialize vars and outfiles
	outfile = open(outfile_name,"w")
	outfile2 = open(outfile_name2,"w")
	features = []
	lipid_dict = {}
	write_to_f = []
	write_to_f2 = []
	
	# Open the concatenated file and iterate over every row (molecule)
	infile = open(infile_name)
	for line in infile:
		write_temp = []
		line = line.strip().split("\t")
		inchi_hash = line[2]
		ident = line[1].replace(",","_").replace(" ","_").replace("-","_")
		time = line[3]
		system = line[0]
		dup = line[4]
		
		# Try to use either an inchi or SMILES
		try: mol = Chem.inchi.MolFromInchi(inchi_hash)
		except: mol = Chem.inchi.MolFromSmiles(inchi_hash)
		if mol == None: mol = Chem.MolFromSmiles(inchi_hash)
		# Was not able to create a molecule object; skip
		if mol == None: continue
	
		# Get molecular descriptors from rdkit
		mol_feat = getf(mol)["rdkit"]
	
		# Need for column names later; if not set; set them now
		if len(features) == 0:
			features = mol_feat.keys()
	
		# Store meta info
		write_temp.append(ident)
		write_temp.append(time)
		write_temp.append(system)
		
		# Store molecular descriptors
		for f in features:
			write_temp.append(str(mol_feat[f]))
		
		# Check if we need to write to file with and without duplications
		write_to_f.append(write_temp)
		if dup == "no": write_to_f2.append(write_temp)
	
	# Write the header (column names)
	outfile.write("IDENTIFIER,time,system,%s\n" % (",".join(features)))
	outfile2.write("IDENTIFIER,time,system,%s\n" % (",".join(features)))

	# Write molecular descriptors
	for line in write_to_f:
		print line
		outfile.write("%s\n" % (",".join(line)))
	outfile.close()

	for line in write_to_f2:
		outfile2.write("%s\n" % (",".join(line)))
	outfile2.close()

if __name__ == "__main__":
	# Parse the input databases
	parse_predret()
	parse_mona()
	parse_kohlbacher()
	
	# Concatenate all the parsed files
	concat_files(["mona_parsed.csv","kohlbacher_parsed.csv","predret_parsed.csv"])

	# Calculate the molecular descriptors and write results to a file
	get_features()
