from rdkit import Chem
from getf import getf

def get_features(infile_name="data/LMSDFDownload28Jun15FinalAll.sdf",outfile_name="lmfeatures.csv",id_index=0,mol_index=1,time_index=None):
	outfile = open(outfile_name,"w")

	features = []
	lipid_dict = {}
	time_dict = {}
	counter = 0

	mols = open(infile_name)

	
	for mol in mols:
		counter += 1

		if (counter % 100) == 0: print counter

		if "\t" in mol: mol = mol.strip().split("\t")
		else: mol = mol.strip().split(",")

		identifier = mol[id_index]
		mol_str = mol[mol_index]
		if mol_str == "SMILES": continue
		if time_index: 
			rt = mol[time_index]
			time_dict[identifier] = rt

		m = Chem.MolFromSmiles(mol_str)

		if m == None: 
			#TODO write error msg
			continue

		try: fdict = getf(m)
		except: continue

		lipid_dict[identifier] = fdict["rdkit"]
		
		features.extend(lipid_dict[identifier].keys())
	
	features = list(set(features))
	


	if time_index: outfile.write("IDENTIFIER,time,%s\n" % (",".join(features)))
	else: outfile.write("IDENTIFIER,%s\n" % (",".join(features)))
	for identifier in lipid_dict.keys():
		
		if time_index:
			outfile.write("%s," % (identifier))
			outfile.write("%s" % (time_dict[identifier]))

		else: outfile.write("%s" % (identifier))
		for f in features:
			outfile.write(",%s" % (lipid_dict[identifier][f]))
		outfile.write("\n")
	outfile.close()

if __name__ == "__main__":
	get_features()