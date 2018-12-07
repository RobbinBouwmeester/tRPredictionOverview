import subprocess

from random import shuffle
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

import pandas as pd

from train import train_func
from apply import apply_models

import random

adds=["_r1","_r1","_r1","_r1","_r1","_r1","_r1","_r1",
      "_r2","_r2","_r2","_r2","_r2","_r2","_r2","_r2",
      "_r3","_r3","_r3","_r3","_r3","_r3","_r3","_r3",
      "_r4","_r4","_r4","_r4","_r4","_r4","_r4","_r4",
      "_r5","_r5","_r5","_r5","_r5","_r5","_r5","_r5"]

n_all = [20,40,60,80,100,120,140,160,
         20,40,60,80,100,120,140,160,
         20,40,60,80,100,120,140,160,
         20,40,60,80,100,120,140,160,
         20,40,60,80,100,120,140,160]

def remove_low_std(X,std_val=0.01):
    rem_f = []
    std_dist = X.std(axis=0)
    rem_f.extend(list(std_dist.index[std_dist<std_val]))
    return(rem_f)

def remove_high_cor(X,upp_cor=0.98,low_cor=-0.98):
    rem_f = []
    keep_f = []

    new_m = X.corr()
    new_m = list(new_m.values)
    for i in range(len(new_m)):
        for j in range(len(new_m[i])):
            if i == j: continue
            if new_m[i][j] > upp_cor or new_m[i][j] < low_cor:
                if X.columns[j] not in keep_f:
                    rem_f.append(X.columns[j])
                    keep_f.append(X.columns[i])
    return(rem_f)

def sel_features(infile,verbose=True,remove_std=True,remove_cor=True,std_val=0.01,upp_cor=0.99,low_cor=-0.99,ignore_cols=["system","IDENTIFIER","time"]):
    if remove_std or remove_cor:
        rem_f = []

        if remove_std: rem_f.extend(remove_low_std(infile,std_val=std_val))
        if remove_cor: rem_f.extend(remove_high_cor(infile,))

        rem_f = list(set(rem_f))
        [rem_f.remove(x) for x in rem_f if x in ignore_cols]
        if verbose: print "Removing the following features: %s" % rem_f
        
        infile.drop(rem_f, axis=1, inplace=True)
    return(infile,infile.columns)

def get_sets(infile):
    sets_dict = {}

    unique_systems = list(set(infile["system"]))
    
    for us in unique_systems:
        temp_us = infile[infile["system"]==us]
        sets_dict[us] = infile[infile["system"]==us]

    return(sets_dict)

def remove_models(k,n):
    cmd = "rm -rf mods_l1/%s_*%s.pickle" % (k,n)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
    p.communicate()

def scale_cols(selected_set):
    X = selected_set.drop(["time","IDENTIFIER","system"],axis=1)
    X_scaled = pd.DataFrame(scale(X))
    X_scaled.columns = X.columns
    X_scaled.index = X.index
    for c in X_scaled.columns:
        selected_set[c] = X_scaled[c]
    return(selected_set)

def main(infilen="train/retmetfeatures.csv",scale=True):
    global adds
    global n_all
    
    outfile_time = open("time_algos.csv","w")
    outfile_time.write("Algorithm,Dataset,Time\n")
    outfile_time.close()
    
    infile = pd.read_csv(infilen)    
    try:
        keep_f = [x.strip() for x in open("features/selected_features_big.txt").readlines()]
        infile = infile[keep_f]
    except IOError:
        infile,keep_f = sel_features(infile)
        outfile = open("features/selected_features_big.txt","w")
        outfile.write("\n".join(list(keep_f)))
        outfile.close()

    sets = get_sets(infile)
    n = 42

    for k in sets.keys():
        
        selected_set = sets[k]
        select_index = range(len(selected_set.index))
    
        if len(selected_set.index) < 40: continue
        
        if scale: selected_set = scale_cols(selected_set)

        train = selected_set

        cv = KFold(len(train.index),n_folds=10)

        preds_own = train_func(train,names=[k,k,k,k,k,k,k],adds=[n,n,n,n,n,n,n,n],cv=cv)

        remove_models(k,n)


if __name__ == "__main__":
    random.seed(42)
    main()
