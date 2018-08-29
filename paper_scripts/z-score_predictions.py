#! /usr/bin/env python

import glob
import pandas as pd
import scipy.stats

# Calculate desired threshold in notebook
ZSCORETHRESH = 20.5009 # Calculated as 25% FDR in human

infiles = glob.glob("*_dataset*_07302018.csv")

corum_train = pd.read_table("/project/bjl786/PhyloPIN/training/CORUM/CORUM_subunits_human_euNOG.train_ppis.txt",header=None,names=["ID1","ID2"])
corum_test = pd.read_table("/project/bjl786/PhyloPIN/training/CORUM/CORUM_subunits_human_euNOG.test_ppis.txt",header=None,names=["ID1","ID2"])

corum_train["Hs_CORUM_train"] = "Hs_CORUM_train"
corum_test["Hs_CORUM_test"] = "Hs_CORUM_test"

yeast_train = pd.read_table("/project/bjl786/PhyloPIN/training/EMBL_yeast/EMBL_yeast_euNOG_complexes.train_ppis.txt",header=None,names=["ID1","ID2"])
yeast_test = pd.read_table("/project/bjl786/PhyloPIN/training/EMBL_yeast/EMBL_yeast_euNOG_complexes.test_ppis.txt",header=None,names=["ID1","ID2"])

yeast_train["Sc_EMBL_train"] = "Sc_EMBL_train"
yeast_test["Sc_EMBL_test"] = "Sc_EMBL_test"

labeled_pairs = reduce( lambda l,r: pd.merge(l,r,on=["ID1","ID2"],how='outer'), [corum_train,corum_test,yeast_train,yeast_test] )
labeled_pairs["TrainTestPairs"] = labeled_pairs["Hs_CORUM_train"].copy()
labeled_pairs["TrainTestPairs"] = labeled_pairs["TrainTestPairs"].fillna(labeled_pairs["Sc_EMBL_train"])
labeled_pairs["TrainTestPairs"] = labeled_pairs["TrainTestPairs"].fillna(labeled_pairs["Hs_CORUM_test"])
labeled_pairs["TrainTestPairs"] = labeled_pairs["TrainTestPairs"].fillna(labeled_pairs["Sc_EMBL_test"])
labeled_pairs = labeled_pairs.drop(["Hs_CORUM_train","Hs_CORUM_test","Sc_EMBL_train","Sc_EMBL_test"],axis=1)

for infile in infiles:
    print "Reading {}".format(infile)
    df = pd.read_csv(infile).sort_values("P_1",ascending=False)
    df["P1_zscore"] = scipy.stats.zscore(df.P_1)
    out = df[df["P1_zscore"] >= ZSCORETHRESH]
    out = out.merge(labeled_pairs,on=["ID1","ID2"],how='left')
    out.to_csv( infile.split(".")[0] + "_zscore.csv", index=False )
