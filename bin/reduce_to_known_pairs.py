import gc
import argparse
import numpy as np
import pandas as pd

pd.set_option('mode.chained_assignment', None)

# First run Kevin's split_complexes.py to get a test and training that are non-redundant 
# and non-overlapping. Then, use makeTrainingData.py to create a csv with the columns
# "ID1" "ID2" "species" "state", where `state` is {0,1}. This file can then be joined into
# CF/MS data. This script picks up at that point to create cross-validation sets


parser = argparse.ArgumentParser(description='Split a gold standard set for use with PhyloPIN')
parser.add_argument("--infile", required=True, help="datafile with columns [ID1, ID2, species, data, state]")
parser.add_argument("--outfile", required=True, help="File name to write to")
parser.add_argument("--species_filter", default=1, type=int, help="Delete pairs present in fewer than this many species. Writes a csv file of these groups")
args = parser.parse_args()


if "/" in args.infile: # is a path
    infile_name = args.infile.split("/")[-1]
else:
    infile_name = args.infile
    
    
# Read in data and drop down to knowns
print "Reading {}".format(args.infile)
data = pd.read_csv(args.infile)


if args.species_filter > 1:
    print "Removing groups found in fewer than {} species".format(args.species_filter)
    data = data.groupby(["ID1","ID2"]).filter( lambda x: len(x) > args.species_filter )

assert "ids" not in data.columns
data["ids"] = map(tuple,data[["ID1","ID2"]].values)
knowns = data.dropna(subset=["state"])

print "Reducing to known set"
outdata = data[ data.ids.isin(knowns.ids) ]
outdata.sort_values("ids", inplace=True)
outdata.drop("ids", inplace=True, axis=1)

outdata.to_csv(args.outfile,index=False)