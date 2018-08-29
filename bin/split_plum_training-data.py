import gc
import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description='Split a gold standard set for use with PhyloPIN')
parser.add_argument("--infile", required=True, help="datafile with columns [ID1, ID2, species, data, state]")
parser.add_argument("--fold", required=True, default=5, type=int, help="How many chunks to split data into")
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
    species_string = "{}-species_".format(args.species_filter)
    reduced_outfile = species_string + infile_name
    print "Writing dataset reduced by species abundance to {}".format(reduced_outfile)
    data.to_csv(reduced_outfile, index=False)

assert "ids" not in data.columns
data["ids"] = map(tuple,data[["ID1","ID2"]].values)
knowns = data.dropna(subset=["state"])

# Add a column with both ids as (ID1,ID2) to select on and split into positive and negatives
print "Creating shuffled splits"

pos = knowns[(knowns.state == 1) | (knowns.state == 1.0)]
neg = knowns[(knowns.state == 0) | (knowns.state == 0.0)]
known_set = knowns.ids.unique()
knowns = None
gc.collect()

print "Totals:\tpositives: {}\tnegatives: {}".format(len(pos),len(neg))

# Build index-able and split-able arrays of ids
# Have to use the dtype to specify a 1-d array of tuples, rather than a 2-d array
pos_ids = np.unique(np.array( map(tuple,pos[["ID1","ID2"]].values), dtype='U25,U25' ))
neg_ids = np.unique(np.array( map(tuple,neg[["ID1","ID2"]].values), dtype='U25,U25' ))
np.random.shuffle(pos_ids)
np.random.shuffle(neg_ids)

# Split up the arrays
pos_splits = np.array_split(pos_ids, args.fold)
neg_splits = np.array_split(neg_ids, args.fold)

print "Splitting into {} datasets".format(args.fold)
data = data[data.ids.isin(known_set)]
for index in xrange(args.fold):

    # Write test data
    p = data[ data.ids.isin(pos_splits[index]) ]
    n = data[ data.ids.isin(neg_splits[index]) ]
    out = p.append(n)
    out.drop("ids", inplace=True,axis=1)
    print "Test set {}: {} rows".format(index, len(out))
    if args.species_filter > 1:
        out.to_csv( "_".join( ["test{}".format(index), species_string, infile_name] ), index=False )
    else:
        out.to_csv( "_".join( ["test{}".format(index), infile_name] ), index=False )
    
    # Create and write training data
    merged = data.merge(out, how='left', indicator=True)
    merged.drop("ids", inplace=True, axis=1)
    out = None
    gc.collect()
    tmp = merged[merged["_merge"] == "left_only"]
    out = tmp.drop("_merge", axis=1)
    print "Training set {}: {} rows".format(index, len(out))
    merged,tmp = None,None
    if args.species_filter > 1:
        out.to_csv( "_".join( ["train{}".format(index), species_string, infile_name] ), index=False )
    else:
        out.to_csv( "_".join( ["train{}".format(index), infile_name] ), index=False )
    out = None
    gc.collect()
