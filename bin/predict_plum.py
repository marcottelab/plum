import gc
import os
import argparse
import numpy as np
import pandas as pd
import plum.predict
import plum.util.data
import plum.models.ErrorModels
import plum.models.MarkovModels


parser = argparse.ArgumentParser("Predict interactions using a fitted PLVM model")
parser.add_argument("--datafile", required=True, help="Tidy csv with columns ID1, ID2, species, data, state")
parser.add_argument("--treefile", required=True, help="A newick treefile")
parser.add_argument("--paramfile", required=True, help="Parameter file determining model and starting params")
parser.add_argument("--outfile", required=True, help="File to write to")
parser.add_argument("--as_sorted", action='store_true', help='Flag if infile is already sorted on ID1,ID2 fields (recommended)')

args = parser.parse_args()


## Read in parameters
print("Reading parameter file")
em,mm = plum.util.data.get_models_from_file(args.paramfile)
    
predictor = plum.predict.node_probabilities(markov_model=mm, error_model=em, tree=args.treefile, data=args.datafile, outfile=args.outfile, as_sorted=args.as_sorted)

print("Predicting")
predictor.to_file()
