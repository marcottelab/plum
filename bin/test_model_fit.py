#! /usr/bin/env python

import copy
import argparse
import plum.util.data
import plum.training.recall_precision
import plum.models.ErrorModels, plum.models.MarkovModels

from collections import defaultdict

parser = argparse.ArgumentParser("Get precision-recall metrics of a fully specified plum Model")
parser.add_argument("--param_files", required=True, nargs="+", help="Parameter files with plum model specs")
parser.add_argument("--data_files", required=True, nargs="+", help="Data files to test against")
parser.add_argument("--tree_file", required=True, help="A nexus formatted phylogeny with node labels")
parser.add_argument("--out_file", required=True, help="File to write summary to")

args = parser.parse_args()

models = defaultdict(dict)
for i in args.param_files:
    em,mm = plum.util.data.get_models_from_file(i)
    mname = str(em).split(".")[-1]
    fname = i.split(".")[0]
    models[mname][fname] = (em,mm)

outfile = open(args.out_file, 'w')
outfile.write( ",".join( ["datafile", "model type", "paramfile", "APS"]  ) + "\n" )
for mname in models:
    is_first = True
    print "Model: {}".format(mname)
    for d in args.data_files:
        datafile = ".".join( d.split("/")[-1].split(".")[:-1] )
        print "Datafile: {}".format(datafile)
        for fname,(em,mm) in models[mname].iteritems():
            if is_first:
                model = plum.training.recall_precision.plumRecallPrecision(error_model=em,markov_model=mm,tree=args.tree_file,data=d)
                is_first = False
            else:
                paramD = copy.deepcopy( em.freeParamDict )
                print "Updating model from {}".format(fname)
                paramD.update( copy.deepcopy( mm.freeParamDict ) )
                model.update_from_dict(paramD)
            
            outfile.write( ",".join([datafile, mname, fname, str(model.aps)]) + "\n" )