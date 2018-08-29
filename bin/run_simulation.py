#! /usr/bin/env python

import argparse
import dendropy
import plum.sim
import plum.util.data
import plum.models.ErrorModels
import plum.models.MarkovModels

parser = argparse.ArgumentParser(description="Collect parameters for plum simulation")
parser.add_argument("--tree",required=True,help="Newick tree file. Nodes must be labeled")
parser.add_argument("--param_file",required=True,help="Parameter file specifying model to simulate under")

parser.add_argument("--node_outfile",required=True,help="Name of file to write all data to")
parser.add_argument("--tip_outfile",default=None,help="Name of file to write just tip data to")
parser.add_argument("--nchars",type=int,default=1000,help="Number of characters to simulate")
parser.add_argument("--as_pickle",action='store_true',help="If flagged, pickle output")

args = parser.parse_args()

## Read in and create models
print "Reading parameter file"
em,mm = plum.util.data.get_models_from_file(args.param_file)
    
tree = dendropy.Tree.get_from_path(args.tree,'newick',suppress_internal_node_taxa=False,
                                        rooting='default-rooted',preserve_underscores=True)
                                        
sim = plum.sim.TwoStateSim(tree=tree,markov_model=mm,error_model=em,nchars=args.nchars)
if args.tip_outfile == None:
    sim(node_outfile=args.node_outfile,as_pickle=args.as_pickle)
else:
    sim(node_outfile=args.node_outfile,tip_outfile=args.tip_outfile,as_pickle=args.as_pickle,tip_data=True)