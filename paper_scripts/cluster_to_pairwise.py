#! /usr/bin/env python

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--edge_file",required=True,help="File with edges. CSV with header, first two columns are pairs")
parser.add_argument("--cluster_file",required=True,help="File with clusters. One per line, space delimited")
parser.add_argument("--outfile",required=True,help="File name to write to")
parser.add_argument("--edge_weight_column",default=False,help="Include this columns as an edge weight")
args = parser.parse_args()

print "Reading {}".format(args.edge_file)

edges = pd.read_csv(args.edge_file)
id1,id2 = list(edges.columns)[:2]
print id1,id2
out = open(args.outfile,'w')

if args.edge_weight_column == False:
    out.write("\t".join([id1 + "_clustid", id2 + "_clustid", id1, id2]) + "\n")
else: 
    out.write("\t".join([id1 + "_clustid", id2 + "_clustid", id1, id2, args.edge_weight_column]) + "\n")

cluster_count = 0
with open(args.cluster_file) as f:
    for line in f:
        cluster = line.strip().split()
        cluster_edges = edges[ (edges[id1].isin(cluster)) & (edges[id2].isin(cluster)) ]
        cluster_edges.loc[ :, id1 + "_clustid" ] = cluster_edges[id1].apply(lambda x: "{}_{}".format(cluster_count,x))
        cluster_edges.loc[ :, id2 + "_clustid" ] = cluster_edges[id2].apply(lambda x: "{}_{}".format(cluster_count,x))
        if args.edge_weight_column == False:
            cluster_string = cluster_edges[[id1+"_clustid",id2+"_clustid",id1,id2]].to_string(header=False,index=False)
        else:
            cluster_string = cluster_edges[[id1+"_clustid",id2+"_clustid",id1,id2,args.edge_weight_column]].to_string(header=False,index=False)
        for row in cluster_string.split("\n"):
            row = row.strip().split()
            out.write("\t".join(row) + "\n")
        cluster_count += 1
        print cluster_count
