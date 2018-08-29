#! /usr/bin/env python

#### To do: allow training data from multiple species ####

import argparse
import pandas as pd

parser = argparse.ArgumentParser("Gather averaged feature files ")
parser.add_argument("--infile", required=True, help="Prediction file, tab-separated, columns 'ID1,ID2,node,P_1,known_state'")
parser.add_argument("--outfile", required=True, help='Name of file to write to.')
parser.add_argument("--drop_diagonals", action='store_true', help="Remove rows where ID1 = ID2")
parser.add_argument("--known_positives", default=None, nargs="+", help="File(s) with known positive pairs. Space or tab separated, no header")
parser.add_argument("--known_negatives", default=None, nargs="+", help="File(s) with known negative pairs. Space or tab separated, no header")
parser.add_argument("--species_to_label", default=None, nargs="+", help="Species for each known set, in same order")
parser.add_argument("--drop_unlabeled", action='store_true', help="Flag to drop rows that are unlabeled")
parser.add_argument("--delimiter", default="\t", help="Delimiter of input known pair files")
args =  parser.parse_args()

if args.known_positives != None:
    assert len(args.species_to_label) == len(args.known_positives), "Must specify which species each training set comes from"
    posD = {}
    for ind,posfile in enumerate(args.known_positives):
        with open(posfile) as p:
            species = args.species_to_label[ind]
            pos = set( [ tuple(sorted(line.strip().split(args.delimiter))) for line in p ] )
            posD[species] = pos
            print "Found {} positive pairs for {}".format(len(pos), species)
else:
    posD = {}
        
if args.known_negatives != None:
    assert len(args.species_to_label) == len(args.known_negatives), "Must specify which species training sets come from"
    negD = {}
    for ind,negfile in enumerate(args.known_negatives):
        with open(negfile) as n:
            species = args.species_to_label[ind]
            neg = set( [ tuple(sorted(line.strip().split(args.delimiter))) for line in n ] )
            negD[species] = neg
            print "Found {} negative pairs for {}".format(len(neg), species)
else:
    negD = {}
        
out = open(args.outfile, 'w')

print "Parsing {}".format(args.infile)

with open(args.infile) as f:
    header = f.readline().strip()
    out.write(header + ",label\n")
    for line in f:
        line = line.strip().split(",")
        pair = tuple( sorted(line[:2]) )
        if args.drop_diagonals:
            if pair[0] == pair[1]:
                continue
        species = line[2]
        if species in posD:
            if pair in posD[species]:
                state = "1"
            elif pair in negD[species]:
                state = "0"
            else:
                if args.drop_unlabeled == True:
                    continue
                state = ''
        else:
            if args.drop_unlabeled == True:
                continue
            state = ''
                    
        out.write( ",".join(line + [state]) + "\n" )
        
out.close()
