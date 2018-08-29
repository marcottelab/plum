import copy
import dendropy
import operator
import itertools
import scipy.stats
import numpy as np
import pandas as pd
from plum.util.data import plumData
from plum.util.cpruning import getNodeProbsAndTransitions

class node_probabilities(plumData):
    '''
    Infer node probabilities for an interaction given a Markov and error model.
    
    -`markov_model`: A plum.models.MarkovModels object. Its parameters will be used for prediction
    -`error_model`: A plum.models.ErrorModels object. Its parameters will be used for prediction
    -`tree`: Path to the input newick tree
    -`data`: A comma-separated file in tidy data format.
    -`outfile`: File to predictions to
    '''
    
    def __init__(self,markov_model,error_model,tree,data,outfile=None,as_sorted=True):
        self.outfile = outfile
        self._data = data
        self._tree = dendropy.Tree.get_from_path(tree,'newick',preserve_underscores=True,suppress_internal_node_taxa=False,rooting='default-rooted')
        self._em = error_model
        self._mm = markov_model
        self._state_priors = dict(zip(*self._mm.stationaryFrequencies))  
        assert type(as_sorted) == bool, "`as_sorted` must be True or False"
        self.as_sorted = as_sorted
        
    def _safe_float_cast(self,value):
        '''Cast to np.float64 with np.nan as default if input can't be so cast'''
        try:
            return np.float64(value)
        except ValueError, TypeError:
            if type(value) is list:
                a = np.empty( (len(value),) )
                a.fill(np.nan)
            else:
                return np.nan
                
    def _safe_str_cast(self,value):
        if value is np.nan:
            return ''
        elif value == None:
            return ''
        else:
            return str(value)
    
    def to_iter(self):
        '''Run predictions on loaded data. Generator over (i,j) pair, node_string, P(1)'''
        get_known = lambda x: np.nan if x == '' else int(float(x))
        if self.as_sorted == False:
            plumData.__init__(self,markov_model=self._mm,error_model=self._em,tree=self._tree,data=self._data)
            for dataD, knownD, pair in itertools.izip(self._featureDs, self._knownknownDs, self._pairs):
                nodeprobs,transitionprobs = getNodeProbsAndTransitions(tree=self._tree,
                                        dataD=dataD,
                                        knownD=knownD,
                                        error_model=self._error_model,
                                        markov_model=self._markov_model,
                                        priors=self._state_priors)
                for node, prob in nodeprobs.iteritems():
                    yield pair, node, prob[1]
        else:
            with open(self._data) as f:
                header = f.readline().strip().split(",")
                is_first = True
                dataD,knownD = {},{}
                for line in f:
                    line = line.strip().split(",")
                    ids = (line[0],line[1])
                    if is_first == True: # first line
                        is_first = False
                        current_pair = ids
                    
                    if ids != current_pair:
                        nodeprobs,transitionprobs = getNodeProbsAndTransitions(tree=self._tree,
                                                 dataD=dataD,
                                                 knownD=knownD,
                                                 error_model=self._em,
                                                 markov_model=self._mm,
                                                 priors=self._state_priors)
                        for node,prob in nodeprobs.iteritems():
                            if node in transitionprobs:
                                tprob = transitionprobs[node]
                            else:
                                tprob = 0.
                            yield current_pair, node, prob[1], tprob, dataD.get(node,None), knownD.get(node,None)
                        current_pair = ids
                        dataD,knownD = {},{}
                    
                    species = line[2]
                    if self._em.is_multivariate:
                        dataD[species] = self._safe_float_cast(line[3:-1])
                        knownD[species] = get_known(line[-1])  
                    else:
                        dataD[species] = self._safe_float_cast(line[3])
                        knownD[species] = get_known(line[4])
                
                nodeprobs,transitionprobs = getNodeProbsAndTransitions(tree=self._tree,
                                        dataD=dataD,
                                        knownD=knownD,
                                        error_model=self._em,
                                        markov_model=self._mm,
                                        priors=self._state_priors)
                for node,prob in nodeprobs.iteritems():
                    if node in transitionprobs:
                        tprob = transitionprobs[node]
                    else:
                        tprob = 0.
                    yield current_pair, node, prob[1], tprob, dataD.get(node,None), knownD.get(node,None)
            
    def to_file(self):
        '''Write predictions to outfile'''
        assert self.outfile is not None, "Must specify outfile"
        with open(self.outfile, 'w') as out:
            out.write(",".join(["ID1","ID2","node","P_1", "P_event", "known_state"]) + "\n")
            count = 0
            for pair, node, prob, tprob, data, state in self.to_iter():
                pair_string = ",".join(map(str,pair))
                out.write(",".join([ pair_string, node, str(prob), str(tprob), self._safe_str_cast(state) ]) + "\n")
                count += 1
                if count % 1000 == 0:
                    print count
        
