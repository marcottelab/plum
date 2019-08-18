# cython: profile=True
# cython: linetrace=True
# cython: boundscheck=False, wraparound=False

import copy
import operator
import dendropy
import numpy as np
cimport numpy as np
cimport cython
import pandas as pd
from functools import reduce
from libc.math cimport log, abs, isnan
from collections import defaultdict


ctypedef np.float_t DTYPE_t

cdef int other_state(int state) except -1:
    '''Return the other state'''
    if state == 1:
        return 0
    elif state == 0:
        return 1
    else:
        raise Exception("") # will return -1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=1] _ctreeL(object tree, dict[:] data, dict[:] known, object error_model, object markov_model, dict priors):

    # Defs for postorder trace
    cdef Py_ssize_t s, x, Ni, Di
    
    cdef Py_ssize_t N = len(data)
    cdef Py_ssize_t nNodes = tree.n_nodes
    cdef double [:] logLs = np.empty(N)
    cdef object [:] postnodes = tree.postorder_nodes
    cdef dict dataD, knownD
    
    cdef object n,c # ctree node object
    cdef int[2] states = [0,1]
    cdef double child_condL, treeL
    cdef str taxon, ctaxon, parent
    cdef double ntau, ctau # branch lengths
    cdef tuple pij # tuple of (tau,state1,state2)
    cdef double [:,:] nodeLs = np.zeros(shape = (2,nNodes))
    cdef dict Pijs = {}
    cdef list condL

    for Di in xrange(N):
        dataD = data[Di]
        knownD = known[Di]
        #nodeLs = defaultdict(dict)
        nodeLs = np.zeros(shape = (2,nNodes))
        ## Calculate "down" variables (conditional likelihoods)
        for Ni in xrange(nNodes): 
            n = postnodes[Ni] # Ni indexes nodes in post-order
            taxon = n.name          
            if n.is_leaf: # initialize cLs at leaves from error model or missing or known data
                ntau = n.edge_length
                if taxon in knownD:
                    if isnan(knownD[taxon]): # unknown
                        pass
                    else: # state is known
                        s = knownD[taxon] 
                        nodeLs[s][Ni] = 1.
                        nodeLs[other_state(s)][Ni] = 0.
                        continue
    
                if taxon in dataD:
                    datum = dataD[taxon]
                    if np.all( pd.isnull(datum) ): # missing data, unknown state
                        for s in states:
                            nodeLs[s][Ni] = 1.
                    else:
                        for s in states: # get Ls from error model
                            nodeLs[s][Ni] = error_model.P(data=datum,state=s)
    
                            # While we're already looping over states,
                            # start populating Pijs
                            for x in states:
                                pij = (ntau,s,x)
                                if pij not in Pijs:
                                    Pijs[pij] = markov_model.P(*pij)
    
                else: # missing data
                    for s in states:
                        nodeLs[s][Ni] = 1.
    
            else: # is interior node, get cLs from child nodes
                for s in states:
                    condL = []
                    for c in n.children:
                        ctau = c.edge_length # get branch length
                        ctaxon = c.name
                        child_condL = 0.
                        for x in states:
                            pij = (ctau,s,x)
                            try:
                                mmP = Pijs[pij] # cached Markov probability
                            except KeyError:
                                Pijs[pij] = markov_model.P(*pij)
                                mmP = Pijs[pij]
                            child_condL += mmP * nodeLs[x][c.postindex]
                        condL.append(child_condL)
                    nodeLs[s][Ni] = reduce(operator.mul, condL)
            
            if n == tree.root: # termination after root cond Ls
                treeL = 0.
                for s in states:
                    treeL += priors[s] * nodeLs[s][Ni]
                if treeL == 0.:
                    return 0.
                logLs[Di] = log(treeL)
                
    return np.array(logLs)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[object, ndim=1] _cgetNodeProbs(object tree, dict[:] data, dict[:] known, object error_model, object markov_model, dict priors):

    # Defs for postorder trace
    cdef Py_ssize_t s, x, j, k, Ni, Di, condL_index
    
    cdef Py_ssize_t N = len(data)
    cdef Py_ssize_t nNodes = tree.n_nodes
    #cdef double [:, :, :] nodeProbs_vec = np.empty( shape = (N,2,nNodes) ) # stores node probabilities for each data point
    cdef object [:] nodeProbs_vec = np.empty(N, dtype=object)
    cdef object [:] postnodes = tree.postorder_nodes
    cdef object [:] prenodes = tree.preorder_nodes
    cdef dict dataD, knownD
    
    cdef object n,c # ctree node object
    cdef object datum # can be float or null
    cdef int[2] states = [0,1]
    cdef double child_condL, treeL
    cdef str taxon
    #cdef double ntau, ctau # branch lengths
    cdef tuple pij # tuple of (tau,state1,state2)
    cdef double [:,:] nodeLs = np.zeros(shape = (2,nNodes))
    cdef dict Pijs = {} # cache of pij values so it doesn't have to calculated multiple times.
    cdef double condL[2]
    condL[:] = [0,0] # initialize
    
    # Defs for preorder trace
    cdef double up_s, sister_term
    cdef double [:,:] nodeUps = np.zeros(shape = (2,nNodes))
    cdef object sister, parent # ctree.Nodes
    cdef double parent_term, parent_pij, sister_pij

    # Defs for final probability calculations
    #cdef double [:,:] nodeProbs = np.zeros(shape = (2,nNodes))
    cdef dict nodeProbs = {}
    cdef double [:,:] numerators = np.zeros(shape = (2,nNodes))
    cdef double denominator, numerator
    
    
    for Di in xrange(N): # indexes the vector of data dictionaries
        dataD = data[Di]
        knownD = known[Di]
        #nodeLs = defaultdict(dict)
        nodeLs = np.zeros(shape = (2,N))
        ## Calculate "down" variables (conditional likelihoods)
        for Ni in xrange(nNodes): 
            n = postnodes[Ni] # Ni indexes nodes in post-order
            taxon = n.name          
            if n.is_leaf: # initialize cLs at leaves from error model or missing or known data
                #ntau = n.edge_length
                if taxon in knownD:
                    if isnan(knownD[taxon]): # unknown
                        pass
                    else: # state is known
                        s = knownD[taxon] 
                        nodeLs[s][Ni] = 1.
                        nodeLs[other_state(s)][Ni] = 0.
                        continue
    
                if taxon in dataD:
                    datum = dataD[taxon]
                    if np.all( pd.isnull(datum) ): # missing data, unknown state
                        for s in states:
                            nodeLs[s][Ni] = 1.
                    else:
                        for s in states: # get Ls from error model
                            nodeLs[s][Ni] = error_model.P(data=datum,state=s)
    
                            # While we're already looping over states,
                            # start populating Pijs
                            for x in states:
                                pij = (n.edge_length,s,x)
                                if pij not in Pijs:
                                    Pijs[pij] = markov_model.P(*pij)
    
                else: # missing data
                    for s in states:
                        nodeLs[s][Ni] = 1.
    
            else: # is interior node, get cLs from child nodes
                for s in states:
                    condL_index = 0
                    for c in n.children:
                        #ctau = c.edge_length # get branch length
                        child_condL = 0.
                        for x in states:
                            pij = (c.edge_length ,s,x)
                            try:
                                mmP = Pijs[pij] # cached Markov probability
                            except KeyError:
                                Pijs[pij] = markov_model.P(*pij)
                                mmP = Pijs[pij]
                            child_condL += mmP * nodeLs[x][c.postindex]
                        condL[condL_index] = child_condL
                        condL_index += 1
                        
                    #nodeLs[s][Ni] = reduce(operator.mul, condL)
                    nodeLs[s][Ni] = condL[0] * condL[1]
        
        ## Calculate "up" variables
        #nodeProbs = np.zeros(shape = (2,nNodes))
        nodeProbs = {}
        for Ni in xrange(nNodes):
            n = prenodes[Ni]
            denominator = 0.
            if n == tree.root: # is root
                for s in states:
                    nodeUps[s][n.postindex] = priors[s]
                    numerator = priors[s] * nodeLs[s][n.postindex]
                    denominator += numerator
                    numerators[s][n.postindex] = numerator
            else:
               # ntau = n.edge_length
                for s in states: # is interior node or tip
                    up_s = 0.
                    for j in states: #outer sum over j
                        parent_pij = Pijs[(n.edge_length,j,s)]
                        parent_term = nodeUps[j][n.parent.postindex] * parent_pij
    
                        # inner sum over k
                        sister_term = 0.
                        for k in states:
                            sister_pij = Pijs[(n.edge_length,j,k)]
                            sister_term += sister_pij * nodeLs[k][n.sister.postindex]
                        up_s += parent_term * sister_term
                    nodeUps[s][n.postindex] = up_s
                    numerator = up_s * nodeLs[s][n.postindex]
                    denominator += numerator
                    numerators[s][n.postindex] = numerator
            #for s in states:
            #    if denominator != 0.:
            #        nodeProbs[s][n.postindex] = numerators[s][n.postindex]/denominator
            #    else:
            #        nodeProbs[s][n.postindex] = .5
            if denominator != 0:
                nodeProbs[n.name] = numerators[1][n.postindex]/denominator
            else:
                nodeProbs[n.name] = .5
        nodeProbs_vec[Di] = nodeProbs
    return np.array( nodeProbs_vec )
    
@cython.boundscheck(False)
@cython.wraparound(False)
def cgetNodeProbsAndTransitions(object tree, dict[:] data, dict[:] known, object error_model, object markov_model, dict priors):
    '''

    ## Parameters ##
        `tree`: a dendropy Tree object with interior taxa labeled.
        `dataD`: dictionary mapping tip node labels to continuous data
        `knownD`: dictionary mapping tip node labels to known states, or None
        `error_model`: an ErrorModels instance
        `markov_model`: a MarkovModels instance
        `priors`: dictionary mapping each state to prior. From the stationary frequencies in the MarkovModel
    `'''

    # Defs for postorder trace
    cdef Py_ssize_t s, x, j, k, Ni, Di, condL_index
    
    cdef Py_ssize_t N = len(data)
    cdef Py_ssize_t nNodes = tree.n_nodes
    #cdef double [:, :, :] nodeProbs_vec = np.empty( shape = (N,2,nNodes) ) # stores node probabilities for each data point
    cdef object [:] nodeProbs_vec = np.empty(N, dtype=object)
    cdef object [:] transitionProbs_vec = np.empty(N, dtype=object)
    cdef object [:] postnodes = tree.postorder_nodes
    cdef object [:] prenodes = tree.preorder_nodes
    cdef dict dataD, knownD
    
    cdef object n,c # ctree node object
    cdef object datum # can be float or null
    cdef int[2] states = [0,1]
    cdef double child_condL, treeL
    cdef str taxon
    #cdef double ntau, ctau # branch lengths
    cdef tuple pij # tuple of (tau,state1,state2)
    cdef double [:,:] nodeLs = np.zeros(shape = (2,nNodes))
    cdef dict Pijs = {} # cache of pij values so it doesn't have to calculated multiple times.
    cdef double condL[2]
    condL[:] = [0,0] # initialize
    
    # Defs for preorder trace
    cdef double up_s, sister_term
    cdef double [:,:] nodeUps = np.zeros(shape = (2,nNodes))
    cdef object sister, parent # ctree.Nodes
    cdef double parent_term, parent_pij, sister_pij

    # Defs for final probability calculations
    #cdef double [:,:] nodeProbs = np.zeros(shape = (2,nNodes))
    cdef dict nodeProbs = {}
    cdef dict transitionProbs = {}
    cdef double [:,:] numerators = np.zeros(shape = (2,nNodes))
    cdef double denominator, numerator
    
    
    for Di in xrange(N): # indexes the vector of data dictionaries
        dataD = data[Di]
        knownD = known[Di]
        #nodeLs = defaultdict(dict)
        nodeLs = np.zeros(shape = (2,N))
        ## Calculate "down" variables (conditional likelihoods)
        for Ni in xrange(nNodes): 
            n = postnodes[Ni] # Ni indexes nodes in post-order
            taxon = n.name          
            if n.is_leaf: # initialize cLs at leaves from error model or missing or known data
                #ntau = n.edge_length
                if taxon in knownD:
                    if isnan(knownD[taxon]): # unknown
                        pass
                    else: # state is known
                        s = knownD[taxon] 
                        nodeLs[s][Ni] = 1.
                        nodeLs[other_state(s)][Ni] = 0.
                        continue
    
                if taxon in dataD:
                    datum = dataD[taxon]
                    if np.all( pd.isnull(datum) ): # missing data, unknown state
                        for s in states:
                            nodeLs[s][Ni] = 1.
                    else:
                        for s in states: # get Ls from error model
                            nodeLs[s][Ni] = error_model.P(data=datum,state=s)
    
                            # While we're already looping over states,
                            # start populating Pijs
                            for x in states:
                                pij = (n.edge_length,s,x)
                                if pij not in Pijs:
                                    Pijs[pij] = markov_model.P(*pij)
    
                else: # missing data
                    for s in states:
                        nodeLs[s][Ni] = 1.
    
            else: # is interior node, get cLs from child nodes
                for s in states:
                    condL_index = 0
                    for c in n.children:
                        #ctau = c.edge_length # get branch length
                        child_condL = 0.
                        for x in states:
                            pij = (c.edge_length ,s,x)
                            try:
                                mmP = Pijs[pij] # cached Markov probability
                            except KeyError:
                                Pijs[pij] = markov_model.P(*pij)
                                mmP = Pijs[pij]
                            child_condL += mmP * nodeLs[x][c.postindex]
                        condL[condL_index] = child_condL
                        condL_index += 1
                        
                    #nodeLs[s][Ni] = reduce(operator.mul, condL)
                    nodeLs[s][Ni] = condL[0] * condL[1]
        
        ## Calculate "up" variables
        #nodeProbs = np.zeros(shape = (2,nNodes))
        nodeProbs = {}
        for Ni in xrange(nNodes):
            n = prenodes[Ni]
            denominator = 0.
            if n == tree.root: # is root
                for s in states:
                    nodeUps[s][n.postindex] = priors[s]
                    numerator = priors[s] * nodeLs[s][n.postindex]
                    denominator += numerator
                    numerators[s][n.postindex] = numerator
            else:
               # ntau = n.edge_length
                for s in states: # is interior node or tip
                    up_s = 0.
                    for j in states: #outer sum over j
                        parent_pij = Pijs[(n.edge_length,j,s)]
                        parent_term = nodeUps[j][n.parent.postindex] * parent_pij
    
                        # inner sum over k
                        sister_term = 0.
                        for k in states:
                            sister_pij = Pijs[(n.edge_length,j,k)]
                            sister_term += sister_pij * nodeLs[k][n.sister.postindex]
                        up_s += parent_term * sister_term
                    nodeUps[s][n.postindex] = up_s
                    numerator = up_s * nodeLs[s][n.postindex]
                    denominator += numerator
                    numerators[s][n.postindex] = numerator
            #for s in states:
            #    if denominator != 0.:
            #        nodeProbs[s][n.postindex] = numerators[s][n.postindex]/denominator
            #    else:
            #        nodeProbs[s][n.postindex] = .5
            if denominator != 0:
                nodeProbs[n.name] = numerators[1][n.postindex]/denominator
            else:
                nodeProbs[n.name] = .5
            if n != tree.root:
                transitionProbs[n.name] = abs( nodeProbs[n.name] - nodeProbs[n.parent.name] )
        nodeProbs_vec[Di] = nodeProbs
        transitionProbs_vec[Di] = transitionProbs
    return np.array( nodeProbs_vec ), np.array( transitionProbs_vec )
    
def getNodeProbsAndTransitions(tree,dataD,knownD,error_model,markov_model,priors):
    '''Felsenstein's pruning algorithm with error models at the tips.

    Returns ancestral state reconstructions of the two states for the tips
    and interior nodes of the given tree, as well as posterior probability of
    a transition at each node

    ## Parameters ##
        `tree`: a dendropy Tree object with interior taxa labeled.
        `dataD`: dictionary mapping tip node labels to continuous data
        `knownD`: dictionary mapping tip node labels to known states, or None
        `error_model`: an ErrorModels instance
        `markov_model`: a MarkovModels instance
        `priors`: dictionary mapping each state to prior. From the stationary frequencies in the MarkovModel
    `'''

    # Defs for postorder trace
    cdef Py_ssize_t s, x, j, k, o
    cdef object n # dendropy node object
    cdef int[2] states = [0,1]
    cdef double child_condL
    cdef str taxon, ctaxon, parent
    cdef double ntau, ctau # branch lengths
    cdef tuple pij # tuple of (tau,state1,state2)
    cdef object nodeLs = defaultdict(dict)
    cdef dict Pijs = {}
    cdef list condL


    # Defs for preorder trace
    cdef double up_s, sister_term
    cdef object nodeUps = defaultdict(dict)
    cdef str sister
    cdef double parent_term, parent_pij, atau

    # Defs for final probability calculations
    cdef object nodeProbs = defaultdict(dict)
    cdef object transitionProbs = {}
    cdef object numerators = defaultdict(dict)
    cdef double denominator, numerator

    ## Calculate "down" variables (conditional likelihoods)
    for n in tree.postorder_node_iter():
        taxon = n.taxon.label
        if n != tree.seed_node: # if root edge_length is None
            ntau = n.edge_length
        if n.is_leaf(): # initialize cLs at leaves from error model or missing or known data

            if taxon in knownD:
                if pd.isnull(knownD[taxon]): # unknown
                    pass
                else: # state is known
                    nodeLs[taxon] = {knownD[taxon]:1.,other_state(knownD[taxon]):0.}
                    continue

            if taxon in dataD:
                datum = dataD[taxon]
                if np.all( pd.isnull(datum) ): # missing data, unknown state
                    nodeLs[taxon] = {states[0]:1.,states[1]:1.}
                else:
                    for s in states: # get Ls from error model
                        nodeLs[taxon][s] = error_model.P(data=datum,state=s)

                        # While we're already looping over states,
                        # start populating Pijs
                        for x in states:
                            pij = (ntau,s,x)
                            if pij not in Pijs:
                                Pijs[pij] = markov_model.P(*pij)

            else: # missing data
                nodeLs[taxon] = {states[0]:1.,states[1]:1.}

        else: # is interior node, get cLs from child nodes
            for s in states:
                condL = []
                for c in n.child_node_iter():
                    ctau = c.edge_length # get branch length
                    ctaxon = c.taxon.label
                    child_condL = 0.
                    for x in states:
                        pij = (ctau,s,x)
                        try:
                            mmP = Pijs[pij] # cached Markov probability
                        except KeyError:
                            Pijs[pij] = markov_model.P(*pij)
                            mmP = Pijs[pij]
                        child_condL += mmP * nodeLs[ctaxon][x]
                    condL.append(child_condL)
                nodeLs[taxon][s] = reduce(operator.mul, condL)

    ## Calculate "up" variables
    for n in tree.preorder_node_iter():
        taxon = n.taxon.label
        denominator = 0.
        if n == tree.seed_node: # is root
            for s in states:
                nodeUps[taxon][s] = priors[s]
                numerator = priors[s] * nodeLs[taxon][s]
                denominator += numerator
                numerators[taxon][s] = numerator
        else:
            ntau = n.edge_length
            parent = n.parent_node.taxon.label
            sister = n.sister_nodes()[0].taxon.label
            for s in states: # is interior node or tip
                up_s = 0.
                for j in states: #outer sum over j
                    parent_pij = Pijs[(ntau,j,s)]
                    parent_term = nodeUps[parent][j] * parent_pij

                    # inner sum over k
                    sister_term = 0.
                    for k in states:
                        sister_pij = Pijs[(ntau,j,k)]
                        sister_term += sister_pij * nodeLs[sister][k]
                    up_s += parent_term * sister_term
                nodeUps[taxon][s] = up_s
                numerator = up_s * nodeLs[taxon][s]
                denominator += numerator
                numerators[taxon][s] = numerator
                
        for s in states:
            if denominator != 0.:
                nodeProbs[taxon][s] = numerators[taxon][s]/denominator
            else:
                nodeProbs[taxon][s] = .5
        if n != tree.seed_node:
            transitionProbs[taxon] = abs( nodeProbs[taxon][s] - nodeProbs[parent][s] )
    return nodeProbs, transitionProbs   
   
@cython.boundscheck(False)
@cython.wraparound(False)
def cgetNodeProbs(object tree, dict[:] data, dict[:] known, object error_model, object markov_model, dict priors):
    '''Return a vector of inferred ancestral state reconstructions (probabilities of interaction) for each pair
    of proteins given a tree, a transition model, and an error model
    
        ## Parameters ##
        `tree`: a dendropy Tree object with interior taxa labeled.
        `dataD`: dictionary mapping tip node labels to continuous data
        `knownD`: dictionary mapping tip node labels to known states, or None
        `error_model`: an ErrorModels instance
        `markov_model`: a MarkovModels instance
        `priors`: dictionary mapping each state to prior. From the stationary frequencies in the MarkovModel'''
    return _cgetNodeProbs(tree, data, known, error_model, markov_model, priors)

@cython.boundscheck(False)
@cython.wraparound(False)
def ctreeL(object tree, dict[:] data, dict[:] known, object error_model, object markov_model, dict priors):
    '''Return vector of likelihoods for each pair of proteins given a tree, a transition model, and an error model
    
        ## Parameters ##
        `tree`: a dendropy Tree object with interior taxa labeled.
        `dataD`: dictionary mapping tip node labels to continuous data
        `knownD`: dictionary mapping tip node labels to known states, or None
        `error_model`: an ErrorModels instance
        `markov_model`: a MarkovModels instance
        `priors`: dictionary mapping each state to prior. From the stationary frequencies in the MarkovModel'''
    return _ctreeL(tree,data,known,error_model,markov_model,priors)
                
                
### Non-vectorized functions - deprecated         
         
#def treeL(tree,dataD,knownD,error_model,markov_model,priors):
#    '''Felsenstein's pruning algorithm with error models at the tips.
#
#    Returns the likelihood of the tree given the data and model
#
#    ## Parameters ##
#        `tree`: a dendropy Tree object with interior taxa labeled.
#        `dataD`: dictionary mapping tip node labels to continuous data
#        `knownD`: dictionary mapping tip node labels to known states, or None
#        `error_model`: an ErrorModels instance
#        `markov_model`: a MarkovModels instance
#        `priors`: dictionary mapping each state to prior. From the stationary frequencies in the MarkovModel
#    `'''
#
#    # Defs for postorder trace
#    cdef Py_ssize_t s, x
#    cdef object n # dendropy node object
#    cdef int[2] states = [0,1]
#    cdef double child_condL, treeL
#    cdef str taxon, ctaxon, parent
#    cdef double ntau, ctau # branch lengths
#    cdef tuple pij # tuple of (tau,state1,state2)
#    cdef object nodeLs = defaultdict(dict)
#    cdef dict Pijs = {}
#    cdef list condL
#
#    ## Calculate "down" variables (conditional likelihoods)
#    for n in tree.postorder_node_iter():
#        taxon = n.taxon.label            
#        if n.is_leaf(): # initialize cLs at leaves from error model or missing or known data
#            ntau = n.edge_length
#            if taxon in knownD:
#                if pd.isnull(knownD[taxon]): # unknown
#                    pass
#                else: # state is known
#                    nodeLs[taxon] = {knownD[taxon]:1.,other_state(knownD[taxon]):0.}
#                    continue
#
#            if taxon in dataD:
#                datum = dataD[taxon]
#                if np.all( pd.isnull(datum) ): # missing data, unknown state
#                    nodeLs[taxon] = {states[0]:1.,states[1]:1.}
#                else:
#                    for s in states: # get Ls from error model
#                        nodeLs[taxon][s] = error_model.P(data=datum,state=s)
#
#                        # While we're already looping over states,
#                        # start populating Pijs
#                        for x in states:
#                            pij = (ntau,s,x)
#                            if pij not in Pijs:
#                                Pijs[pij] = markov_model.P(*pij)
#
#            else: # missing data
#                nodeLs[taxon] = {states[0]:1.,states[1]:1.}
#
#        else: # is interior node, get cLs from child nodes
#            for s in states:
#                condL = []
#                for c in n.child_node_iter():
#                    ctau = c.edge_length # get branch length
#                    ctaxon = c.taxon.label
#                    child_condL = 0.
#                    for x in states:
#                        pij = (ctau,s,x)
#                        try:
#                            mmP = Pijs[pij] # cached Markov probability
#                        except KeyError:
#                            Pijs[pij] = markov_model.P(*pij)
#                            mmP = Pijs[pij]
#                        child_condL += mmP * nodeLs[ctaxon][x]
#                    condL.append(child_condL)
#                nodeLs[taxon][s] = reduce(operator.mul, condL)
#        
#        if n == tree.seed_node: # termination after root cond Ls
#            treeL = 0.
#            for s in states:
#                treeL += priors[s] * nodeLs[taxon][s]
#            if treeL == 0.:
#                return 0.
#            return log(treeL)
#            
#
#
#def getNodeProbs(tree,dataD,knownD,error_model,markov_model,priors):
#    '''Felsenstein's pruning algorithm with error models at the tips.
#
#    Returns ancestral state reconstructions of the two states for the tips
#    and interior nodes of the given tree.
#
#    ## Parameters ##
#        `tree`: a dendropy Tree object with interior taxa labeled.
#        `dataD`: dictionary mapping tip node labels to continuous data
#        `knownD`: dictionary mapping tip node labels to known states, or None
#        `error_model`: an ErrorModels instance
#        `markov_model`: a MarkovModels instance
#        `priors`: dictionary mapping each state to prior. From the stationary frequencies in the MarkovModel
#    `'''
#
#    # Defs for postorder trace
#    cdef Py_ssize_t s, x, j, k
#    cdef object n # dendropy node object
#    cdef int[2] states = [0,1]
#    cdef double child_condL
#    cdef str taxon, ctaxon, parent
#    cdef double ntau, ctau # branch lengths
#    cdef tuple pij # tuple of (tau,state1,state2)
#    cdef object nodeLs = defaultdict(dict)
#    cdef dict Pijs = {}
#    cdef list condL
#
#
#    # Defs for preorder trace
#    cdef double up_s, sister_term
#    cdef object nodeUps = defaultdict(dict)
#    cdef str sister
#    cdef double parent_term, parent_pij, atau
#
#    # Defs for final probability calculations
#    cdef object nodeProbs = defaultdict(dict)
#    cdef object numerators = defaultdict(dict)
#    cdef double denominator, numerator
#
#    ## Calculate "down" variables (conditional likelihoods)
#    for n in tree.postorder_node_iter():
#        taxon = n.taxon.label
#        if n != tree.seed_node: # if root edge_length is None
#            ntau = n.edge_length
#        if n.is_leaf(): # initialize cLs at leaves from error model or missing or known data
#
#            if taxon in knownD:
#                if pd.isnull(knownD[taxon]): # unknown
#                    pass
#                else: # state is known
#                    nodeLs[taxon] = {knownD[taxon]:1.,other_state(knownD[taxon]):0.}
#                    continue
#
#            if taxon in dataD:
#                datum = dataD[taxon]
#                if np.all( pd.isnull(datum) ): # missing data, unknown state
#                    nodeLs[taxon] = {states[0]:1.,states[1]:1.}
#                else:
#                    for s in states: # get Ls from error model
#                        nodeLs[taxon][s] = error_model.P(data=datum,state=s)
#
#                        # While we're already looping over states,
#                        # start populating Pijs
#                        for x in states:
#                            pij = (ntau,s,x)
#                            if pij not in Pijs:
#                                Pijs[pij] = markov_model.P(*pij)
#
#            else: # missing data
#                nodeLs[taxon] = {states[0]:1.,states[1]:1.}
#
#        else: # is interior node, get cLs from child nodes
#            for s in states:
#                condL = []
#                for c in n.child_node_iter():
#                    ctau = c.edge_length # get branch length
#                    ctaxon = c.taxon.label
#                    child_condL = 0.
#                    for x in states:
#                        pij = (ctau,s,x)
#                        try:
#                            mmP = Pijs[pij] # cached Markov probability
#                        except KeyError:
#                            Pijs[pij] = markov_model.P(*pij)
#                            mmP = Pijs[pij]
#                        child_condL += mmP * nodeLs[ctaxon][x]
#                    condL.append(child_condL)
#                nodeLs[taxon][s] = reduce(operator.mul, condL)
#
#    ## Calculate "up" variables
#    for n in tree.preorder_node_iter():
#        taxon = n.taxon.label
#        denominator = 0.
#        if n == tree.seed_node: # is root
#            for s in states:
#                nodeUps[taxon][s] = priors[s]
#                numerator = priors[s] * nodeLs[taxon][s]
#                denominator += numerator
#                numerators[taxon][s] = numerator
#        else:
#            ntau = n.edge_length
#            parent = n.parent_node.taxon.label
#            sister = n.sister_nodes()[0].taxon.label
#            for s in states: # is interior node or tip
#                up_s = 0.
#                for j in states: #outer sum over j
#                    parent_pij = Pijs[(ntau,j,s)]
#                    parent_term = nodeUps[parent][j] * parent_pij
#
#                    # inner sum over k
#                    sister_term = 0.
#                    for k in states:
#                        sister_pij = Pijs[(ntau,j,k)]
#                        sister_term += sister_pij * nodeLs[sister][k]
#                    up_s += parent_term * sister_term
#                nodeUps[taxon][s] = up_s
#                numerator = up_s * nodeLs[taxon][s]
#                denominator += numerator
#                numerators[taxon][s] = numerator
#        for s in states:
#            if denominator != 0.:
#                nodeProbs[taxon][s] = numerators[taxon][s]/denominator
#            else:
#                nodeProbs[taxon][s] = .5
#    return nodeProbs