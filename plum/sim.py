import copy
import random
import dendropy
import numpy as np
import pandas as pd
from plum.models.modelABC import Model

## Suppress pandas' Future Warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class TwoStateSim():

    def __init__(self,tree,markov_model,error_model,nchars=100):
        '''
            - `tree`: dendropy tree object with labeled interior nodes as taxa
            -`markov_model`: A plum.models.MarkovModels object to simulate under
            -`error_model`: A plum.models.ErrorModels object to simulate under
        '''

        ## Might want to do this differently, e.g. have reader function for tree,
        ## instantiator functions for models, etc.
        assert isinstance(tree,dendropy.Tree), "`tree` must be dendropy.Tree instance"
        assert isinstance(markov_model,Model), "`markov_model` must be instance deriving from modelABC.Model"
        assert isinstance(error_model,Model), "`error_model` must be instance deriving from modelABC.Model"
        assert markov_model.states == error_model.states, "States in error and Markov models don't match"
        assert len(markov_model.states) == 2, "Model should have just two states"
        assert np.isclose( sum(markov_model.stationaryFrequencies[1]), 1. ), \
            "Stationary frequencies from Markov model don't sum to 1"
        
        self.nchars = nchars # fine to give access to this attribute    
        self._tree = tree
        self._states = markov_model.stationaryFrequencies[0] # list
        
        # Markov model setup
        self._stationary_frequencies = markov_model.stationaryFrequencies[1] # list
        self._forward_rate = markov_model._params["alpha"]
        self._backward_rate = markov_model._params["beta"]
        self._dwell_params = {self._states[0]:self._forward_rate,self._states[1]:self._backward_rate}
        
        self._error_model = error_model
        

    def _draw_root_states(self):
        '''Return a vector of root states drawn from stationary frequencies'''
        return np.random.choice(self._states,size=self.nchars,p=self._stationary_frequencies)
    
    def _site_evol(self,start_state,brlen):
        '''A two-state Markov process evolving along a branch of length `brlen`
        for a single site. Yields state at the end of the branch.'''

        s1,s2 = self._states
        get_newstate = {s1:s2,s2:s1}
        
        state = start_state
        switched = False
        t = 0.
        while t < brlen:
            dt = random.expovariate(self._dwell_params[state]) # time until next event: ~Exp(lambda = -qii = qij)
            t += dt
            if t >= brlen:
                break # no switch occurred on branch
            state = get_newstate[state]
            switched = True
        return state, switched
    
    def format_output(self,tip_data=False):
        
        ## Data munge munge
        states = pd.DataFrame(self._node_states)
        switches = pd.DataFrame(self._node_switches)
        data = pd.DataFrame(self._leaf_data)
        states["site"] = list(range(len(states)))
        switches["site"] = list(range(len(switches)))
        data["site"] = list(range(len(data)))
        taxa = [t.label for t in self._tree.taxon_namespace]
        tidy_states = pd.melt(states,id_vars=['site'],
                    value_vars=taxa,var_name='node',value_name='state').set_index(["site",'node'])
        tidy_switches = pd.melt(switches,id_vars=['site'],
                            value_vars=taxa,
                                var_name='node',value_name='switch').set_index(
                                    ["site",'node'])
        tidy_data = pd.melt(data,id_vars=['site'],
                            value_vars=taxa,
                                var_name='node',value_name='data').set_index(
                                    ["site",'node'])
                                    
        if self._error_model.is_multivariate:
            # split up the multivariate data column, which is currently a column of tuples,  into separate columns
            tidy_data[['data{}'.format(i) for i in range(self._error_model.size)]] = tidy_data['data'].apply(pd.Series)
            tidy_data.drop('data',inplace=True,axis=1)
        
        self.output = tidy_states.join(
            tidy_switches.join(
                tidy_data,how='outer'),how='outer')
        self.output.reset_index(inplace=True)
        
        if tip_data:
            tips = self.output.copy()
            tips.drop(['site',"switch"],inplace=True,axis=1)
            tips["ID1"] = self.output["site"]
            tips["ID2"] = self.output["site"]
            tip_taxa = [x.taxon.label for x in self._tree.leaf_node_iter()]
            self.tip_data = tips[tips.node.isin(tip_taxa)][["ID1","ID2","node"] + [c for c in tips.columns if "data" in c] + ["state"]]
            self.tip_data.columns = [["ID1","ID2","species"] + [c for c in tips.columns if "data" in c] + ["state"]]
            
        
    
    def __call__(self,node_outfile=None,tip_outfile=None,as_pickle=False,tip_data=False):
        '''Simulate stochastic evolution of two states along a tree, with
        an error model yielding continuous data at the tips.
        
        Populates self.output and optionally writes to outfile.'''
        
        branch_evol = np.vectorize(self._site_evol)
        if self._error_model.is_multivariate: # numpy needs to know how to read and output multivariate data
            error_vec = np.vectorize(self._error_model.draw, signature='()->(i)')
        else:
            error_vec = np.vectorize(self._error_model.draw)
        
        self._node_states = {}
        self._node_switches = {}
        self._leaf_data = {}
        
        current_states = None
        for n in self._tree.preorder_node_iter():
            assert n.taxon != None, "Nodes must have associated taxon objects"
            if n == self._tree.seed_node: # is root
                print("Drawing initial states at root")
                current_states = self._draw_root_states()
                switch_vec = np.array([np.nan]*self.nchars)
            else:
                print("Evolving to node {}".format(n.taxon.label))
                current_states, switch_vec = branch_evol(self._node_states[n.parent_node.taxon.label],n.edge_length)
            if n.is_leaf():
                if self._error_model.is_multivariate:
                    # tuple-ize multivariate data so it can be loaded into single dataframe column in format_output.
                    self._leaf_data[n.taxon.label] = list(map(tuple,error_vec(current_states)))
                else:
                    self._leaf_data[n.taxon.label] = error_vec(current_states)
            self._node_states[n.taxon.label] = current_states
            self._node_switches[n.taxon.label] = switch_vec
        
        self.format_output(tip_data) # populate self.output and optionally self.tip_data
        
        if node_outfile:
            if as_pickle:
                self.output.to_pickle(node_outfile)
            else:
                self.output.to_csv(node_outfile,index=False)
                
        if tip_outfile:
            if as_pickle:
                self.tip_data.to_pickle(tip_outfile)
            else:
                self.tip_data.to_csv(tip_outfile,index=False)