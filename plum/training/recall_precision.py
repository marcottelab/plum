import copy
import operator
import itertools
import scipy.stats
import numpy as np
import pandas as pd
import plum.util.data
import scipy.optimize
import sklearn.metrics

from plum.models.modelABC import Model
#from plum.util.cpruning import getNodeProbs, getNodeProbs_prior_weighted
from plum.util.cpruning import cgetNodeProbs
from plum.util.cfitting import _simulated_annealing, _simulated_annealing_multivariate



class plumRecallPrecision(plum.util.data.plumData):
    '''Base class for optimizers that use recall-precision.
    
    Input file should be training data with each pairs having at least one taxon with a known state.
    The known labels will be ignored for inferring ancestral state probabilities, but will be used to
    compare predictions to when calculating recall-precision'''
    
    def __init__(self,markov_model,error_model,tree,data=None,as_sorted=False,prior_weighted=False):
        plum.util.data.plumData.__init__(self,markov_model,error_model,tree,data,as_sorted)
        
        assert type(prior_weighted) is bool, "`prior_weighted` must be boolean"
        if prior_weighted:
            raise Exception("**option `prior_weighted` needs to be deleted**")
            self._prior_weighted = True
            self._calc_ancStates = np.vectorize(getNodeProbs_prior_weighted)
        else:
            self._prior_weighted = False
            #self._calc_ancStates = np.vectorize(getNodeProbs)
            self._calc_ancStates = cgetNodeProbs
        
        self._outputDF = None
        self._blank_knownDs = np.array( [{}] * len(self._featureDs) )
        
        print "Initializing model"
        self._update_all(self._param_vec) # calculate initial state
        print "Finished initializing model"
        
        self._precision_recall_dataframe = None
        self._results_dataframe = None
        
    #@property
    #def prior_weighted(self):
    #    '''Whether to weight the likelihood under the error model by
    #    the state prior from the stationary frequencies'''
    #    return self._prior_weighted
    
    #@prior_weighted.setter
    #def prior_weighted(self,pw):
    #    assert type(pw) is bool, "`prior_weighted` must be boolean"
    #    if pw == True:
    #        self._prior_weighted = True
    #        self._calc_ancStates = np.vectorize(getNodeProbs_prior_weighted)
    #    else:
    #        self._prior_weighted = False
    #        self._calc_ancStates = np.vectorize(getNodeProbs)
            
        
    @property
    def recall(self):
        '''Recall under current model'''
        return self._recall
    
    @property
    def precision(self):
        '''Precision under current model'''
        return self._precision
        
    @property
    def thresholds(self):
        '''Recall-precision thresholds under current model'''
        return self._thresholds
      
    @property
    def aps(self):
        '''Area under the recall-precision curve under current model'''
        return self._aps
      
    @property
    def probabilities(self):
        '''Vector of probabilities of known states under current model'''
        return self._classifier_probs
        
    @property
    def precision_recall_DF(self):
        '''Return a dataframe with the precision recall curve'''
        if self._precision_recall_dataframe is None:
            self._precision_recall_dataframe = pd.DataFrame( {"precision":self.precision, "recall":self.recall} )
            self._precision_recall_dataframe.sort_values("recall", inplace=True)
        return self._precision_recall_dataframe
       
    @property
    def results_DF(self):
        if self._results_dataframe is None:
            self._results_dataframe = pd.DataFrame( {"ID1": self._outID1,
                                                     "ID2": self._outID2,
                                                     "species": self._outspecies,
                                                     "state": self._labels,
                                                     "P_1": self._classifier_probs},
                                                     columns=["ID1","ID2","species","state","P_1"]
                                                     ).sort_values("P_1",ascending=False)
            self._results_dataframe["FDR"] = 1 - ( self._results_dataframe["state"].cumsum() / (np.arange(self._results_dataframe.shape[0])+1) )
            self._results_dataframe.index = np.arange(self._results_dataframe.shape[0])
        return self._results_dataframe

    def update_from_dict(self,param_dict,save_state=False):
        '''Update the parameters from a dictionary mapping parameter names to values.
        Doesn't need to include all parameters and will only update parameters included
        in the provided dictionary.
        
        If `save_state` is True, the previous state of the model will be saved in hidden
        attributes, this is only used internally by MCMC samplers at the moment'''
        
        if save_state:
            self._last_classifier_probs = copy.deepcopy(self._classifier_probs)
            self._last_params_vec = copy.deepcopy(self._param_vec)
            self._last_paramD = self._paramD.copy()
            self._last_pairAncStates = self._pairAncStates.copy()
            self._last_aps = copy.deepcopy(self._aps)
        
        for name,val in param_dict.iteritems():
            assert name in self._paramD, "Invalid parameter name for this model: {}".format(name)
            self._paramD[name] = np.float64( val ) # this is only a good idea so far as *every* parameter should have this type
            
        self._param_vec = [self._paramD[i] for i in self._free_params]
        
        # use properties to update models (they parse self._paramD)
        self._error_model.updateParams(self.errorModelParams)
        self._markov_model.updateParams(self.markovModelParams)
        self._state_priors = dict(zip(*self._markov_model.stationaryFrequencies))
        
        # An empty dictionary is passed to knownD argument, so all labels in dataset will be
        # ignored. This is something that I could do differently
        
        
        self._pairAncStates = self._calc_ancStates(self.tree,self._featureDs,self._blank_knownDs,
                                                    self._error_model,
                                                    self._markov_model,
                                                    self._state_priors)
        self._aps = self._calc_aps()
        
    def _reset_last(self):
        '''Reset to the last set of parameters, likelihoods etc.
        This is primarily for MCMC.'''
        assert self._last_params_vec != None, "No previous states saved"
        self._param_vec = self._last_params_vec
        self._paramD = self._last_paramD
        self._pairAncStates = self._last_pairAncStates
        self._aps = self._last_aps
        self._classifier_probs = self._last_classifier_probs
        self._error_model.updateParams(self.errorModelParams)
        self._markov_model.updateParams(self.markovModelParams)
 
    def _calc_aps(self):
        '''Calculate the area under the recall-precision curve for all labeled taxa'''
        self._classifier_probs = [] # probabilities of known tips from ancestral state reconstruction
        self._outspecies = []
        self._outID1, self._outID2 = [], []
        self._labels = []
        for index,nodeD in enumerate(self._knownLabelDs):
            for taxon,label in nodeD.iteritems():
                if not pd.isnull(label):
                    try:
                       #prob = self._pairAncStates[index][taxon][1]
                       prob = self._pairAncStates[index][taxon]
                    except KeyError:
                        raise Exception("{}".format(self._pairAncStates[index]))
                    if pd.isnull(prob): # seems that these can be nan, perhaps with certain parameter combinations?
                        prob = 0     # not sure this is the right way to do this. Got almost all nans for both means=0, both sds=.1, alpha/beta=.2
                    self._labels.append(label)
                    self._classifier_probs.append(prob)
                    self._outID1.append(self._pairs[index][0])
                    self._outID2.append(self._pairs[index][1])
                    self._outspecies.append(taxon)
        assert len(self._classifier_probs) == len(self._labels)
        self._precision,self._recall,self._thresholds = sklearn.metrics.precision_recall_curve(self._labels,self._classifier_probs,pos_label=1.)
        return sklearn.metrics.average_precision_score(self._labels,self._classifier_probs)
            
    
    
    def _update_all(self,param_array,save_state=False):
        '''Update all the params.
            1. First turn parameter vector to dictionary keyed on parameters names. Works
                on the fact that self._free_params is list of keys phased to self._param_vec,
                the values
            2. Then assign this dictionary to self._paramD
            3. Then use markovModelParams and errorModelParams properties to update the models
                themselves. These will be passed to the tree likelihood function
            4. Finally, call the tree likelihood function (Felsenstein pruning algorithm) and
                update the current site and tree likelihoods'''
        
        if save_state:
            self._last_classifier_probs = copy.deepcopy(self._classifier_probs)
            self._last_params_vec = copy.deepcopy(self._param_vec)
            self._last_paramD = self._paramD.copy()
            self._last_pairAncStates = self._pairAncStates.copy()
            self._last_aps = copy.deepcopy(self._aps)
        self._param_vec = param_array # could do this via a @property setter method instead
        assert len(self._param_vec) == len(self._free_params), "Why aren't parameter vector and free params same length"
        self._paramD = dict(zip(self._free_params,self._param_vec)) # update main holder of parameters
        
        # use properties to update models (they parse self._paramD)
        self._error_model.updateParams(self.errorModelParams)
        self._markov_model.updateParams(self.markovModelParams)
        self._state_priors = dict(zip(*self._markov_model.stationaryFrequencies))
        
        # An empty dictionary is passed to knownD argument, so all labels in dataset will be
        # ignored. This is something that I could do differently
        
        self._pairAncStates = self._calc_ancStates(self.tree,self._featureDs,self._blank_knownDs,
                                                    self._error_model,
                                                    self._markov_model,
                                                    self._state_priors)
        self._aps = self._calc_aps()
        
        
    def write_parameter_file(self,outfile):
        '''Write current parameters to a parameter file
            - `outfile`: <str> File to write to (will overwrite an existing file of the same name)'''
        plum.util.data.write_parameter_file(error_model=self._error_model,markov_model=self._markov_model,outfile=outfile)
        
class sweep(plumRecallPrecision):
    '''Perform simple 1-dimensional sweep on a model parameter, logging effect on aps'''

    def __init__(self,markov_model,error_model,param,bound,tree,data=None,as_sorted=False,prior_weighted=False,step=.1):
        plumRecallPrecision.__init__(self,markov_model,error_model,tree,data,as_sorted,prior_weighted)
        self._param_sweep = []
        self._aps_vec = []
        self._best_aps = None
        self._best_param = None
        self._param_to_sweep = param
        self._bound = bound
        self._step = float(step)
        
    @property
    def bestAPS(self):
        '''Return the best APS value found on the sweep'''
        return self._best_aps
        
    @property
    def APS_sweep(self):
        '''Return the vector of APS values over the sweep'''
        return self._aps_vec
        
    @property
    def param_sweep(self):
        '''Return the vector of parameter values swept over'''
        return self._param_sweep
    
    def sweep(self):
        '''Sweep a single parameter value over given range, saving states and best aps and parameter
        values'''
        for i in np.arange(self._bound[0],self._bound[1],self._step):
            self._paramD[self._param_to_sweep] = i
            new_vec = [self._paramD[p] for p in self._free_params]
            self._update_all(new_vec)
            self._param_sweep.append(i)
            self._aps_vec.append(self._aps)
            if self._aps > self._best_aps:
                self._best_aps = self._aps
                self._best_param = i
                
            
        
class mcmc(plumRecallPrecision):
    '''MCMC sampling using area under the recall-precision curve as a metric'''

    def __init__(self,markov_model,error_model,outfile,tree,data,as_sorted=False,prior_weighted=False,n_iters=10000,save_every=10,stringency=1,scale_dict=None):
        '''
            -`markov_model`: A plum.models.MarkovModels object. Its parameters will be used as a starting point for optimization
            -`error_model`: A plum.models.ErrorModels object. Its parameters will be used as a starting point for optimization
            -`outfile`: File to write iterations to
            -`tree`: Path to the input newick tree
            -`data`: A file in tidy data format.
            -`as_sorted`: whether input data is presorted on ID1,ID2
            -`n_iters`: number of MCMC iterations to perform
            -`save_every`: Save state after this many of generations
            -`stringency`: Scaling factor for acceptance ratio. Must be between 0 and 1, where 1 does not scale the Hastings ratio, and
                            0 means a lower APS will never be accepted
            -`scale_dict`: Standard deviations of normal distributions to sample parameter proposals from. Dictionary mapping parameter 
                            names to numbers. Defaults to .5 for every parameter
        '''
        plumRecallPrecision.__init__(self,markov_model,error_model,tree,data,as_sorted,prior_weighted)
        
        # Run params
        self._map_aps = None
        self._map_params = None
        self.n_iters = n_iters
        self.save_every = save_every
        self.outfile = outfile
        self._accepts = []
        assert np.logical_and(stringency >= 0, stringency <= 1), "`stringency` must be between 0 and 1"
        self._stringency = stringency
        self._scale_dict = {p:.5 for p in self.freeParams} #
        if scale_dict != None:
            for p,val in scale_dict:
                assert p in self.freeParams, "Parameter '{}' in scale_dict not found".format(p)
                self._scale_dict[p] = val
        
    @property
    def best_parameters(self):
        '''The free parameters with the highest APS found in the search and that APS value.
        Returns a tuple of the parameters dictionary and the APS value'''
        return self._map_params, self._map_aps
        
    def _draw_proposal(self):
        '''Draw a new parameter value from free parameters and update model'''
        
        # Right now, only move is a normally distributed step. Worth thinking
        # about the right way to make this more flexible, using scaling moves etc.
        # the way RevBayes does
        
        param = np.random.choice(self.freeParams) # could implement weights here
        proposal = scipy.stats.norm.rvs(loc=self._paramD[param],scale=self._scale_dict[param]) # I could tune the scale here
        
        ## hacky hard coding time!! ##
        if param in ["alpha",'beta','sd0','sd1']:
            proposal = abs(proposal)
            
        tmpD = self._paramD.copy()
        tmpD[param] = proposal
        new_vec = [tmpD[p] for p in self._free_params]
        self._update_all(new_vec,save_state=True)
    
    def metropolis_hastings(self):
        '''Accept or reject new proposal with the Hastings ratio'''
        old_state = self._paramD
        self._draw_proposal()
        alpha =  self._aps / float(self._last_aps)
        if alpha > 1:
            self._accepts.append(True)
            if self._aps > self._map_aps:
                self._map_aps = copy.deepcopy(self._aps)
                self._map_params = copy.deepcopy(self._paramD)
        else:
            scaled = alpha * self._stringency
            pull = np.random.uniform()
            if pull >= scaled: # don't accept
                self._reset_last()
                self._accepts.append(False)
            else:
                self._accepts.append(True)
                if self._aps > self._map_aps:
                    self._map_aps = copy.deepcopy(self._aps)
                    self._map_params = copy.deepcopy(self._paramD)
                
     
    def run(self):
        '''Run the MCMC sampler, writing to outfile'''
        out = open(self.outfile,'w')
        out.write(",".join(["gen","APS"]+self._free_params)+ "\n")
        gen = 0
        is_first = True
        while gen <= self.n_iters:
            if is_first: # initialize
                self._update_all(self._param_vec)
                is_first = False
            else:
                self.metropolis_hastings()
            print gen
            gen += 1
            if gen % self.save_every == 0:
                out.write(",".join([str(gen),str(self._aps)] + map(str,self._param_vec)) + "\n")
        out.close()
        
class basinhopping(plumRecallPrecision):
    '''Wrapper for scipy.optimize.basinhopping. Fits the model by basin-hopping using L-BFGS-B bounded
    minimization. Can modify bounds via the input error model and Markov model.
    
    -`markov_model`: A plum.models.MarkovModels object. Its parameters will be used as a starting point for optimization
    -`error_model`: A plum.models.ErrorModels object. Its parameters will be used as a starting point for optimization
    -`tree`: Path to the input newick tree or dendropy Tree object
    -`data`: A file in tidy data format or DataFrame 
    -`as_sorted`: whether input data is presorted on ID1,ID2
    -`n_iters`: Number of iterations for the basin-hopping algorithm to perform. Default = 100
    -`temp`: Temperature scaling for the Metropolis acceptance criterion. See scipy docs. Default = 1.0
    '''
    
    def __init__(self,markov_model,error_model,tree,data,as_sorted=False,prior_weighted=False,n_iters=100,temp=1.0,stepsize=1.0):
        plumRecallPrecision.__init__(self,markov_model,error_model,tree,data,as_sorted,prior_weighted)
        self.n_iters = n_iters
        self.temp = temp
        self.stepsize = stepsize
    
    def fit(self):
        '''Fit the model by basin-hopping + L-BFGS-B'''
        print "Fitting model by basinhopping"
        self.results = scipy.optimize.basinhopping(self._run_calc,self._param_vec,disp=True,niter=self.n_iters, T=self.temp,
                        stepsize=self.stepsize,minimizer_kwargs={"method":"L-BFGS-B",'bounds':self._param_boundVec})
        self.estimate = dict(zip(self._free_params,self.results.x))
        self.best_aps = 1 - self.results.fun
            
    def _run_calc(self,param_array):
        '''When called, reset self._param_vec and return negative log likelihood
        to feed into scipy.optimize.minimize'''
        self._update_all(param_array)
        return 1 - self.aps
        
class simulated_annealing(plumRecallPrecision):
    '''Fit a PLVM by simulated annealing.
    
    -`markov_model`: A plum.models.MarkovModels object. Its parameters will be used as a starting point for optimization
    -`error_model`: A plum.models.ErrorModels object. Its parameters will be used as a starting point for optimization
    -`tree`: Path to the input newick tree or dendropy Tree object
    -`data`: A file in tidy data format or DataFrame 
    -`as_sorted`: whether input data is presorted on ID1,ID2
    -`start_temp`: Starting temperature for the simulated annealing.
    -`alpha`: Parameter that decrements the temperature.
    -`temp_steps`: Number of steps to take at each temperature.
    -`mutation_sd`: Standard deviation of the sampling distribution (a Gaussian)
    '''
    
    def __init__(self,markov_model,error_model,tree,data,as_sorted=False,prior_weighted=False,start_temp=1, alpha=.9, temp_steps=10, mutation_sd=.5):
        plumRecallPrecision.__init__(self,markov_model,error_model,tree,data,as_sorted,prior_weighted)
        self.start_temp = start_temp
        self.alpha = alpha
        self.temp_steps = temp_steps
        self.mutation_sd = mutation_sd
        
    def _run_calc(self,param_array):
        self._update_all(param_array)
        return self.aps
    
    @property
    def alpha(self):
        ''''''
        return self._alpha
        
    @alpha.setter
    def alpha(self, new_value):
        assert np.logical_and( 0 < new_value, new_value < 1), "Alpha must be between 1 and 0"
        self._alpha = new_value
        
    @property
    def start_temp(self):
        ''''''
        return self._start_temp
        
    @start_temp.setter
    def start_temp(self, new_value):
        assert new_value > .000001, "Starting temperature must be greater than .000001"
        self._start_temp = new_value
        
    @property
    def temp_steps(self):
        return self._temp_steps
        
    @temp_steps.setter
    def temp_steps(self, new_value):
        assert new_value > 0, "Temperature steps must be an integer greater than 0"
        assert type(new_value) is int, "Temperature steps must be an integer"
        self._temp_steps = new_value
        
    @property
    def mutation_sd(self):
        return self._mutation_sd
        
    @mutation_sd.setter
    def mutation_sd(self, new_value):
        self._mutation_sd = new_value
        
    def fit(self):
        '''Run the simulated annealing procedure. Populate class fields:
            - best_params: The best parameters found during the run
            - best_aps: The best fit, as determined by the average precision score (APS)
            - acceptance_rate: The proportion of steps that were accepted.
        '''
        pvec = np.array(self._param_vec)
        if self.is_multivariate:
            best_params, best_score = _simulated_annealing_multivariate(self._run_calc, pvec, self._param_boundVec, 
                                                        self.start_temp, self.alpha, self.temp_steps, self.mutation_sd)
        else:
            best_params, best_score = _simulated_annealing(self._run_calc, pvec, self._param_boundVec, 
                                                        self.start_temp, self.alpha, self.temp_steps, self.mutation_sd)
        self.best_params = dict(zip(self.freeParams,best_params))
        self.best_aps = best_score
        
        self._update_all(best_params) # final update turns model state to best parameters found during search
        self._param_vec = best_params # re-assign parameter vector as well