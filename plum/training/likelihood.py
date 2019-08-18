import os
import copy
import random
import dendropy
import operator
import scipy.stats
import numpy as np
import pandas as pd
import plum.util.data
import scipy.optimize

from plum.models.modelABC import Model
from plum.util.cpruning import ctreeL
from plum.util.cfitting import _simulated_annealing, _simulated_annealing_multivariate

class plumLikelihood(plum.util.data.plumData):
    '''
    Base model for the likelihood of data under a PLVM model
    '''
    
    def __init__(self,markov_model,error_model,tree,data=None):
        '''
        - `tree`: A rooted newick tree with labeled internal nodes and root
        - `datafile`:
        - `starting_params`: Reasonable starting parameter estimates for the ML search, <dictionary>
        - `params_bounds`: Restrict the search space for the ML algorithm. <dictionary>
        - `missing_key`: The data value to interpret as missing data
        '''
        
        plum.util.data.plumData.__init__(self,markov_model,error_model,tree,data)
        
        self._siteLs = None
        self._logL = None

        self._calc_L = ctreeL # vectorized version of _treeL
        self._blank_knownDs = np.array( [{}] * len(self._featureDs) )
        
        print("Initializing model")
        self._update_all(self._param_vec) # calculate initial state
        print("Finished initializing model")
        
    @property
    def markovModelParams(self):
        '''Return dictionary of Markov Model Parameters given current'''
        return {p:self._paramD[p] for p in self._mm_param_names}
        
    @property
    def errorModelParams(self):
        '''Return dictionary of current Error Model Parameters'''
        return {p:self._paramD[p] for p in self._em_param_names}
        
    @property
    def freeParams(self):
        '''The parameters that will be optimized.
            - A list of strings'''
        return self._free_params
        
    @property
    def logL(self):
        '''The current log likelihood of the model'''
        return self._logL
    
    def _reset_last(self):
        '''Reset to the last set of parameters, likelihoods etc.
        This is primarily for MCMC.'''
        assert self._last_params_vec != None, "No previous states saved"
        self._param_vec = self._last_params_vec
        self._paramD = self._last_paramD
        self._siteLs = self._last_siteLs 
        self._logL = self._last_logL
        self._error_model.updateParams(self.errorModelParams)
        self._markov_model.updateParams(self.markovModelParams)
    
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
            self._last_params_vec = self._param_vec
            self._last_paramD = self._paramD
            self._last_siteLs = self._siteLs
            self._last_logL = self._logL
        self._param_vec = param_array # could do this via a @property setter method instead
        assert len(self._param_vec) == len(self._free_params), "Why aren't parameter vector and free params same length"
        self._paramD = dict(list(zip(self._free_params,self._param_vec))) # update main holder of parameters
        
        # use properties to update models (they parse self._paramD)
        self._error_model.updateParams(self.errorModelParams)
        self._markov_model.updateParams(self.markovModelParams)
        self._state_priors = dict(list(zip(*self._markov_model.stationaryFrequencies)))
        
        self._siteLs = self._calc_L(self.tree,self._featureDs,self._blank_knownDs,
                                                        self._error_model,
                                                        self._markov_model,
                                                        self._state_priors)
        
        if not type(self._siteLs) is np.ndarray: # for some reason this seems possible. Need to check pruning algorithm.
            if self._siteLs == 0.0:
                self._logL = -np.inf
            else:
                raise Exception("Invalide site liklihood value. Type: {}\n Value:{}".format(type(self._siteLs),self._siteLs))
        else:
            self._logL = np.sum(self._siteLs)
        
    def write_parameter_file(self,outfile):
        '''Write current parameters to a parameter file
            - `outfile`: <str> File to write to (will overwrite an existing file of the same name)'''
        plum.util.data.write_parameter_file(error_model=self._error_model,markov_model=self._markov_model,outfile=outfile)
        

class plumMCMC(plumLikelihood):
    '''Estimate model parameters by MCMC'''
    
    def __init__(self,treefile,datafile,markov_model,error_model,outfile,priors=None,n_iters=10000,save_every=10,missing_key=None,run_under_prior=False):
        '''
            -`priors`: dictionary mapping parameter names to scipy.stats models
            -`niters`: number of MCMC iterations to perform
        '''
        plumLikelihood.__init__(self,markov_model,error_model,tree=treefile,data=datafile)
        
        # Default Priors
        if priors == None:
            self._prior_dists = {i:scipy.stats.norm(loc=self._paramD[i],scale=1) for i in self._free_params if "mean" in i} # set priors to means ~N(x,1)
            self._prior_dists.update({i:scipy.stats.betaprime(self._paramD[i],self._paramD[i]) for i in self._free_params if "sd" or "lambda" in i}) # sd priors ~BetaPrime(x,x)
            self._prior_dists.update({"alpha":scipy.stats.uniform(0,50),"beta":scipy.stats.uniform(0,50)})
        else:
            self._prior_dists = priors # distributions
        assert all([x in self._paramD for x in (i for i in self._prior_dists)]), "Priors {} don't match parameters {}".format(list(priors.keys()),list(self._paramD.keys()))
        self._start_log_priors = {param: np.log( self._prior_dists[param].pdf(self._paramD[param]) ) for param in self._prior_dists}
        self._old_log_prior = None
        
        self.n_iters = n_iters
        self.save_every = save_every
        self.outfile = outfile
        self._accepts = []
        self._mcmc_bounds = copy.deepcopy(self._error_model.paramBounds)
        self._mcmc_bounds.update(self._markov_model.paramBounds)
        
    def _draw_proposal(self):
        
        param = np.random.choice(self.freeParams) # could implement weights here
        proposal = scipy.stats.norm.rvs(loc=self._paramD[param],scale=.5) # I could tune the scale here
        while np.logical_and( proposal > self._mcmc_bounds[param][0], proposal < self._mcmc_bounds[param][1]) == False:
            proposal = scipy.stats.norm.rvs(loc=self._paramD[param],scale=.5)
        if self._old_log_prior == None:
            self._old_log_prior = self._start_log_priors[param]
        self._curr_log_prior = np.log( self._prior_dists[param].pdf(proposal) )
        
        tmpD = self._paramD.copy()
        tmpD[param] = proposal
        new_vec = [tmpD[p] for p in self._free_params]
        self._update_all(new_vec,save_state=True)
    
    def metropolis_hastings(self):
        old_state = self._paramD
        self._draw_proposal()
        expr = np.exp( (self._curr_log_prior + self._logL) - (self._old_log_prior + self._last_logL) )
        alpha = min(expr,1)
        pull = np.random.uniform()
        if pull >= alpha: # don't accept
            self._reset_last()
            self._accepts.append(False)
        else:
            self._accepts.append(True)
            
     
    def run(self):   
        out = open(self.outfile,'w')
        out.write(",".join(["gen","logL"]+self._free_params)+ "\n")
        gen = 0
        is_first = True
        while gen <= self.n_iters:
            if is_first: # initialize
                self._update_all(self._param_vec)
                is_first = False
            else:
                self.metropolis_hastings()
            print(gen)
            gen += 1
            if gen % self.save_every == 0:
                out.write(",".join([str(gen),str(self._logL)] + list(map(str,self._param_vec))) + "\n")
        out.close()
            
            
            
class plumML(plumLikelihood):
    '''Fit model by maximum likelihood'''
    
   ## Currently sorta working but ML fitting not changing the error model parameters for some reason.
   ## Also the data reading, especially the dtype setting of the "state" column and use of `missing_key`
   ## is in total disarray. These could be related, should step away and check back on this later.
    
    def __init__(self,treefile,datafile,markov_model,error_model,missing_key=None):
        plumLikelihood.__init__(self,treefile,datafile,markov_model,error_model,missing_key)
    
    def fit(self):
        '''Fit the model '''
        ML = scipy.optimize.minimize(self._run_calc,self._param_vec,method="L-BFGS-B",bounds=self._param_boundVec,
                                        options={'maxiter':100,'disp':True})
        if ML.success:
            self.ml_estimate = dict(list(zip(self._free_params,ML.x)))
            self.results = ML
        else:
            self.failure = ML
            
        
    def _run_calc(self,param_array):
        '''When called, reset self._param_vec and return negative log likelihood
        to feed into scipy.optimize.minimize'''
        self._update_all(param_array)
        return -self._logL
        

class simulated_annealing(plumLikelihood):
    '''Fit a PLVM by simulated annealing using likelihood.
    
    -`markov_model`: A plum.models.MarkovModels object. Its parameters will be used as a starting point for optimization
    -`error_model`: A plum.models.ErrorModels object. Its parameters will be used as a starting point for optimization
    -`tree`: Path to the input newick tree or dendropy Tree object
    -`data`: A file in tidy data format or DataFrame 
    -`start_temp`: Starting temperature for the simulated annealing.
    -`alpha`: Parameter that decrements the temperature.
    -`temp_steps`: Number of steps to take at each temperature.
    -`mutation_sd`: Standard deviation of the sampling distribution (a Gaussian)
    '''
    
    def __init__(self,markov_model,error_model,tree,data,start_temp=1, alpha=.9, temp_steps=10, mutation_sd=.5):
        plumLikelihood.__init__(self,markov_model,error_model,tree,data)
        self.start_temp = start_temp
        self.alpha = alpha
        self.temp_steps = temp_steps
        self.mutation_sd = mutation_sd
        
    def _run_calc(self,param_array):
        self._update_all(param_array)
        return self._logL
    
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
            - best_logL: The best fit, as determined by the log likelihood
            - acceptance_rate: The proportion of steps that were accepted
        '''
        pvec = np.array(self._param_vec)
        if self.is_multivariate:
            best_params, best_score = _simulated_annealing_multivariate(self._run_calc, pvec, self._param_boundVec, 
                                                    self.start_temp, self.alpha, self.temp_steps, self.mutation_sd)
        else:
            best_params, best_score = _simulated_annealing(self._run_calc, pvec, self._param_boundVec, 
                                                            self.start_temp, self.alpha, self.temp_steps, self.mutation_sd)
        self.best_params = dict(list(zip(self.freeParams,best_params)))
        self.best_logL = best_score
        
        self._update_all(best_params) # final update turns model state to best parameters found during search
        self._param_vec = best_params # re-assign parameter vector as well