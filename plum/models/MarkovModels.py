import abc
import numpy as np
from plum.util import cprobs
from scipy import stats, special
from plum.models.modelABC import Model

    
class TwoState(Model):

    '''
    A two-state continuous time Markov model with Q-matrix:
    
           0      1
        |------------
    0   |-alpha alpha
        | 
    1   | beta  -beta
    
    Parameters:
        `alpha` - the forward instantaneous transition rate
        `beta` - the backwards instantaneous transition rate
        `mu` - the average transition rate
        `pi0` - the stationary frequency of state 0 (no PPI)
        `pi1` - the stationary frequency of state 1 (PPI)
    '''
    
    ## Name and state attributes are class attributes/properties, not instance
    
    _independent_params = ['alpha','beta']
    _dependent_params = ['mu','pi0','pi1']
    _param_names = _independent_params + _dependent_params
    
    @property
    def independentParams(self):
        '''The names of the independent parameters'''
        return self._independent_params
        
    @property
    def paramNames(self):
        '''The names of all parameters'''
        return self._param_names
                    
    _states = [0,1]
    
    @property
    def states(self):
        '''The latent states of the model'''
        return self._states
        
    _outer_bounds = {'alpha':(0,np.inf),
                    'beta':(0,np.inf)}
    _param_types = {'alpha': np.float64,
                    'beta': np.float64}
                    
    
    
    def __init__(self,alpha=1.,beta=1.,freeparams=['alpha','beta']):

        #checks
        _ = [self._validateParam(p) for p in freeparams] # validate input free parameters
            
        #initialize run options
        self._free_params = freeparams
        self._param_bounds = self._outer_bounds.copy()
        
        #initialize independent params
        self._params = {}
        loader = dict(zip(['alpha','beta'],np.array([alpha,beta],dtype=np.float64)))# necessary so all parameter values go through updateParams
        self.updateParams(loader) # this will initialize dependent params
        
        # Do I want containers to track e.g. parameter updates?
        
    def __repr__(self):
        return "plum.models.MarkovModels.TwoState({})".format( ",".join("=".join([i,str(j)]) for i,j in self.freeParamDict.iteritems()) )
        
    def __str__(self):
        return "plum.models.MarkovModels.TwoState"
        
    ## Define instance properties ##
    
    @property
    def params(self):
        '''The current parameters of the model'''
        return self._params
        
    @property
    def freeParamDict(self):
        '''The current free parameters of the model'''
        return {i:self._params[i] for i in self._free_params}
    
    @property
    def freeParams(self):
        '''Get or set the parameters that will be optimized.
            - A list of strings'''
        return self._free_params
    
    @property
    def paramBounds(self):
        '''Get or set bounds of each parameter. Restricts the search space during optimization
            - Dictionary of tuples mapping strings to values.'''
        return self._param_bounds
        
    @property
    def stationaryFrequencies(self):
        '''The stationary frequencies, pi0 and pi1.'''
        return self._stat_freqs
        
    ## Define setting and updating functions ##
    
    @freeParams.setter
    def freeParams(self,newfreeparams):
        _ = [self._validateParam(x) for x in newfreeparams]
        self._free_params = newfreeparams
        
    @paramBounds.setter
    def paramBounds(self,newbounds):
        _ = [self._validateParam(p,value=None,bounds=bounds) for p,bounds in newbounds.iteritems()]
        self._param_bounds.update(newbounds)
        
    def updateParams(self,newvalues):
        '''Update with new parameters.
            `newvalues` - a dictionary mapping parameter names to new values'''
        # update independent parameters
        for name,val in newvalues.iteritems():
            val = np.float64(val)
            self._validateParam(name,value=val)
            self._params[name] = val
        # update dependent parameters
        self._params['mu'] =  self._params['beta'] + self._params['alpha']
        self._params['pi0'] = self._params['beta']/self._params['mu']
        self._params['pi1'] = 1 - self._params['pi0']
        self._stat_freqs = [[0,1],[self._params['pi0'],self._params['pi1']]]
        
    ## Define other necessary methods ##

    def _validateParam(self,param,value=None,bounds=None):
        '''Make sure `param` is and acceptable name of a independent parameter.
        Optionally check that `value` and/or `bounds` is within the acceptable bounds.
        
            - `param` is a string
            - `value` is an integer or float
            - `bounds` is a 2 tuple of integers or floats. Lower bound is first element'''
        # Check strings
        assert param in self._independent_params, "'{}' not a valid independent parameter name".format(param)
        # Check values
        if value != None:
            if type(value) == np.ndarray:
                assert value.dtype is self._param_types[param], \
                "Parameter '{}' must be {}".format(param,self._param_types[param])
            else:
                assert type(value) is self._param_types[param], \
                "Parameter '{}' must be {}, not {}".format(param,self._param_types[param],type(value))
            assert (value >= self._outer_bounds[param][0]) & (value <= self._outer_bounds[param][1]), \
                "'{0}' value {1} is out of acceptable range: {2}".format(
                    param,value,self._outer_bounds[param])
        # Check bounds
        if bounds != None:
            assert len(bounds) == 2, "Bounds must be a 2 tuple"
            assert (bounds[0] >= self._outer_bounds[param][0]) & (bounds[1] <= self._outer_bounds[param][1]), \
            "New bounds for parameter {0} is out of range: {1}\n Acceptable range is:{2}".format(
                param,bounds,self._outer_bounds[param])
        
    def P(self,tau,i,j):
        '''Return the probability of transitioning from state `i` to state `j` over time `tau`'''
        return cprobs.twoStatePmat(self._params['pi0'],self._params['pi1'],self._params['mu'],tau,i,j)
    

    
class TwoStateGamma(TwoState):

    '''
    Two-state model with Gamma distributed rates. Q-matrix:
    
           0      1
        |------------
    0   |-alpha alpha
        | 
    1   | beta  -beta
    
    Parameters:
        `alpha` - the forward instantaneous transition rate
        `beta` - the backwards instantaneous transition rate
        `ncats` - the number of discrete Gamma rate categories
        `shape` - the shape parameter for the Gamma distribution
        `rate` - the rate parameter for the Gamma distribution
        `mu` - the average transition rate
        `pi0` - the stationary frequency of state 0 (no PPI)
        `pi1` - the stationary frequency of state 1 (PPI)
    '''
    
    _independent_params = ['alpha','beta','shape','rate','ncats']
    _dependent_params = ['mu','pi0','pi1','rates']
    _param_names = _independent_params + _dependent_params
    _outer_bounds = {'alpha':(0,np.inf),
                    'beta':(0,np.inf),
                    'shape':(0,np.inf),
                    'rate':(0,np.inf),
                    'ncats':(1,100)}
    _param_types = {'alpha': np.float64,
                    'beta': np.float64,
                    'shape': np.float64,
                    'rate': np.float64,
                    'ncats': int,
                    'rates': np.dtype("float64")}
                    
    def __init__(self,alpha=1.,beta=1.,shape=1.,rate=1.,ncats=4,freeparams=['alpha','beta','shape','rate']):

        #checks
        _ = [self._validateParam(p) for p in freeparams] # validate input free parameters
            
        TwoState.__init__(self,alpha,beta,freeparams)    
        
        #initialize independent params
        loader = {} # necessary so all param values go through updateParams
        loader['shape'] = np.float64(shape)
        loader['rate'] = np.float64(rate)
        loader['ncats'] = ncats
        self.updateParams(loader)
        
        #initialize dependent params

        
        # Do I want containers to track e.g. parameter updates?
        
        
    def updateParams(self,newvalues):
        '''Update with new parameters.
            `newvalues` - a dictionary mapping parameter names to new values'''
        # update independent parameters
        update_rates = False
        update_matrix = False
        for name,val in newvalues.iteritems():
            if name in ['shape','rate']:
                update_rates = True
            else:
                update_matrix = True
            self._validateParam(name,value=val)
            self._params[name] = val
        # update dependent parameters
        if update_matrix:
            self._params['mu'] =  self._params['beta'] + self._params['alpha']
            self._params['pi0'] = self._params['beta']/self._params['mu']
            self._params['pi1'] = 1 - self._params['pi0']
            self._stat_freqs = [[0,1,],[self._params['pi0'],self._params['pi1']]]
        if update_rates:
            self._params['rates'] = np.array([i for i in self._DiscreteGammaRates()])
        assert update_rates == True or update_matrix == True
        
    def P(self,tau,r,i,j):
        '''Return the probability of transitioning between two states
        over time <tau>, given instantaneous transition rates <alpha> 
        and <beta>, and rate scaling factor <r>'''
        mat = {"00": self._params['pi0'] + self._params['pi1'] * np.exp( -(r*self._params['mu']*tau) ),
            "01": self._params['pi1'] - self._params['pi1'] * np.exp( -(r*self._params['mu']*tau) ),
            "10": self._params['pi0'] - self._params['pi0'] * np.exp( -(r*self._params['mu']*tau) ),
            "11": self._params['pi1'] + self._params['pi0'] * np.exp( -(r*self._params['mu']*tau) )}
        return mat["".join(map(str,[i,j]))]

    ## Private methods ##
    
    def _DiscreteGammaRates(self):
        '''Generate percentage points (abcissa cuts) for a Gamma with
        shape and rate parameters and ncats rate categories.'''
        scale = 1./self._params['rate'] # scipy usesscale param for Gamma, rather than rate
        for k in np.arange(self._params['ncats']):
            if k == 0: # first section
                lowcut = 0
                upcut = stats.gamma.ppf( (k+1)/float(self._params['ncats']), 
                                            self._params['shape'], scale=scale )
                gammainc_lower = 0
                gammainc_upper = special.gammainc(self._params['shape']+1,upcut*self._params['rate'])
            elif k == self._params['ncats'] - 1: # last section
                lowcut = upcut # from last iter
                upcut = np.inf
                gammainc_lower = gammainc_upper
                gammainc_upper = 1.
            else:
                lowcut = upcut # from last iter
                upcut = stats.gamma.ppf( (k+1)/float(self._params['ncats']), 
                                            self._params['shape'], scale=scale)
                gammainc_lower = gammainc_upper
                gammainc_upper = special.gammainc(self._params['shape']+1,upcut*self._params['rate'])
            yield self._params['shape'] * self._params['ncats'] * (gammainc_upper - gammainc_lower) / self._params['rate']

        
