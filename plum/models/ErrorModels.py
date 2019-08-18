import abc
import numpy as np
from plum.util import cprobs
from scipy.stats import norm
from plum.models.modelABC import Model


class Gaussian(Model,object):

    '''
    Models binary latent states with two single Normals
    
    Parameters
        - mean0, sd0: The mean and standard deviation for state 0
        - mean1, sd1: The mean and standard deviation for state 1
    '''
    
    _param_names = ["mean0","sd0","mean1","sd1"]
        
    _outer_bounds = {"mean0":(-np.inf,np.inf),
                        "sd0":(0.,np.inf),
                        "mean1":(-np.inf,np.inf),
                        "sd1":(0.,np.inf)}
                        
    _param_types = {"mean0":np.float64,
                        "sd0":np.float64,
                        "mean1":np.float64,
                        "sd1":np.float64}
                        
    _states = [0,1]
    
    @property
    def is_multivariate(self):
        return False
    
    @property
    def states(self):
        '''The latent states of the model'''
        return self._states
    
    @property
    def paramNames(self):
        '''The names of all parameters'''
        return self._param_names
    
    def __init__(self,mean0=0.,sd0=1.,mean1=0.,sd1=1.,freeparams = ["mean0","sd0","mean1","sd1"]):
    
        loader = dict(list(zip(["mean0","sd0","mean1","sd1"],np.array([mean0,sd0,mean1,sd1],dtype=np.float64))))
        self._params = {}
        self.updateParams(loader)
        self._param_bounds = self._outer_bounds.copy()
        self._free_params = freeparams
        
    def __repr__(self):
        '''Representation for evaluation'''
        return "plum.models.ErrorModels.Gaussian({})".format( ",".join("=".join([i,str(j)]) for i,j in self.freeParamDict.items()) )
    
    def __str__(self):
        return "plum.models.ErrorModels.Gaussian"
        
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
         
    @freeParams.setter
    def freeParams(self,newfreeparams):
        _ = [self._validateParam(x) for x in newfreeparams]
        self._free_params = newfreeparams
        
    @property
    def paramBounds(self):
        '''Get or set bounds of each parameter. Restricts the search space during optimization
            - Dictionary of tuples mapping strings to values'''
        return self._param_bounds

    @paramBounds.setter
    def paramBounds(self,newbounds):
        _ = [self._validateParam(p,value=None,bounds=bounds) for p,bounds in newbounds.items()]
        self._param_bounds.update(newbounds)
        
    def updateParams(self,newvalues):
        '''Update with new parameters.
            `newvalues` - a dictionary mapping parameter names to new values'''
        # update independent parameters
        for name,val in newvalues.items():
            self._validateParam(name,value=val)
            self._params[name] = val
            
    def _validateParam(self,param,value=None,bounds=None):
        '''Check relevant properties of parameter(s), such as allowable bounds.
        This function should be called by all other methods that change or set
        the parameters. It should be a private function so it can't be tinkered with.'''
        # Check strings
        assert param in self._param_names, "'{}' not a valid parameter name".format(param)
        # Check value type and bound
        if value != None:
            assert type(value) is self._param_types[param], \
                "Parameter '{}' must be {}, not {}".format(param,self._param_types[param], type(value))
            assert (value >= self._outer_bounds[param][0]) & (value <= self._outer_bounds[param][1]), \
                "'{0}' value {1} is out of acceptable range: {2}".format(
                    param,value,self._outer_bounds[param])
        # Check bounds
        if bounds != None:
            assert len(bounds) == 2, "Bounds must be a 2 tuple"
            assert (bounds[0] >= self._outer_bounds[param][0]) & (bounds[1] <= self._outer_bounds[param][1]), \
            "New bounds for parameter {0} is out of range: {1}\n Acceptable range is:{2}".format(
                param,bounds,self._outer_bounds[param])
        
    def P(self,data,state):
        '''Probability using cprobs.pyx (Cython compiled probabilities) instead of 
        scipy.stats, which is quite slow'''
        if state == 0:
            return cprobs.normpdf(data,self._params["mean0"],self._params["sd0"])
        elif state == 1:
            return cprobs.normpdf(data,self._params["mean1"],self._params["sd1"])
        else:
            raise Exception("wtf")
            
    def draw(self,state):
        '''Draw random numbers from one of the two error models for simulation.
        `state`: Determines which state's error model to draw from'''
        if state == 0:
            return np.random.normal(self._params["mean0"],self._params["sd0"])
        elif state == 1:
            return np.random.normal(self._params["mean1"],self._params["sd1"])
        else:
            raise Exception("wtf")
            
class GaussianMix(Gaussian):
    
    '''
    Models negative interactions with a single Normal and positive with
    a mixture of two Normals.
    
    Parameters
        - mean0, sd0: The mean and standard deviation for state 0
        - lambda1_1: Mixture weight for the first component of the state 1 model
        - mean1_1: Mean of the first component of the state 1 model
        - sd1_1: Standard deviation of the first component of the state 1 model
        - mean1_2: Mean of the second component of the state 1 model
        - sd1_2: Standard deviation of the second component of the state 1 model
    '''
    
    _independent_params = ["mean0","sd0","lambda1_1","mean1_1","sd1_1","mean1_2","sd1_2"]
    _dependent_params = ["lambda1_2"]
    _param_names = _independent_params + _dependent_params
    
    @property
    def independentParams(self):
        '''The names of the independent parameters'''
        return self._independent_params
        
    _outer_bounds = {"mean0":(-np.inf,np.inf),
                        "sd0":(0,np.inf),
                        "lambda1_1":(0.,1.),
                        "mean1_1":(-np.inf,np.inf),
                        "sd1_1":(0,np.inf),                       
                        "lambda1_2":(0.,1.),
                        "mean1_2":(-np.inf,np.inf),
                        "sd1_2":(0,np.inf)}
                        
    _param_types = {"mean0":np.float64,
                        "sd0":np.float64,
                        "lambda1_1":np.float64,
                        "mean1_1":np.float64,
                        "sd1_1":np.float64,
                        "lambda1_2":np.float64,
                        "mean1_2":np.float64,
                        "sd1_2":np.float64}
                        
    
    def __init__(self,mean0=0.,sd0=1.,lambda1_1=.5,mean1_1=0.,sd1_1=1.,mean1_2=0.,sd1_2=1.,\
                    freeparams=["mean0","sd0","lambda1_1","mean1_1","sd1_1","mean1_2","sd1_2"]):
    
        #initialize independent parameters
        loader = {"mean0":np.float64(mean0),
                    "sd0":np.float64(sd0),
                    "lambda1_1":np.float64(lambda1_1),
                    "mean1_1":np.float64(mean1_1),
                    "sd1_1":np.float64(sd1_1),
                    "mean1_2":np.float64(mean1_2),
                    "sd1_2":np.float64(sd1_2)}
        self._params = {}
        self.updateParams(loader)
        
        #initialize dependent parameter
        self._params["lambda1_2"] = 1. - self._params['lambda1_1']
        
        self._param_bounds = self._outer_bounds.copy()
        self._free_params = freeparams

    def __repr__(self):
        '''Representation for evaluation'''
        return "plum.models.ErrorModels.GaussianMix({})".format( ",".join("=".join([i,str(j)]) for i,j in self.freeParamDict.items()) )
        
    def __str__(self):
        return "plum.models.ErrorModels.GaussianMix"
        
    def _validateParam(self,param,value=None,bounds=None):
        '''Check relevant properties of parameter(s), such as allowable bounds.
        This function should be called by all other methods that change or set
        the parameters. It should be a private function so it can't be tinkered with.'''
        # Check strings
        assert param in self._independent_params, "'{}' not a valid independent parameter name".format(param)
        # Check value type and bound
        if value != None:
            assert type(value) is self._param_types[param], \
                "Parameter '{}' must be {}".format(param,self._param_types[param])
            assert (value > self._outer_bounds[param][0]) & (value < self._outer_bounds[param][1]), \
                "'{0}' value {1} is out of acceptable range: {2}".format(
                    param,value,self._outer_bounds[param])
        # Check bounds
        if bounds != None:
            assert len(bounds) == 2, "Bounds must be a 2 tuple"
            assert (bounds[0] >= self._outer_bounds[param][0]) & (bounds[1] <= self._outer_bounds[param][1]), \
            "New bounds for parameter {0} is out of range: {1}\n Acceptable range is:{2}".format(
                param,bounds,self._outer_bounds[param])
                
    def updateParams(self,newvalues):
        '''Update with new parameters.
            `newvalues` - a dictionary mapping parameter names to new values'''
        # update independent parameters
        for name,val in newvalues.items():
            self._validateParam(name,value=val)
            self._params[name] = val
            if name == 'lambda1_1':
                self._params['lambda1_2'] = 1. - self._params['lambda1_1'] # update dependent parameter 
        
    def P(self,data,state):
        '''Return the probability of `data` under one of two normal distributions,
        depending on `state`.'''
        if state == 0:
            p = cprobs.normpdf(data,self._params["mean0"],self._params["sd0"])
            return p
        elif state == 1:
            p = cprobs.normMix2pdf(data, self._params["lambda1_1"], self._params["lambda1_2"],
                                           self._params["mean1_1"], self._params["mean1_2"],
                                           self._params["sd1_1"], self._params["sd1_2"])
            return p
        else:
            raise Exception("State {} no recognized".format(state))
            
    def draw(self,state):
        raise Exception("Haven't written this function yet")

class Gumbel(Gaussian):
    '''
    Models binary latent states with two single Gumbels
    
    Parameters
        - mean0, sd0: The mean and standard deviation for state 0
        - mean1, sd1: The mean and standard deviation for state 1
    '''
   
    def __repr__(self):
        '''Representation for evaluation'''
        return "plum.models.ErrorModels.Gumbel({})".format( ",".join("=".join([i,str(j)]) for i,j in self.freeParamDict.items()) )
        
    def __str__(self):
        return "plum.models.ErrorModels.Gumbel"
        
    def P(self,data,state):
        '''Probability using cprobs.pyx (Cython compiled probabilities) instead of 
        scipy.stats, which is quite slow'''
        if state == 0:
            return cprobs.gumbelpdf(data,self._params["mean0"],self._params["sd0"])
        elif state == 1:
            return cprobs.gumbelpdf(data,self._params["mean1"],self._params["sd1"])
        else:
            raise Exception("wtf")
            
    def draw(self,state):
        raise Exception("Haven't written this function yet")
            
class GumbelMix(GaussianMix):
    
    '''
    Models negative interactions with a single Gumbel and positive with
    a mixture of two Gumbels
    
    Parameters
        - mean0, sd0: The mean and standard deviation for state 0
        - lambda1_1: Mixture weight for the first component of the state 1 model
        - mean1_1: Mean of the first component of the state 1 model
        - sd1_1: Standard deviation of the first component of the state 1 model
        - mean1_2: Mean of the second component of the state 1 model
        - sd1_2: Standard deviation of the second component of the state 1 model
    '''
    
    def __repr__(self):
        '''Representation for evaluation'''
        return "plum.models.ErrorModels.GumbelMix({})".format( ",".join("=".join([i,str(j)]) for i,j in self.freeParamDict.items()) )
        
    def __str__(self):
        return "plum.models.ErrorModels.GumbelMix"
        
    def P(self,data,state):
        '''Return the probability of `data` under one of two normal distributions,
        depending on `state`.'''
        if state == 0:
            p = cprobs.gumbelpdf(data,self._params["mean0"],self._params["sd0"])
            return p
        elif state == 1:
            p = cprobs.gumbelMix2pdf(data, self._params["lambda1_1"], self._params["lambda1_2"],
                                           self._params["mean1_1"], self._params["mean1_2"],
                                           self._params["sd1_1"], self._params["sd1_2"])
            return p
        else:
            raise Exception("State {} no recognized".format(state))
                    
        
class Gamma(Gaussian):

    '''
    Models binary latent states with two single Gammas
    
    Parameters
        - k0, theta0: The shape and scale parameters for state 0
        - k1, theta1: The shape and scale parameters for state 1
    '''
    
    _param_names = ["k0","theta0","k1","theta1"]
        
    _outer_bounds = {"k0":(0.,np.inf),
                        "theta0":(0.,np.inf),
                        "k1":(0.,np.inf),
                        "theta1":(0.,np.inf)}
                        
    _param_types = {"k0":np.float64,
                        "theta0":np.float64,
                        "k1":np.float64,
                        "theta1":np.float64}
    
    def __init__(self,k0=0.,theta0=1.,k1=0.,theta1=1.,freeparams = ["k0","theta0","k1","theta1"]):
    
        loader = dict(list(zip(["k0","theta0","k1","theta1"],np.array([k0,theta0,k1,theta1],dtype=np.float64))))
        self._params = {}
        self.updateParams(loader)
        self._param_bounds = self._outer_bounds.copy()
        self._free_params = freeparams

    def __repr__(self):
        '''Representation for evaluation'''
        return "plum.models.ErrorModels.Gamma({})".format( ",".join("=".join([i,str(j)]) for i,j in self.freeParamDict.items()) )

    def __str__(self):
        return "plum.models.ErrorModels.Gamma"
        
    def P(self,data,state):
        '''Probability using cprobs.pyx (Cython compiled probabilities) instead of 
        scipy.stats, which is quite slow'''
        if state == 0:
            return cprobs.gammapdf(data,self._params["k0"],self._params["theta0"])
        elif state == 1:
            return cprobs.gammapdf(data,self._params["k1"],self._params["theta1"])
        else:
            raise Exception("wtf")
            
    def draw(self,state):
        '''Draw random numbers from one of the two error models for simulation.
        `state`: Determines which state's error model to draw from'''
        if state == 0:
            return np.random.gamma(self._params["k0"],self._params["theta0"])
        elif state == 1:
            return np.random.gamma(self._params["k1"],self._params["theta1"])
        else:
            raise Exception("wtf")
            
    def draw(self,state):
        raise Exception("Haven't written this function yet")
            
class GammaMix(GaussianMix):
    
    '''
    Models negative interactions with a single Gamma and postive with
    a mixture of two Gammas.
    
    Parameters
        - k0, theta0: The mean and standard deviation for state 0
        - lambda1_1, k1_1, theta1_1: The mixture weight, mean and standard deviation for the first component of the state 1 mixture
        - lambda1_2, k1_2, theta1_2: The mixture weight, mean and standard deviation for the second component of the state 1 mixture
          * Note that lambda1_2 is not a free parameter. It's just 1 - lambda1_1
    '''
    
    _independent_params = ["k0","theta0","lambda1_1","k1_1","theta1_1","k1_2","theta1_2"]
    _dependent_params = ["lambda1_2"]
    _param_names = _independent_params + _dependent_params
        
    _outer_bounds = {"k0":(0.,np.inf),
                        "theta0":(0,np.inf),
                        "lambda1_1":(0.,1.),
                        "k1_1":(0.,np.inf),
                        "theta1_1":(0,np.inf),                       
                        "lambda1_2":(0.,1.),
                        "k1_2":(0.,np.inf),
                        "theta1_2":(0,np.inf)}
                        
    _param_types = {"k0":np.float64,
                        "theta0":np.float64,
                        "lambda1_1":np.float64,
                        "k1_1":np.float64,
                        "theta1_1":np.float64,
                        "lambda1_2":np.float64,
                        "k1_2":np.float64,
                        "theta1_2":np.float64}
                        
    
    def __init__(self,k0=0.,theta0=1.,lambda1_1=.5,k1_1=0.,theta1_1=1.,k1_2=0.,theta1_2=1.,\
                    freeparams=["k0","theta0","lambda1_1","k1_1","theta1_1","k1_2","theta1_2"]):
    
        #initialize independent parameters
        loader = {"k0":np.float64(k0),
                    "theta0":np.float64(theta0),
                    "lambda1_1":np.float64(lambda1_1),
                    "k1_1":np.float64(k1_1),
                    "theta1_1":np.float64(theta1_1),
                    "k1_2":np.float64(k1_2),
                    "theta1_2":np.float64(theta1_2)}
        self._params = {}
        self.updateParams(loader)
        
        #initialize dependent parameter
        self._params["lambda1_2"] = 1. - self._params['lambda1_1']
        
        self._param_bounds = self._outer_bounds.copy()
        self._free_params = freeparams

    def __repr__(self):
        '''Representation for evaluation'''
        return "plum.models.ErrorModels.GammaMix({})".format( ",".join("=".join([i,str(j)]) for i,j in self.freeParamDict.items()) )
        
    def __str__(self):
        return "plum.models.ErrorModels.GammaMix"

    def P(self,data,state):
        '''Return the probability of `data` under one of two normal distributions,
        depending on `state`.'''
        if state == 0:
            p = cprobs.gammapdf(data,self._params["k0"],self._params["theta0"])
            return p
        elif state == 1:
            p = cprobs.gammaMix2pdf(data, self._params["lambda1_1"], self._params["lambda1_2"],
                                           self._params["k1_1"], self._params["k1_2"],
                                           self._params["theta1_1"], self._params["theta1_2"])
            return p
        else:
            raise Exception("State {} no recognized".format(state))

class Cauchy(Gaussian):

    '''
    Models binary latent states with two single Cauchys
    
    Parameters
        - loc0, theta0: The shape and scale parameters for state 0
        - loc1, theta1: The shape and scale parameters for state 1
    '''
    
    _param_names = ["loc0","scale0","loc1","scale1"]
        
    _outer_bounds = {"loc0":(-np.inf,np.inf),
                        "scale0":(0.,np.inf),
                        "loc1":(-np.inf,np.inf),
                        "scale1":(0.,np.inf)}
                        
    _param_types = {"loc0":np.float64,
                        "scale0":np.float64,
                        "loc1":np.float64,
                        "scale1":np.float64}
    
    def __init__(self,loc0=0.,scale0=1.,loc1=0.,scale1=1.,freeparams = ["loc0","scale0","loc1","scale1"]):
    
        loader = dict(list(zip(["loc0","scale0","loc1","scale1"],np.array([loc0,scale0,loc1,scale1],dtype=np.float64))))
        self._params = {}
        self.updateParams(loader)
        self._param_bounds = self._outer_bounds.copy()
        self._free_params = freeparams

    def __repr__(self):
        '''Representation for evaluation'''
        return "plum.models.ErrorModels.Cauchy({})".format( ",".join("=".join([i,str(j)]) for i,j in self.freeParamDict.items()) )

    def __str__(self):
        return "plum.models.ErrorModels.Cauchy"
        
    def P(self,data,state):
        '''Probability using cprobs.pyx (Cython compiled probabilities) instead of 
        scipy.stats, which is quite slow'''
        if state == 0:
            return cprobs.cauchypdf(data,self._params["loc0"],self._params["scale0"])
        elif state == 1:
            return cprobs.cauchypdf(data,self._params["loc1"],self._params["scale1"])
        else:
            raise Exception("wtf")
            
    def draw(self,state):
        raise Exception("Haven't written this function yet")
        
class CauchyGumbel(Gaussian):

    '''
    Models binary latent states with a single Cauchys for state 0 and a single Gumbel for state 1
    
    Parameters
        - loc0, theta0: The shape and scale parameters for state 0
        - loc1, theta1: The shape and scale parameters for state 1
    '''
    
    _param_names = ["loc0","scale0","loc1","scale1"]
        
    _outer_bounds = {"loc0":(-np.inf,np.inf),
                        "scale0":(0.,np.inf),
                        "loc1":(-np.inf,np.inf),
                        "scale1":(0.,np.inf)}
                        
    _param_types = {"loc0":np.float64,
                        "scale0":np.float64,
                        "loc1":np.float64,
                        "scale1":np.float64}
    
    def __init__(self,loc0=0.,scale0=1.,loc1=0.,scale1=1.,freeparams = ["loc0","scale0","loc1","scale1"]):
    
        loader = dict(list(zip(["loc0","scale0","loc1","scale1"],np.array([loc0,scale0,loc1,scale1],dtype=np.float64))))
        self._params = {}
        self.updateParams(loader)
        self._param_bounds = self._outer_bounds.copy()
        self._free_params = freeparams

    def __repr__(self):
        '''Representation for evaluation'''
        return "plum.models.ErrorModels.CauchyGumbel({})".format( ",".join("=".join([i,str(j)]) for i,j in self.freeParamDict.items()) )

    def __str__(self):
        return "plum.models.ErrorModels.CauchyGumbel"
        
    def P(self,data,state):
        '''Probability using cprobs.pyx (Cython compiled probabilities) instead of 
        scipy.stats, which is quite slow'''
        if state == 0:
            return cprobs.cauchypdf(data,self._params["loc0"],self._params["scale0"])
        elif state == 1:
            return cprobs.gumbelpdf(data,self._params["loc1"],self._params["scale1"])
        else:
            raise Exception("wtf")
            
    def draw(self,state):
        raise Exception("Haven't written this function yet")
        
class MultivariateGaussian(Gaussian):

    '''
    Models binary latent states with two single Multivariate Gaussians
    
    Parameters
        - mean0, sigma0: The mean vector and covariance matrix for state 0
        - mean1, sigma1: The mean vector and covariance matrix for state 1
        
    
    '''
    
    _param_names = ["mean0","sigma0","mean1","sigma1"]
        
    # this is awkward, because the diagonals of the covariance matrix must
    # be bounded at 0, because they're variances, but the covariances
    # on the off-diagnonals can be negative. Will have to think about this.
    _outer_bounds = {"mean0":(-np.inf,np.inf),
                        "sigma0":(-np.inf,np.inf),
                        "mean1":(-np.inf,np.inf),
                        "sigma1":(-np.inf,np.inf)}
                        
    _param_types = {"mean0":np.ndarray,
                        "sigma0":np.ndarray,
                        "mean1":np.ndarray,
                        "sigma1":np.ndarray}
    
    def __init__(self,mean0=[0,0],sigma0=[[1,0],[0,1]],mean1=[0,0],sigma1=[[1,0],[0,1]],freeparams = ["mean0","sigma0","mean1","sigma1"]):
    
        if sigma0 is 1:
            sigma0 = np.eye( len(mean0) )
        if sigma1 is 1:
            sigma1 = np.eye( len(mean1) )
        loader = dict(list(zip( ["mean0","sigma0","mean1","sigma1"], [np.array(x,dtype=np.float64) for x in [mean0,sigma0,mean1,sigma1]] )))
        self._params = {}
        self.updateParams(loader)
        self._param_bounds = self._outer_bounds.copy()
        self._free_params = freeparams
        
    def __repr__(self):
        '''Representation for evaluation'''
        return "plum.models.ErrorModels.MultivariateGaussian({})".format( ",".join("=".join([i,str(j)]) for i,j in self.freeParamDict.items()) )
    
    def __str__(self):
        return "plum.models.ErrorModels.MultivariateGaussian"
    
    @property
    def paramBounds(self):
        '''Get or set bounds of each parameter. Restricts the search space during optimization
            - Dictionary of tuples mapping strings to values'''
        return self._param_bounds
        
    @paramBounds.setter
    def paramBounds(self,newbounds):
        assert type(newbounds) is dict, "`newbounds` must be type dict"
        for p,bounds in newbounds.items():
            if type(bounds) is list:
                bounds = np.array(bounds)
                self._validateParam(p,value=None,bounds=bounds)
        self._param_bounds.update(newbounds) 
        
    @property
    def size(self):
        '''The size (length) of the feature vector'''
        return self.params["mean0"].shape[0]
    
    @property
    def is_multivariate(self):
        return True
        
    def updateParams(self,newvalues):
        '''Update with new parameters.
            `newvalues` - a dictionary mapping parameter names to new values'''
        # update independent parameters
        change_0 = False
        change_1 = False
        for name,val in newvalues.items():
            if type(val) != np.ndarray:
                val = np.array(val, dtype=np.float64)
            self._validateParam(name,value=val)
            self._params[name] = val
            if name == "sigma0":
                change_0 = True
            if name == "sigma1":
                change_1 = True

        assert self._params["mean1"].shape[0] == self._params["mean0"].shape[0], "Mean vectors must be the same length"
        assert self._params["sigma0"].shape[0] == self._params["mean0"].shape[0], "Covariance matrices must be n X n where n = length of mean vector"
        assert self._params["sigma1"].shape[0] == self._params["mean1"].shape[0], "Covariance matrices must be n X n where n = length of mean vector"
        
        if change_0 == True:
            self._0_cov_inv = np.linalg.inv(self._params["sigma0"])
            self._0_cov_det = np.linalg.det(self._params["sigma0"])
        if change_1 == True:
            self._1_cov_inv = np.linalg.inv(self._params["sigma1"])
            self._1_cov_det = np.linalg.det(self._params["sigma1"])
     
    def _validateParam(self,param,value=None,bounds=None):
        '''Check relevant properties of parameter(s), such as allowable bounds.
        This function should be called by all other methods that change or set
        the parameters. It should be a private function so it can't be tinkered with.'''
        # Check strings
        assert param in self._param_names, "'{}' not a valid parameter name".format(param)
        # Check value type and bound
        if value is not None:
            assert type(value) is self._param_types[param], \
                "Parameter '{}' must be {}, not {}".format(param,self._param_types[param], type(value))
            assert np.all(value > self._outer_bounds[param][0]) & np.all(value < self._outer_bounds[param][1]), \
                "'{0}' value {1} is out of acceptable range: {2}".format(
                    param,value,self._outer_bounds[param])
                    
            if "sigma" in param:
                assert np.allclose(value, value.T), "Covariance matrix must be symmetric: {}".format(value)
                assert np.all(value.diagonal() >= 0.), "Covariance matrix diagonals must be greater than or equal to 0: '{}'".format(param)
                assert np.linalg.det(value) != 0., "Covariance matrix '{}' is singular".format(param)
                assert np.all(np.linalg.eigvalsh(value) >= 0), "Covariance matrix '{}' is not positive semidefinite".format(param)
                assert value.shape[0] == value.shape[1], "Covariance matrices must be square"
            else:
                assert value.shape[0] > 1, "Mean vectors must be length 2 or greater"
                assert value.ndim == 1, "Mean vectors must be one-dimensional: {}".format(self._params)
                
                
        # Check bounds
        if bounds is not None:
            assert len(bounds) == 2, "Bounds must be a 2 tuple"
            assert (bounds[0] >= self._outer_bounds[param][0]) & (bounds[1] <= self._outer_bounds[param][1]), \
            "New bounds for parameter {0} is out of range: {1}\n Acceptable range is:{2}".format(
                param,bounds,self._outer_bounds[param])
        
    ## **Note** there is currently nothing checking that the input data is the right shape
    ## Except that it will throw an error if it's not a list. Not sure what to do about it 
    ## right now, as putting too much error checking will probably slow things down considerably.
    def P(self,data,state):
        '''Probability using cprobs.pyx (Cython compiled probabilities) instead of 
        scipy.stats, which is quite slow'''
        if state == 0:
            return cprobs.mvnormpdf(data, self._params["mean0"], self._params["sigma0"],
                                        self._0_cov_inv, self._0_cov_det)
        elif state == 1:
            return cprobs.mvnormpdf(data, self._params["mean1"], self._params["sigma1"],
                                        self._1_cov_inv, self._1_cov_det)
        else:
            raise Exception("wtf")
            
    def draw(self,state):
        '''Draw random numbers from one of the two error models for simulation.
        `state`: Determines which state's error model to draw from'''
        if state == 0:
            return np.random.multivariate_normal(self._params["mean0"],self._params["sigma0"])
        elif state == 1:
            return np.random.multivariate_normal(self._params["mean1"],self._params["sigma1"])
        else:
            raise Exception("wtf")
            
class MultivariateGaussianMixture(MultivariateGaussian):

    '''
    Models binary latent states with a single Multivariate Gaussians for
    the 0 state and a mixture of 2 Multivariate Gaussians for the 1 state.
    
    Free Parameters
        - mean0, sigma0: The mean vector and covariance matrix for state 0
        - mean1, sigma1: The mean vector and covariance matrix for state 1
        - lambda1_1: The mixture weight for the lower component
        
    
    '''
    
    _independent_params = ["mean0","sigma0","lambda1_1","mean1_1","sigma1_1","mean1_2","sigma1_2"]
    _dependent_params = ["lambda1_2"]
    _param_names = _independent_params + _dependent_params
        
    # this is awkward, because the diagonals of the covariance matrix must
    # be bounded at 0, because they're variances, but the covariances
    # on the off-diagnonals can be negative. Will have to think about this.
    _outer_bounds = {"mean0":(-np.inf,np.inf),
                        "sigma0":(-np.inf,np.inf),
                        "lambda1_1": (0.,1.),
                        "mean1_1":(-np.inf,np.inf),
                        "sigma1_1":(-np.inf,np.inf),
                        "mean1_2":(-np.inf,np.inf),
                        "sigma1_2":(-np.inf,np.inf)}
                        
    _param_types = {"mean0":np.ndarray,
                        "sigma0":np.ndarray,
                        "lambda1_1": np.float64,
                        "mean1_1":np.ndarray,
                        "sigma1_1":np.ndarray,
                        "mean1_2":np.ndarray,
                        "sigma1_2":np.ndarray}
    
    def __init__(self,mean0=[0,0],sigma0=1,mean1_1=[0,0],sigma1_1=1,mean1_2=[0,0],sigma1_2=1,lambda1_1=.5,
                freeparams = ["mean0","sigma0","lambda1_1","mean1_1","sigma1_1","mean1_2","sigma1_2"]):
         
        if sigma0 is 1:
            sigma0 = np.eye(len(mean0))
        if sigma1_1 is 1:
            sigma1_1 = np.eye(len(mean0))
        if sigma1_2 is 1:
            sigma1_2 = np.eye(len(mean0))
            
        loader = dict(list(zip( ["mean0","sigma0","mean1_1","sigma1_1","mean1_2","sigma1_2"], 
                            [np.array(x,dtype=np.float64) for x in [mean0,sigma0,mean1_1,sigma1_1,mean1_2,sigma1_2]] )))
        loader["lambda1_1"] = np.float64(lambda1_1)
        self._params = {}
        self.updateParams(loader)
        self._param_bounds = self._outer_bounds.copy()
        self._free_params = freeparams
        
    def __repr__(self):
        '''Representation for evaluation'''
        return "plum.models.ErrorModels.MultivariateGaussianMixture({})".format( ",".join("=".join([i,str(j)]) for i,j in self.freeParamDict.items()) )
    
    def __str__(self):
        return "plum.models.ErrorModels.MultivariateGaussianMixture"
        
    def updateParams(self,newvalues):
        '''Update with new parameters.
            `newvalues` - a dictionary mapping parameter names to new values'''
        # update independent parameters
        change_0 = False
        change_1_1 = False
        change_1_2 = False
        for name,val in newvalues.items():
            if type(val) is list:
                val = np.array(val, dtype=np.float64)
            else:
                val = np.float64(val)
            self._validateParam(name,value=val)
            self._params[name] = val
            if name == "sigma0":
                change_0 = True
            if name == "sigma1_1":
                change_1_1 = True
            if name == "sigma1_2":
                change_1_2 = True
            if name == 'lambda1_1':
                self._params['lambda1_2'] = 1. - self._params['lambda1_1'] # update dependent parameter 

        assert self._params["mean1_1"].shape[0] == self._params["mean1_2"].shape[0] == self._params["mean0"].shape[0], \
                "Mean vectors must be the same length"
        assert self._params["sigma0"].shape[0] == self._params["mean0"].shape[0], "Covariance matrices must be n X n where n = length of mean vector"
        assert self._params["sigma1_1"].shape[0] == self._params["mean1_1"].shape[0], "Covariance matrices must be n X n where n = length of mean vector"
        assert self._params["sigma1_2"].shape[0] == self._params["mean1_2"].shape[0], "Covariance matrices must be n X n where n = length of mean vector"
        
        # Pre-calculate linear algebra for multivariate pdf.
        if change_0 == True:
            self._0_cov_inv = np.linalg.inv(self._params["sigma0"])
            self._0_cov_det = np.linalg.det(self._params["sigma0"])
        if change_1_1 == True:
            self._1_1_cov_inv = np.linalg.inv(self._params["sigma1_1"])
            self._1_1_cov_det = np.linalg.det(self._params["sigma1_1"])
        if change_1_2 == True:
            self._1_2_cov_inv = np.linalg.inv(self._params["sigma1_2"])
            self._1_2_cov_det = np.linalg.det(self._params["sigma1_2"])
     
    def _validateParam(self,param,value=None,bounds=None):
        '''Check relevant properties of parameter(s), such as allowable bounds.
        This function should be called by all other methods that change or set
        the parameters. It should be a private function so it can't be tinkered with.'''
        # Check strings
        assert param in self._independent_params, "'{}' not a valid free parameter name".format(param)
        # Check value type and bound
        if value is not None:
            assert type(value) is self._param_types[param], \
                "Parameter '{}' must be {}, not {}".format(param,self._param_types[param], type(value))
            assert np.all(value > self._outer_bounds[param][0]) & np.all(value < self._outer_bounds[param][1]), \
                "'{0}' value {1} is out of acceptable range: {2}".format(
                    param,value,self._outer_bounds[param])
                    
            if "sigma" in param:
                assert np.allclose(value, value.T), "Covariance matrix must be symmetric: {}".format(value)
                assert np.all(value.diagonal() >= 0.), "Covariance matrix diagonals must be greater than or equal to 0: '{}'".format(param)
                assert np.linalg.det(value) != 0., "Covariance matrix '{}' is singular".format(param)
                assert np.all(np.linalg.eigvalsh(value) >= 0), "Covariance matrix '{}' is not positive semidefinite".format(param)
                assert value.shape[0] == value.shape[1], "Covariance matrices must be square"
            elif "mean" in param:
                assert value.shape[0] > 1, "Mean vectors must be length 2 or greater"
                assert value.ndim == 1, "Mean vectors must be one-dimensional: {}".format(self._params)
            else:
                pass
                
                
        # Check bounds
        if bounds is not None:
            assert len(bounds) == 2, "Bounds must be a 2 tuple"
            assert (bounds[0] >= self._outer_bounds[param][0]) & (bounds[1] <= self._outer_bounds[param][1]), \
            "New bounds for parameter {0} is out of range: {1}\n Acceptable range is:{2}".format(
                param,bounds,self._outer_bounds[param])
        
    ## **Note** there is currently nothing checking that the input data is the right shape
    ## Except that it will throw an error if it's not a list. Not sure what to do about it 
    ## right now, as putting too much error checking will probably slow things down considerably.
    def P(self,data,state):
        '''Probability using cprobs.pyx (Cython compiled probabilities) instead of 
        scipy.stats, which is quite slow'''
        if state == 0:
            return cprobs.mvnormpdf(data, self._params["mean0"], self._params["sigma0"],
                                        self._0_cov_inv, self._0_cov_det)
        elif state == 1:
            return self._params["lambda1_1"] * \
                                        cprobs.mvnormpdf(data, self._params["mean1_1"], self._params["sigma1_1"],self._1_1_cov_inv, self._1_1_cov_det) + \
                                        self._params["lambda1_2"] * \
                                        cprobs.mvnormpdf(data, self._params["mean1_2"], self._params["sigma1_2"],self._1_2_cov_inv, self._1_2_cov_det)
        else:
            raise Exception("wtf")
            
    def draw(self,state):
        '''Draw random numbers from one of the two error models for simulation.
        `state`: Determines which state's error model to draw from'''
        if state == 0:
            return np.random.multivariate_normal(self._params["mean0"],self._params["sigma0"])
        elif state == 1:
            component = np.random.choice( [0,1], p=[self._params["lambda1_1"], self._params["lambda1_2"]] )
            if component == 0:
                return np.random.multivariate_normal(self._params["mean1_1"],self._params["sigma1_1"])
            elif component == 1:
                return np.random.multivariate_normal(self._params["mean1_2"],self._params["sigma1_2"])
            else:
                raise
        else:
            raise Exception("wtf")
            
class MultivariateGaussianDiagCov(MultivariateGaussian):

    '''
    Models binary latent states with two single Multivariate Gaussians
    with diagonal covariance matrices (off-diagonals = 0)
    
    Independent parameters
        - mean0, sigmadiag0: The mean vector and diagonal of covariance matrix for state 0
        - mean1, sigmadiag1: The mean vector and diagonal covariance matrix for state 1
    '''
    
    _independent_params = ["mean0","sigmadiag0","mean1","sigmadiag1"]
    _dependent_params = ["sigma0","sigma1"]
    _param_names = _independent_params + _dependent_params
        
    # this is awkward, because the diagonals of the covariance matrix must
    # be bounded at 0, because they're variances, but the covariances
    # on the off-diagnonals can be negative. Will have to think about this.
    _outer_bounds = {"mean0":(-np.inf,np.inf),
                        "sigmadiag0":(0.,np.inf),
                        "mean1":(-np.inf,np.inf),
                        "sigmadiag1":(0.,np.inf)}
                        
    _param_types = {"mean0":np.ndarray,
                        "sigmadiag0":np.ndarray,
                        "mean1":np.ndarray,
                        "sigmadiag1":np.ndarray}
                        
    @property
    def independentParams(self):
        '''The names of the independent parameters'''
        return self._independent_params
    
    def __init__(self,mean0=[0,0],sigmadiag0=[1,1],mean1=[0,0],sigmadiag1=[1,1],freeparams = ["mean0","sigmadiag0","mean1","sigmadiag1"]):
    
        if sigmadiag0 is 1:
            sigmadiag0 = np.ones( len(mean0) )
        if sigmadiag1 is 1:
            sigmadiag1 = np.ones( len(mean1) )
        loader = dict(list(zip( ["mean0","sigmadiag0","mean1","sigmadiag1"], [np.array(x,dtype=np.float64) for x in [mean0,sigmadiag0,mean1,sigmadiag1]] )))
        self._params = { "sigma0": np.eye( len(mean0) ), "sigma1": np.eye( len(mean1) ) }
        self.updateParams(loader)
        self._param_bounds = self._outer_bounds.copy()
        self._free_params = freeparams
        
    def __repr__(self):
        '''Representation for evaluation'''
        return "plum.models.ErrorModels.MultivariateGaussianDiagCov({})".format( ",".join("=".join([i,str(j)]) for i,j in self.freeParamDict.items()) )
    
    def __str__(self):
        return "plum.models.ErrorModels.MultivariateGaussianDiagCov"
        
    def updateParams(self,newvalues):
        '''Update with new parameters.
            `newvalues` - a dictionary mapping parameter names to new values'''
        # update independent parameters
        change_0 = False
        change_1 = False
        for name,val in newvalues.items():
            if type(val) != np.ndarray:
                val = np.array(val, dtype=np.float64)
            self._validateParam(name,value=val)
            self._params[name] = val
            
            # update dependent parameter: covariance matrix
            if "sigma" in name:
                if "0" in name:
                    change_0 = True
                if "1" in name:
                    change_1 = True
                state = name[-1]
                np.fill_diagonal( self._params["sigma" + state], val )
                _cov = self._params["sigma" + state]
                assert np.allclose(_cov, _cov.T), "Covariance matrix must be symmetric: {}".format(_cov)
                assert np.all(_cov.diagonal() >= 0.), "Covariance matrix diagonals must be greater than or equal to 0: '{}'".format(_cov)
                assert np.linalg.det(_cov) != 0., "Covariance matrix '{}' is singular".format(_cov)
                assert np.all(np.linalg.eigvalsh(_cov) >= 0), "Covariance matrix '{}' is not positive semidefinite".format(_cov)
                assert _cov.shape[0] == _cov.shape[1], "Covariance matrices must be square"
        
        assert self._params["sigmadiag0"].shape[0] == self._params["mean0"].shape[0] == \
        self._params["sigmadiag1"].shape[0] == self._params["mean1"].shape[0], "Covariance matrix diagonals and mean vectors must all be the same shape"
        
        if change_0 == True:
            self._0_cov_inv = np.linalg.inv(self._params["sigma0"])
            self._0_cov_det = np.linalg.det(self._params["sigma0"])
        if change_1 == True:
            self._1_cov_inv = np.linalg.inv(self._params["sigma1"])
            self._1_cov_det = np.linalg.det(self._params["sigma1"])

     
    def _validateParam(self,param,value=None,bounds=None):
        '''Check relevant properties of parameter(s), such as allowable bounds.
        This function should be called by all other methods that change or set
        the parameters. It should be a private function so it can't be tinkered with.'''
        # Check strings
        assert param in self._independent_params, "'{}' not a valid independent parameter name".format(param)
        # Check value type and bound
        if value is not None:
            assert type(value) is self._param_types[param], \
                "Parameter '{}' must be {}, not {}".format(param,self._param_types[param], type(value))
            assert np.all(value > self._outer_bounds[param][0]) & np.all(value < self._outer_bounds[param][1]), \
                "'{0}' value {1} is out of acceptable range: {2}".format(
                    param,value,self._outer_bounds[param])
            assert value.shape[0] > 1, "Parameter vectors must be length 2 or greater: {}".format(value)
            assert value.ndim == 1, "Parameter vectors must be one-dimensional: {}".format(value)
            
        # Check bounds
        if bounds is not None:
            assert len(bounds) == 2, "Bounds must be a 2 tuple"
            assert (bounds[0] >= self._outer_bounds[param][0]) & (bounds[1] <= self._outer_bounds[param][1]), \
            "New bounds for parameter {0} is out of range: {1}\n Acceptable range is:{2}".format(
                param,bounds,self._outer_bounds[param])
        
    ## **Note** there is currently nothing checking that the input data is the right shape
    ## Except that it will throw an error if it's not a list. Not sure what to do about it 
    ## right now, as putting too much error checking will probably slow things down considerably.
    def P(self,data,state):
        '''Probability using cprobs.pyx (Cython compiled probabilities) instead of 
        scipy.stats, which is quite slow'''
        if state == 0:
            return cprobs.mvnormpdf(data, self._params["mean0"], self._params["sigma0"],
                                        self._0_cov_inv, self._0_cov_det)
        elif state == 1:
            return cprobs.mvnormpdf(data, self._params["mean1"], self._params["sigma1"],
                                        self._1_cov_inv, self._1_cov_det)
        else:
            raise Exception("wtf")
            
    def draw(self,state):
        '''Draw random numbers from one of the two error models for simulation.
        `state`: Determines which state's error model to draw from'''
        if state == 0:
            return np.random.multivariate_normal(self._params["mean0"],self._params["sigma0"])
        elif state == 1:
            return np.random.multivariate_normal(self._params["mean1"],self._params["sigma1"])
        else:
            raise Exception("wtf")