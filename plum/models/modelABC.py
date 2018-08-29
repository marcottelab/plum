import abc

class Model(object):
    '''
    Base class for two-state Markov models and error models.
    '''
    
    __metaclass__ = abc.ABCMeta
    
    ## Class attributes ##
    
    # Model should define class attributes that set the parameter
    # names, the possible states of the model, the types, and 
    # the acceptable bounds. We can't enforce abstract attributes,
    # so they are commented out.
    
       
    #_states = []
    #_param_names = []
    #_param_types = {}   
    #_outer_bounds = {}
  
    ## Properties ##
    
    # Model object should allow the user to contrain likelihood searches etc.
    # by changing the free parameters and by setting parameter bounds, so here
    # we define property functions that will be overridden by appropriate getter
    # and setter functions that do validations etc.
    
    @abc.abstractproperty
    def params(self):
        pass
    
    @abc.abstractproperty
    def freeParams(self):
        '''The parameters that will be optimized.
            - A list of strings'''
        pass
        
    @abc.abstractproperty
    def paramBounds(self):
        '''The bounds of each parameter.
            - A dictionary of tuples'''
        pass
    
    ## Methods ##
    
    # Need methods to update, view, and validate parameters
    
    @abc.abstractmethod
    def updateParams(self, newparams):
        '''Method for updating parameters optimization. All updates,
        edits, instantiations, etc., must go through this function, which
        will in turn call _validateParam. That way all parameters are
        validated before they are changed.'''
        pass
    
    @abc.abstractmethod
    def _validateParam(self):
        '''Check relevant properties of parameter(s), such as allowable bounds.
        This function should be called by all other methods that change or set
        the parameters. It should be a private function so it can't be tinkered with.'''
        pass
        
    @abc.abstractmethod
    def P(self):
        '''Model class must implement a method to return the probability of a 
        single observation given some parameters.'''
        pass