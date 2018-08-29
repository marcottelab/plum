# cython: linetrace=True

import copy
cimport cython
import numpy as np
cimport numpy as np
from cpython cimport bool
from libc.math cimport exp

## Functions for fitting models

cdef c_simulated_annealing(object function, np.ndarray[np.float64_t,ndim=1] param_vec, tuple bounds, double start_temp=1, double alpha=9, int temp_steps=10, double mutation_sd=.5):
    '''Fit model by simulated annealing
     - `function`: Something that maximizes something
     - `param_vec`: np.ndarray of initial parameter values
     - `bounds`: Exclusive boundaries for picking new parameter values. tuple of tuples
     - `start_temp`: Starting temperature for annealing. Defaults to 1
     - `alpha`: Proportion to decrement T by for each round
     - `temp_steps`: Number of steps to take within each temperature iteration
    '''
    
    cdef:
        Py_ssize_t s # step in iterations
        int rand_index # random number for choosing parameters
        unsigned int n_accepts = 0
        float n_rejects = 0.
        double accept_rate = 0.
        double new_val # new value of `function` for each permutation
        double new_param # new parameter value
        double T = start_temp # Tempterature parameter
        double tmin = .000001 # Ending temperature - hard coded
        double current_val = function(param_vec) # the current accepted value of `function` APS
        double best = current_val # best AUC found so far
        unsigned int nparams = len(param_vec) # number of parameters
        np.ndarray[np.float64_t,ndim=1] new_param_vec
        np.ndarray[np.float64_t,ndim=1] current_params = copy.deepcopy(param_vec) # stores current parameters
        np.ndarray[np.float64_t,ndim=1] best_params = np.zeros(param_vec.shape[0])# best set of parameters
    
    randint = np.random.randint
    normal = np.random.normal
    sample = np.random.sample
    
    while T > tmin:
        n_accepts, n_rejects = 0 , 0.
        for s in xrange(temp_steps):
            new_param_vec = copy.deepcopy(current_params)
            rand_index = randint(low=0,high=nparams)
            new_param = normal( loc=new_param_vec[rand_index],scale=mutation_sd )
            
            # If new draw is out of bounds, keep drawing until new parameter is in bounds
            if np.logical_and( new_param > bounds[rand_index][0], new_param < bounds[rand_index][1]) == False:
                while np.logical_and( new_param > bounds[rand_index][0], new_param < bounds[rand_index][1]) == False:
                    new_param = normal( loc=new_param_vec[rand_index],scale=mutation_sd )
            
            new_param_vec[rand_index] =  new_param
            new_val = function(new_param_vec) ## call to AUC

            if np.isnan(new_val):
                n_rejects += 1
                continue
            
            elif new_val > current_val:
                n_accepts += 1
                current_val = new_val
                current_params = copy.deepcopy( new_param_vec )
                if new_val > best:
                    best = new_val
                    best_params = copy.deepcopy( new_param_vec )
            else:
                accept_P = exp( (new_val - current_val) / T )
                if accept_P > sample():
                    n_accepts += 1
                    current_val = new_val
                    current_params = copy.deepcopy( new_param_vec )
                    if new_val > best:
                        best = new_val
                        best_params = copy.deepcopy( new_param_vec )
                else:
                    n_rejects += 1
        T = T * alpha
        accept_rate = n_accepts / (n_accepts + n_rejects)
        print "\t".join( ["Current temp: {}".format(T), "Best score: {}".format(best), "Accept rate at this T: {}".format(accept_rate)] )
    return best_params, best
    
cdef c_simulated_annealing_multivariate(object function, np.ndarray[object,ndim=1] param_vec, tuple bounds, double start_temp=1, double alpha=9, int temp_steps=10, double mutation_sd=.5):
    '''Fit multivariate model by simulated annealing
     - `function`: Something that maximizes something
     - `param_vec`: np.ndarray of initial parameter values
     - `bounds`: Exclusive boundaries for picking new parameter values. tuple of tuples
     - `start_temp`: Starting temperature for annealing. Defaults to 1
     - `alpha`: Proportion to decrement T by for each round
     - `temp_steps`: Number of steps to take within each temperature iteration
    '''
    
    cdef:
        Py_ssize_t s # step in iterations
        bool condition
        int rand_index # random numbers for choosing parameters from vec
        unsigned int n_accepts = 0
        tuple rand_index_2 # index into multivariate params
        float n_rejects = 0.
        double accept_rate = 0.
        double new_val, mutated_param_val # new value of `function` for each permutation
        double T = start_temp # Tempterature parameter
        double tmin = .000001 # Ending temperature - hard coded
        double current_val = function(param_vec) # the current accepted value of `function` APS
        double best = current_val # best AUC found so far
        object new_param, eigvals# new parameter, could be an array or a float
        unsigned int nparams = len(param_vec) # number of parameters 
        np.ndarray[object,ndim=1] new_param_vec
        np.ndarray[object,ndim=1] current_params = copy.deepcopy(param_vec) # stores current parameters
        np.ndarray[object,ndim=1] best_params = copy.deepcopy(param_vec)# best set of parameters
    
    randint = np.random.randint
    normal = np.random.normal
    sample = np.random.sample
    linAlgError = np.linalg.LinAlgError
    
    while T > tmin:
        n_accepts, n_rejects = 0 , 0.
        for s in xrange(temp_steps):
            # Choose which mutivariate parameter and which parameter index to mutate
            new_param_vec = copy.deepcopy(current_params)
            rand_index = randint(low=0,high=nparams)
            new_param = new_param_vec[rand_index]
            
            if type(new_param) is np.ndarray: # one of the error_model parameters
                # call to sorted makes sure index is in bottom triangle of covariance matrix - doesn't affect mean vectors
                rand_index_2 = tuple( sorted(randint(new_param.shape[0], size=new_param.ndim), reverse=True) )
                mutated_param_val = normal( loc=new_param[ rand_index_2 ],scale=mutation_sd )
                new_param[rand_index_2] = mutated_param_val
            
                if new_param.ndim == 2: # is covariance matrix, do extra checks
                    new_param[ tuple(reversed(rand_index_2)) ] = mutated_param_val # make symmetric
                    eigvals = np.linalg.eigvalsh(new_param)
                    condition = bool( np.logical_and.reduce( [np.all(new_param > bounds[rand_index][0]), 
                                    np.all(new_param < bounds[rand_index][1]),
                                    np.all(new_param.diagonal() >= 0.),
                                    np.linalg.det(new_param) != 0.,
                                    np.all( eigvals > 0 ),
                                    np.all(np.isreal(eigvals))] ) )
                    if condition == False:
                        count = 0
                        print "Looking for appropriate covariance matrix"
                        while condition == False:
                            count += 1
                            if count > 10000:
                                raise Exception("Reached 10000 attempts to find a decent covariance matrix: {}".format(new_param))
                            if count % 100 == 0:
                                print "...still looking for appropriate covariance matrix {}".format(new_param)
                            
                            new_param = copy.deepcopy(current_params[rand_index]) # reset param
                            mutated_param_val = normal( loc=new_param[ rand_index_2 ], scale= mutation_sd )
                            new_param[rand_index_2] = mutated_param_val
                            new_param[ tuple(reversed(rand_index_2)) ] = mutated_param_val # make symmetric
                            eigvals = np.linalg.eigvalsh(new_param)
                            condition = bool( np.logical_and.reduce( [np.all(new_param > bounds[rand_index][0]), 
                                            np.all(new_param < bounds[rand_index][1]),
                                            np.all(new_param.diagonal() >= 0.),
                                            np.linalg.det(new_param) != 0.,
                                            np.all(eigvals > 0),
                                            np.all(np.isreal(eigvals))] ) )
                            
                else:
                    if np.logical_and( np.all(new_param > bounds[rand_index][0]), np.all(new_param < bounds[rand_index][1])) == False:
                        count = 0
                        while np.logical_and( np.all(new_param > bounds[rand_index][0]), np.all(new_param < bounds[rand_index][1])) == False:
                            count += 1
                            if count > 10000:
                                raise Exception("Reached 10000 attempts to find a decent new parameter: {}, index {}".format(new_param,rand_index))
                            new_param = copy.deepcopy(current_params[rand_index]) # reset param
                            mutated_param_val = normal( loc=new_param[ rand_index_2 ],scale=mutation_sd )
                            new_param[rand_index_2] = mutated_param_val
                            
            else:
                new_param = normal( loc=new_param_vec[rand_index],scale=mutation_sd )
                
                # If new draw is out of bounds, keep drawing until new parameter is in bounds
                if np.logical_and( new_param > bounds[rand_index][0], new_param < bounds[rand_index][1]) == False:
                    count = 0
                    while np.logical_and( new_param > bounds[rand_index][0], new_param < bounds[rand_index][1]) == False:
                        count += 1
                        if count > 10000:
                            raise Exception("Reached 10000 attempts to find a decent new parameter: {}".format(new_param))
                        new_param = normal( loc=new_param_vec[rand_index],scale=mutation_sd )
                
            
            new_param_vec[rand_index] = new_param
            try:
                new_val = function(new_param_vec) ## call to AUC
            except linAlgError:
                raise Exception("{}".format(new_param_vec))

            if np.isnan(new_val):
                n_rejects += 1
                continue
            
            elif new_val > current_val:
                n_accepts += 1
                current_val = new_val
                current_params = copy.deepcopy( new_param_vec )
                if new_val > best:
                    best = new_val
                    best_params = copy.deepcopy( new_param_vec )
            else:
                accept_P = exp( (new_val - current_val) / T )
                if accept_P > sample():
                    n_accepts += 1
                    current_val = new_val
                    current_params = copy.deepcopy( new_param_vec )
                    if new_val > best:
                        best = new_val
                        best_params = copy.deepcopy( new_param_vec )
                else:
                    n_rejects += 1
        T = T * alpha
        accept_rate = n_accepts / (n_accepts + n_rejects)
        print "\t".join( ["Current temp: {}".format(T), "Best score: {}".format(best), "Accept rate at this T: {}".format(accept_rate)] )
    return best_params, best
    
def _simulated_annealing(function, param_vec, bounds, start_temp=1, alpha=.9, temp_steps=10, mutation_sd=.5):
    '''Fit model by simulated annealing
     - `function`: Something that maximizes something
     - `param_vec`: np.ndarray of initial parameter values
     - `bounds`: Exclusive boundaries for picking new parameter values. tuple of tuples
     - `start_temp`: Starting temperature for annealing. Defaults to 1
     - `alpha`: Proportion to decrement T by for each round
     - `temp_steps`: Number of steps to take within each temperature iteration
    '''
    assert len(param_vec) > 0, "`param_vec` must not be empty"
    assert len(bounds) == len(param_vec), "Bounds and param_vec must be same length"
    return c_simulated_annealing(function, param_vec, bounds, start_temp, alpha, temp_steps, mutation_sd)
    
def _simulated_annealing_multivariate(function, param_vec, bounds, start_temp=1, alpha=.9, temp_steps=10, mutation_sd=.5):
    '''Fit multivariate model by simulated annealing
     - `function`: Something that maximizes something
     - `param_vec`: np.ndarray of initial parameter values
     - `bounds`: Exclusive boundaries for picking new parameter values. tuple of tuples
     - `start_temp`: Starting temperature for annealing. Defaults to 1
     - `alpha`: Proportion to decrement T by for each round
     - `temp_steps`: Number of steps to take within each temperature iteration
    '''
    assert len(param_vec) > 0, "`param_vec` must not be empty"
    assert len(bounds) == len(param_vec), "Bounds and param_vec must be same length"
    return c_simulated_annealing_multivariate(function, param_vec, bounds, start_temp, alpha, temp_steps, mutation_sd)
    
def sim_anneal_test(function, param_vec, bounds, start_temp=1, alpha=.9, temp_steps=10):
    
    T = start_temp
    tmin = .000001
    best = -np.inf
    current_val = function(param_vec)
    nparams = len(param_vec)
    params = param_vec
    current_params = param_vec
    best_params = np.zeros(param_vec.shape[0])
    mut_sd = .5
    
    arange = np.arange
    randint = np.random.randint
    normal = np.random.normal
    sample = np.random.sample
    
    while T > tmin:
        for s in arange(temp_steps):
            new_param_vec = copy.deepcopy(current_params)
            rand_index = randint(low=0,high=nparams)
            new_param = normal( loc=new_param_vec[rand_index],scale=mut_sd )
            
            # If new draw is out of bounds, keep drawing until new parameter is in bounds
            if np.logical_and( new_param > bounds[rand_index][0], new_param < bounds[rand_index][1]) == False:
                while np.logical_and( new_param > bounds[rand_index][0], new_param < bounds[rand_index][1]) == False:
                    new_param = normal( loc=new_param_vec[rand_index],scale=mut_sd )
            
            new_param_vec[rand_index] =  new_param
            new_val = function(new_param_vec) ## call to AUC

            if np.isnan(new_val):
                continue
            
            elif new_val > current_val:
                current_val = new_val
                current_params = new_param_vec
                if new_val > best:
                    best = new_val
                    best_params = new_param_vec
            else:
                accept_P = exp( (new_val - current_val) / T )
                if accept_P > sample():
                    current_val = new_val
                    current_params = new_param_vec
                    if new_val > best:
                        best = new_val
        T = T * alpha
            
    return best_params, best