cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
import scipy.linalg
from libc.math cimport sqrt, exp, pi, tgamma

ctypedef np.float_t DTYPE_t

## Numpy functions
dot = np.dot
inv = np.linalg.inv
det = np.linalg.det

## linalg funtions
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray _np_inversion(np.ndarray[DTYPE_t, ndim=2] matrix):
    return inv(matrix)
    
## Cython Probability functions

# Gaussian
    
@cython.boundscheck(False)
@cython.wraparound(False) 
cdef double c_normpdf(double x, double mu, double sigma) except -1:
    '''Cython - Gaussian probability density function. Return value for single data point'''
    cdef double s
    s = (x - mu) ** 2 / (2 * (sigma **2))
    return exp(-s) / (sqrt(2 * pi) * sigma)
    
@cython.boundscheck(False)
@cython.wraparound(False) 
cdef double c_norm_mix2_pdf(double x, double l1, double l2, double m1, double m2, double s1, double s2) except -1:
    '''Cython - Probability density function of a mixture of two normals'''
    return (l1 * c_normpdf(x,m1,s1)) + (l2 * c_normpdf(x,m2,s2))

    
@cython.boundscheck(False)
@cython.wraparound(False) 
cdef double c_mvnorm_pdf(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] mu, 
            np.ndarray[DTYPE_t, ndim=2] sigma, 
            np.ndarray[DTYPE_t, ndim=2] invsigma,
            double detsigma) except -1:
    
    cdef double smd
    cdef np.ndarray[DTYPE_t, ndim=1] diff
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] notnan = ~np.isnan(x)
    cdef tuple covnotnan
    
    if np.sum(notnan) == 0:
        return 1
    if notnan.all() == True:
        pass
    else:
        covnotnan = np.ix_(notnan,notnan)
        sigma = sigma[ covnotnan ]
        detsigma = det(sigma)
        invsigma = inv(sigma)
        x = x[notnan]
        mu = mu[notnan]
    
    diff = x - mu
    smd = dot(dot(diff, invsigma), diff) # squared Mahalonobis
    return exp( - (smd)/2 ) / sqrt( ((2 * pi) ** len(x)) * detsigma )

# Gumbel
    
@cython.boundscheck(False)
@cython.wraparound(False) 
cdef double c_gumbelpdf(double x, double mu, double scale) except -1:
    '''Cython - Gumbel probability density function. Return value for single data point'''
    cdef double z
    z = (x - mu) / scale
    return exp(-(z+exp(-z))) / scale
    
@cython.boundscheck(False)
@cython.wraparound(False) 
cdef double c_gumbel_mix2_pdf(double x, double l1, double l2, double m1, double m2, double s1, double s2) except -1:
    return (l1 * c_gumbelpdf(x,m1,s2)) + (l2 * c_gumbelpdf(x,m2,s2))
    
# Gamma    
    
@cython.boundscheck(False)
@cython.wraparound(False)  
cdef double c_gammapdf(double x, double k, double theta) except -1:
    cdef double num, denom
    num = (x**(k - 1)) * exp(-(x/theta))
    denom = (theta ** k) * tgamma(k)
    return  num / denom
    
@cython.boundscheck(False)
@cython.wraparound(False) 
cdef double c_gamma_mix2_pdf(double x, double l1, double l2, double k1, double k2, double t1, double t2) except -1:
    return (l1 * c_gammapdf(x,k1,t1)) + (l1 * c_gammapdf(x,k2,t2))
    
# Cauchy    
    
@cython.boundscheck(False)
@cython.wraparound(False)  
cdef double c_cauchypdf(double x, double loc, double scale):
    '''Cython -  Caucy probability density function for a single data point'''
    cdef double a, d, gamsq
    gamsq = scale**2
    a = 1. / (pi * scale)
    d = ( (x - loc)**2 ) + gamsq
    return a * ( gamsq / d )
    
## Python thin wrappers for Cython prob functions
@cython.boundscheck(False)
@cython.wraparound(False)
def np_matrix_inversion(matrix):
    '''Invert a Hermitian positive-definite matrix'''
    return _np_inversion(matrix)

@cython.boundscheck(False)
@cython.wraparound(False)
def normpdf(x, mu, sigma):
    '''Return the likelihood for a single data point under a normal distribution.
    `x` - data point
    `mu` - mean of normal distribution
    `sigma` - standard deviation of normal distribution.
    '''
    try:
        return c_normpdf(x,mu,sigma)
    except ZeroDivisionError:
        return 0.

@cython.boundscheck(False)
@cython.wraparound(False)     
def mvnormpdf(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] mu, 
                np.ndarray[DTYPE_t, ndim=2] sigma, 
                np.ndarray[DTYPE_t, ndim=2] invsigma,
                double detsigma):
                
    return c_mvnorm_pdf(x,mu,sigma,invsigma,detsigma)

@cython.boundscheck(False)
@cython.wraparound(False)
def normMix2pdf(x, lambda1, lambda2, mu1, mu2, sd1, sd2):
    return c_norm_mix2_pdf(x,lambda1,lambda2,mu1,mu2,sd1,sd2)

@cython.boundscheck(False)
@cython.wraparound(False) 
def gumbelpdf(x, mu, scale):
    return c_gumbelpdf(x,mu,scale)

@cython.boundscheck(False)
@cython.wraparound(False)
def gumbelMix2pdf(x, lambda1, lambda2, mu1, mu2, sd1, sd2):
    return c_gumbel_mix2_pdf(x,lambda1,lambda2,mu1,mu2,sd1,sd2)

@cython.boundscheck(False)
@cython.wraparound(False)
def gammapdf(x, k, theta):
    return c_gammapdf(x, k, theta)

@cython.boundscheck(False)
@cython.wraparound(False)
def gammaMix2pdf(x, lambda1, lambda2, k1, k2, theta1, theta2):
    return c_gamma_mix2_pdf(x, lambda1, lambda2, k1, k2, theta1, theta2)

@cython.boundscheck(False)
@cython.wraparound(False)
def cauchypdf(x, loc, scale):
    return c_cauchypdf(x, loc, scale)

## Cython functions for P-matrix calculations

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double c_twoStatePmat(double pi0,double pi1,double mu,double tau,int i,int j):
        ''''''
        if i == 0:
            if j == 0:
                return pi0 + pi1 * exp( -(mu*tau) )
            else:
                return pi1 - pi1 * exp( -(mu*tau) )
        else:
            if j == 0:
                return pi0 - pi0 * exp( -(mu*tau) )
            else:
                return pi1 + pi0 * exp( -(mu*tau) )

## Python thin wrappers for P-matrix calculations

@cython.boundscheck(False)
@cython.wraparound(False)
def twoStatePmat(pi0,pi1,mu,tau,i,j):
    return c_twoStatePmat(pi0,pi1,mu,tau,i,j)
