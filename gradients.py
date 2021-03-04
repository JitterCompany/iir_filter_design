                           
import jax
import jax.numpy as jnp
import numpy as np

def _group_delay(c, w):
    """
    Calculates the group delay for filter with coefficients c for the given frequencies in w
    
    Parameters
    ----------
    
    c: ndarray
        list of all coefficients of all seconds order stages:
        [a_01 a_11 b_01 b_11 ... b_0N b_1N H0]

    w: ndarray
        frequency bins in the range [0, π] to evaluate the group delay on

    """
    
    J = len(c) // 4 # num stages: we don't use H0 = c[-1]
    group_delay = 0
    for i in range(0, J*4, 4):

        a0, a1, b0, b1 = c[i:i+4]

        alpha_n = 1 - a0**2 + a1*(1 - a0) * jnp.cos(w)
        beta_n = a0**2 + a1**2 + 1 + 2*a0*1*(2*jnp.cos(w)**2 - 1) + 2*a1*(a0 + 1)*jnp.cos(w)
        alpha_d = 1 - b0**2 + b1*(1 - b0)*jnp.cos(w)
        beta_d =  b0**2 + b1**2 + 1 + 2*b0*1*(2*jnp.cos(w)**2 - 1) + 2*b1*(b0 + 1)*jnp.cos(w)

        group_delay += -alpha_n/beta_n + alpha_d/beta_d

    return group_delay



group_delay_gradient = jax.jacrev(_group_delay)


def _group_delay_deviation(x, w):
    """
    Calculates the group delay deviation for filter with coefficients x for the given frequencies in w
    
    parameters
    ----------
    
    x: ndarray
        list of all coefficients of all seconds order stages and tau, the group delay optimization variable:
        [c tau]

    w: ndarray
        frequency bins in the range [0, π] to evaluate the group delay on

    """
    
    J = (len(x) - 2) // 4 # num stages: we don't use H0 = c[-2], tau is c[-1] 
    tau = x[-1]
    group_delay = 0
    for i in range(0, J*4, 4):

        a0, a1, b0, b1 = x[i:i+4]

        alpha_n = 1 - a0**2 + a1*(1 - a0) * jnp.cos(w)
        beta_n = a0**2 + a1**2 + 1 + 2*a0*1*(2*jnp.cos(w)**2 - 1) + 2*a1*(a0 + 1)*jnp.cos(w)
        alpha_d = 1 - b0**2 + b1*(1 - b0)*jnp.cos(w)
        beta_d =  b0**2 + b1**2 + 1 + 2*b0*1*(2*jnp.cos(w)**2 - 1) + 2*b1*(b0 + 1)*jnp.cos(w)

        group_delay += -alpha_n/beta_n + alpha_d/beta_d

    return group_delay - tau

group_delay_deviation_gradient = jax.jacrev(_group_delay_deviation)


def _H_eval(c, w):
    """ Evaluates the filter transfer function for each frequency in w.

    Parameters
    ----------
     c: ndarray
        list of all coefficients of all seconds order stages:
        [a_01 a_11 b_01 b_11 ... b_0N b_1N H0]
            
    w : ndarray
        frequency bins between 0 and 2π
        
    Returns
    -------
    
    H: ndarray with dtype 'complex64'
    
    """
    
    _H = jnp.ones(len(w), dtype='complex64')
    J = len(c) // 4 # number of 2nd order filter sections
    H0 = c[J*4] # H0 is the first element after the coefficients. 
    for i in range(0, J*4, 4):
        a0, a1, b0, b1 = c[i:i+4]
        _H *= (a0 + a1*jnp.exp(1j*w) + jnp.exp(2*1j*w)) / (b0 + b1*jnp.exp(1j*w) + jnp.exp(2*1j*w))
    return H0 * _H

def _H_mag_squared(c, w):
    """ 
    Calculcate the squared magnitude response of the filter.
    
    Parameters
    ----------
     c: ndarray
        list of all coefficients of all seconds order stages:
        [a_01 a_11 b_01 b_11 ... b_0N b_1N H0]
            
    w : ndarray
        frequency bins between 0 and 2π
        
        
    Returns
    -------
    
    H: ndarray with dtype 'float32'
        
    """
    return jnp.abs(_H_eval(c,w))**2


H_mag_squared_gradient = jax.jacrev(_H_mag_squared)



def H_gradient(x, w):
    """ Analytical implementation of the gradient of the 
        transfer function H with respect to the coefficients x
        
        
        
    """
    
    J = len(c) // 4 # number of 2nd order filter sections
    H0 = c[J*4] # H0 is the first element after the coefficients. 
    
    H = H_eval(x, w)
    grad = []
    for i in range(0, J*4, 4):
        a0, a1, b0, b1 = c[i:i+4]
        
        
        dhda0 = (1 / (a0 + (a1 + 1) * np.exp(1j*w))) * H
        dhda1 = (np.exp(1j*w) / (a0 + (a1 + 1) * np.exp(1j*w))) * H
        
        dhdb0 = (1 / (b0 + (b1 + 1) * np.exp(1j*w))) * H
        dhdb1 = (np.exp(1j*w) / (b0 + (b1 + 1) * np.exp(1j*w))) * H
        
        
        grad.extend([dhda0, dhda1, dhdb0, dhdb1 ])

    dhdh0 = np.ones(len(w))
    dhdtau = np.zeros(len(w))
    grad.extend([dhdh0, dhdtau])

    return np.c_[grad].T