""" Unbalanced optimal transport.

Two methods for calculation of optimal transport distance 
between two distributions with uneqaul masses.

Hassan Arbabi, July 2020, arbabiha@gmail.com
"""

#TODO: implement the UW1 explicit difference from Osher paper


import numpy as np
import scipy.linalg as linalg




def sinkhorn_log(mu,nu,c,epsilon, 
                 options={'niter':1000, 'tau':-0.5, 'rho':np.inf}):
    """Computing stabilized sinkhorn over log domain with acceleration.

    Adapted from "SCALING ALGORITHMS FOR UNBALANCED OPTIMAL TRANSPORT PROBLEMS" by Chizat et al, 2017
    and the MATLAB code at 'https://github.com/gpeyre/2017-MCOM-unbalanced-ot' by G. Peyre.

    Args:
        mu: a histogram vector (marginal)
        nu: a histogram vector (marginal)
        c: cost matrix
        epsilon: the entropic regularization strength
        options: 
            niter: max number of the iteration 
            tau: is avering step and negative values usually lead to acceleration
            rho: the strngth of mass variation constraint. np.inf results in classical transport
                but finite values allow mass variation
    
    Returns:
        gamma: the coupling distribution
        Wprimal: the primal transport distance of iterations
        Wdual: its dual
        err: the difference with previous iterate or marginal violation
        Wdistance: the Wasserstein distance at the end (with the entropy term removed)
    """

    for key,val in zip(['tau','rho','niter'],[-.5,np.inf,500]):
        options.setdefault(key, val)
    rho,tau,niter = options['rho'],options['tau'],options['niter']

    lam = rho/(rho+epsilon)
    if rho==np.inf:
        lam=1.0

    H1 = np.ones_like(mu)
    H2 = np.ones_like(nu)

    ave = lambda tau, u, u1: tau*u+(1-tau)*u1

    lse = lambda A: np.log(np.sum(np.exp(A),axis=1))
    M = lambda u,v:(-c+u[:,np.newaxis]@H2[np.newaxis,:] + H1[:,np.newaxis]@v[np.newaxis,:] )/epsilon

    # kullback divergence
    H = lambda p: -np.sum( p.flatten()*(np.log(p.flatten()+1e-20)-1) )
    KL  = lambda h,p: np.sum( h.flatten()* np.log( h.flatten()/p.flatten() ) - h.flatten()+p.flatten())
    KLd = lambda u,p: np.sum( p.flatten()*( np.exp(-u.flatten()) -1) )
    dotp = lambda x,y: np.sum(x*y);     

    err,Wprimal,Wdual = [],[],[]
    u = np.zeros_like(mu)
    v = np.zeros_like(nu)

    for _ in range(niter):
        u1=u
        u = ave(tau, u, lam*epsilon*np.log(mu) - lam*epsilon*lse( M(u,v) ) + lam*u )
        v = ave(tau, v, lam*epsilon*np.log(nu) - lam*epsilon*lse( M(u,v).T) + lam*v )
        gamma = np.exp(M(u,v))

        if rho==np.inf: 
            Wprimal.append(dotp(c,gamma) - epsilon*H(gamma))
            Wdual.append( dotp(u,mu) + dotp(v,nu) - epsilon*np.sum(gamma) )
            err.append( np.linalg.norm( np.sum(gamma,axis=1)-mu ) )
        else:
            Wprimal.append( dotp(c,gamma) - epsilon*H(gamma) \
                            + rho*KL(np.sum(gamma,axis=1),mu) \
                            + rho*KL(np.sum(gamma,axis=0),nu) )

            Wdual.append( -rho*KLd(u/rho,mu) - rho*KLd(v/rho,nu) \
                        - epsilon*np.sum( gamma))
            err.append(np.linalg.norm(u-u1, ord=1) )
    
    WDistance = Wprimal[-1]+epsilon*H(gamma)

    return gamma,Wprimal,Wdual,err,WDistance


def unbalanced_Wasserstein_L1(mu,nu,x = None,alpha = 1):
    """
    Compute the unnormalized L^1 Wasserstein metric for two distributions on the same domain.

    Adapted from "Unnormalized optimal transport" by Gangbo et al, 2019. See refs within.

    Args:
        mu,nu: vector of histogram values on the same (uniformly-sized) bins
        x : the grid for histogram bin centers
        alpha: the inverse of the weight of mass difference in the objective function
            larger alpha means more mass generation/destruction is allowed
    
    Returns:
        the unnormalized L^1 Wasserstein metric between mu and nu

    """

    N = mu.size
    
    if x is None:
        x = np.linspace(0,1,N)

    dx = x[1]-x[0]

    mass_diff = np.sum((mu-nu) * dx) 

    Integrand =  np.abs(    np.cumsum(mu-nu) - x * mass_diff    )


    UW1 = np.sum(Integrand * dx)  + (1/alpha)* np.abs(  mass_diff   )

    return UW1

