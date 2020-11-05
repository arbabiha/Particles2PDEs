""" 
Illustrating gap-tooth method and data generation for Burgers model. 

From the paper
[1]: "Particles to PDEs parsimoniously" by Arbabi & Kevrekidis 2020

Gap-tooth method is based on 
[2]: "The Gap-Tooth Method in Particle Simulations" by Gear, Lu & Kevrekidis 2003


Hassan Arbabi, arbabiha@gmail.com 
April 4, 2020
"""


import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import timeit


from sys import path
path.append('./thehood/')

import BurgersGapTooth as BGT
import CFDroutines as CR



def gaptooth_vs_truth(nu=0.05, N=128, Z=10000.0, alpha=.1):
    """Running gap-tooth and comapre vs finite volume.

    Generates fig 1(b) in the paper.
    
    Args:
        nu: viscosity
        N: numbeer of teeth
        Z: resolution factor, i.e., number of particles per unit mass
        alpha: fraction of space occupied by teeth

    """
    np.random.seed(41)

    rho0=lambda x: 1-np.sin(x)/2


    tsys = BGT.BurgersGapToothSystem(alpha=alpha,N=N,nu=nu)
    tsys.initialize(Z/alpha,rho0=rho0)
    tsys.run_gap_tooth(dt = .002, Nt = 1000,save_history=True)
    grid_g,rho_g,t_g=tsys.tooth_center_x,tsys.rho_history,tsys.rho_history_t
    rho_g=np.stack(rho_g)

    dx = grid_g[1]-grid_g[0]

    def FiniteVolume(t,y):
        dydt= - CR.WENO_FVM_convection(y,dx) + nu * CR.diffusion_term(y,dx)
        return dydt

    u0 = rho0(grid_g) 
    Sol = solve_ivp(FiniteVolume,[0,t_g[-1]],u0,method='BDF',t_eval=t_g,max_step=0.01)
    rho_truth=Sol.y.T


    x_padded = np.concatenate((grid_g,2*grid_g[-1:]-grid_g[-2:-1]))
    periodic_pad = lambda u: np.concatenate([u,u[:,-1:]],axis=-1)

    plt.figure(figsize=[4,2])
    plt.subplot(1,2,1)
    plt.contourf(x_padded,t_g,periodic_pad(rho_g),30,cmap='jet')
    plt.yticks([0,2]),plt.xticks([0,2*np.pi],['0','$2\pi$'])
    plt.title(r'$\rho(x,t)$'+'\n gap tooth')
    plt.colorbar()
    plt.xlabel(r'$x$'),plt.ylabel(r'$t$')

    er = rho_g-rho_truth
    rMSE = np.mean(er**2)/np.var(rho_truth)

    plt.subplot(1,2,2)
    plt.contourf(x_padded,t_g,periodic_pad(er),30,cmap='jet')
    plt.yticks([0,2]),plt.xticks([0,2*np.pi],['0','$2\pi$'])
    plt.title('error \n rMSE={:.1e}'.format(rMSE))
    plt.colorbar()
    plt.xlabel(r'$x$'),plt.ylabel(r'$t$')

    plt.tight_layout()
    plt.savefig('Burgers_gaptooth.png',dpi=450)


def basic_traj():
    pass



def generate_data():
    """Generates 12 trajectories of gap-tooth simulation with random initial conditions."""

    ntraj=12
    alpha = .1
    N =128
    Zk = 500
    Z = Zk*1000
    nu =.05
    Nt = 1000
    dt =.002

    rho_histories=[]
    teeth_histograms=[]
    rho0s=[]

    for k in range(ntraj):
        print(10*'--')
        print('k='+str(k))
        rho0=get_random_rho()
        tsys=BGT.BurgersGapToothSystem(alpha=alpha,N=N,nu=nu)
        tsys.initialize(Z,rho0=rho0)
        rho0s.append(rho0(tsys.tooth_center_x)) 
        print('# of particles='+str(tsys.particle_count))
        tsys.run_gap_tooth(dt=dt,Nt=Nt,save_inner_histograms=True)

        rho_history = np.stack(tsys.rho_history,axis=1).T
        rho_histories.append(rho_history)

        histograms_history = np.stack(tsys.inner_histograms_history)
        teeth_histograms.append(histograms_history)

    np.savez('BurgersGT_Z{}k_N{}_n{}_'.format(Zk,N,ntraj)+'.npz',
                    Density=rho_histories,t=tsys.t,
                    Z=Z,alpha=alpha,nu=nu,x=tsys.tooth_center_x,
                    rho0s=np.stack(rho0s),teeth_histograms=teeth_histograms) 


def get_random_rho():
    """Generates a positive random profile.

    Returns:
        a callable that is positive everywhere on (0,2pi).
    """

    N= 20
    A = np.random.rand(N)-.5
    phi,l=np.random.rand(N)*2*np.pi,np.random.randint(1,high=7,size=N)

    def rho(x):
        y = 0
        for k in range(N):
            y = y + A[k]*np.sin(l[k]*x + phi[k])
        return y
            
    # make it positive
    x = np.linspace(0,2*np.pi,num=128)
    r = rho(x)
    rmin = np.min(r)
    if rmin<0.05:
        rho2= lambda x: rho(x) + np.abs(rmin) + .1
    else:
        rho2 = rho

    return rho2

if __name__ == "__main__":
    ttin = timeit.default_timer()
    gaptooth_vs_truth()
    generate_data()

    print('whole run took {} seconds'.format(timeit.default_timer() - ttin))


