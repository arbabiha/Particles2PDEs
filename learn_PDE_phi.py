"""
Learning PDEs for density field of particles.

From "Particles to PDEs Parsimoniously" by Arbabi & Kevrekidis 2020

H. Arbabi, August 2020, arbabiha@gmail.com.
"""

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import scipy.io as sio
import importlib
import tensorflow as tf
import tensorflow.keras as keras
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter

from sys import path
path.append('./thehood/')
import model_library as ML
import CFDroutines as CR


def prep_data(filename='BurgersGT_Z500k_N128_n12_phi.npz',
              ti=[0,10],
              smoothing_sigma=1):
    """Load and preprocess microscopic data.
    
    Args:
        filename: name of file with gap_tooth data
        ti: index of trajectories to pick up
        smoothing_sigma: the STD of teh gaussian filter
        
    Returns:
        x: space grid
        t: time grid
        v: data-driven variable field
        dvdt: density field time derivative
        rho0s: initial density conditions (not gap-tooth particle estimates)
        data_tag: string to distinguish
    """
    
    Data=np.load(filename)
    x=Data['x'].astype('float32')
    Density=Data['phi'][ti[0]:ti[1]]
    t = Data['t']
    rho0s = Data['rho0s'][ti[0]:ti[1]]
    popt=Data['popt']


    dt = t[1]-t[0]

    
    smoother=  lambda u: gaussian_filter(u,smoothing_sigma,mode='wrap')
    v_temp = np.apply_along_axis(smoother,2,Density)

    dvdt,v = [],[]

    for batch in range(v_temp.shape[0]):
        vt_temp= (v_temp[batch,1:,:]-v_temp[batch,:-1,:])/dt
        dvdt.append(vt_temp)
        v.append(v_temp[batch,:-1,:])
    

    dvdt = np.concatenate(dvdt,axis=0)
    v = np.concatenate(v,axis=0)

    data_tag = '_sigma'+str(smoothing_sigma)+'_N'+str(Density.shape[-1])+'_phi'

    return x,t,v,dvdt,rho0s.squeeze(),popt,data_tag


def learn_functional_model(x,t,v,dvdt,rho0s,popt,data_tag):
    """Learning the functional form of the PDE via neural net.
    
    Args:
        x: space grid
        t: time grid
        v: density field
        dvdt: density field time derivative
        data_tag: string to distinguish

    Returns:
        trained keras model mapping v to dvdt
    """

    print(200*'=')
    print('learning a functional model of the PDE ...')
    model_tag='_arch1o'+data_tag

    x_train,x_test,y_train,y_test=train_test_split(v,dvdt,train_size=.85)

    n_grid = x.shape[0]
    dx = x[1]-x[0]

    nn_model = ML.Functional_PDE_net(n_grid,dx,3,n_conv=3,n_neurons=64)
    print('training functional PDE net ...')

    adam_opt=tf.keras.optimizers.Adam(learning_rate=.001)
    nn_model.compile(optimizer=adam_opt,loss='mse')

    PDEfit_history=nn_model.fit(x_train,y_train,
            batch_size=64,epochs=256,
            verbose=0,validation_split=.1)

    plt.figure(figsize=[3,2.5])
    plt.plot(PDEfit_history.history['loss']/np.var(y_test),label='training loss')
    plt.plot(PDEfit_history.history['val_loss']/np.var(y_test),label='validation loss')
    plt.yscale('log')
    plt.legend(),plt.tight_layout()
    plt.savefig('fit'+model_tag,dpi=350)

    eval_loss=nn_model.evaluate(x=x_test,y=y_test,verbose=0)
    eval_lossp=100*eval_loss/np.var(y_test)
    print('test loss %',eval_lossp )

    return nn_model

def learn_discretized_model(x,t,v,dvdt,rho0s,popt,data_tag):
    """Learning the functional form of the PDE via neural net.
    
    Args:
        x: space grid
        t: time grid
        v: density field
        dvdt: density field time derivative
        data_tag: string to distinguish

    Returns:
        trained keras model mapping v to dvdt
    """

    print(200*'=')
    print('learning a discretized model of the PDE ...')
    model_tag='_arch2o'+data_tag

    x_train,x_test,y_train,y_test=train_test_split(v,dvdt,train_size=.85)

    n_grid = x.shape[0]

    nn_model = ML.Discretized_PDE_net(n_grid,3,n_conv=4,n_neurons=64)

    adam_opt=tf.keras.optimizers.Adam(learning_rate=.001)
    nn_model.compile(optimizer=adam_opt,loss='mse')

    PDEfit_history=nn_model.fit(x_train,y_train,
            batch_size=64,epochs=256,
            verbose=0,validation_split=.1)

    plt.figure(figsize=[3,2.5])
    plt.plot(PDEfit_history.history['loss']/np.var(y_test),label='training loss')
    plt.plot(PDEfit_history.history['val_loss']/np.var(y_test),label='validation loss')
    plt.yscale('log')
    plt.legend(),plt.tight_layout()
    plt.savefig('fit'+model_tag,dpi=350)

    eval_loss=nn_model.evaluate(x=x_test,y=y_test,verbose=0)
    eval_lossp=100*eval_loss/np.var(y_test)
    print('test loss %',eval_lossp )

    return nn_model

def test_models(nn1,nn2,x,t,v,dvdt,rho0s,popt,data_tag):
    """Testing nn models in estimating dvdt and trajectory predictions.
    
    
    Args:
        models: list of nn models
        x: space grid
        t: time grid
        v: density field
        dvdt: density field time derivative
        rho0: initial density conditions
        popt: parameters of regression from density to phi
        data_tag: string to distinguish

    Returns:
        saves comparison figures
    """

    RHS1=lambda t,u: nn1.predict(u[np.newaxis,:]).squeeze()
    RHS2=lambda t,u: nn2.predict(u[np.newaxis,:]).squeeze()

    def func(x, a, b, c, d, e):
        y = a + b*x + c*x**2 + d * x**3 + e* np.cos(2*np.pi*x/6)
        return y
    phi_transform = lambda r: func(r,*popt)

    k = 200

    # dvdt plots
    plt.figure(figsize=[6.75/2,1.7])
    plt.subplot(1,2,1)
    plt.plot(x,dvdt[k],'k',label='gap tooth')
    plt.plot(x,RHS1(0,v[k]))
    plt.subplot(1,2,2)
    plt.plot(x,dvdt[k],'k',label='gap tooth')
    plt.plot(x,RHS2(0,v[k]))

    plt.savefig('dphi_dt.png',dpi=450)


    # trajectory prediction
    v_gt = v[::10]
    t_eval = t[:-1:10]


    u0_truth = rho0s

    dx = x[1]-x[0]
    def Standard_FV(t,y):
        dydt= - CR.WENO_FVM_convection(y,dx) + .05 * CR.diffusion_term(y,dx)
        return dydt

    v_truth = solve_ivp(Standard_FV,[0,t_eval[-1]],u0_truth,method='BDF',t_eval=t_eval,max_step=0.01).y.T
    v_truth = phi_transform(v_truth)

    phi0_truth = phi_transform(u0_truth)
    v_nn1=solve_ivp(RHS1,[0,t_eval[-1]],phi0_truth,method='BDF',t_eval=t_eval,max_step=0.01).y.T
    v_nn2=solve_ivp(RHS2,[0,t_eval[-1]],phi0_truth,method='BDF',t_eval=t_eval,max_step=0.01).y.T

    plt.figure(figsize=[6.75 ,2])


    plt.subplot(1,4,1)
    plt.contourf(x,t_eval,v_truth,30,cmap='RdGy')
    plt.colorbar()
    plt.yticks([0,2]),plt.xticks([0,2*np.pi],['0',r'$2\pi$'])
    plt.title(r'$\phi_1(x,t)$'+'\n truth')
    plt.xlabel(r'$x$'),plt.ylabel(r'$t$')


    er0 =  v_truth - v_gt
    rmse1 = np.mean(er0**2)/np.var(v_truth)
    plt.subplot(1,4,2)
    plt.contourf(x,t_eval,er0,30,cmap='RdGy')
    plt.colorbar()
    plt.yticks([0,2]),plt.xticks([0,2*np.pi],['0',r'$2\pi$'])
    plt.title('gap-tooth error \n rMSE={:.1e}'.format(rmse1))
    plt.xlabel(r'$x$'),plt.ylabel(r'$t$')

    er1 =  v_truth - v_nn1
    rmse1 = np.mean(er1**2)/np.var(v_truth)
    plt.subplot(1,4,3)
    plt.contourf(x,t_eval,er1,30,cmap='RdGy')
    plt.colorbar(ticks=[-.03,0])
    plt.yticks([0,2])
    plt.xticks([0,2*np.pi],['0','$2\pi$'])
    plt.title('arch. 1 error \n rMSE={:.1e}'.format(rmse1))
    plt.xlabel(r'$x$'),plt.ylabel(r'$t$')


    er2 =  v_truth - v_nn2
    rmse2 = np.mean(er2**2)/np.var(v_truth)
    plt.subplot(1,4,4)
    plt.contourf(x,t_eval,er2,30,cmap='RdGy')
    plt.colorbar()
    plt.yticks([0,2]), plt.xticks([0,2*np.pi],['0',r'$2\pi$'])
    plt.title('arch. 2 error \n rMSE={:.1e}'.format(rmse2))
    plt.xlabel(r'$x$'),plt.ylabel(r'$t$')

    plt.tight_layout()
    plt.savefig('phi_traj.png',dpi=450)



if __name__=='__main__':
    ttin=time.time()


    train_data = prep_data(smoothing_sigma=1)
    nn1=learn_functional_model(*train_data)
    nn2=learn_discretized_model(*train_data)

    test_data=prep_data(smoothing_sigma=1,ti=[11,12])
    test_models(nn1,nn2,*test_data)

    print( 'run took {} seconds'.format(time.time()-ttin) )



