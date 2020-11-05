"""Some routines for solving 1D PDES"""


import numpy as np 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import timeit
import scipy.io as sio
from scipy.integrate import solve_ivp





def test_flux_WENO():
    x = np.linspace(0,2*np.pi,num=200,endpoint=False)
    h = x[1]-x[0]
    b = 1.5
    u = np.sin(b*x)


    dfdx_truth=b* np.sin(b*x)*np.cos(b*x)
    dfdx_WENO=WENO_FVM_convection(u,h)

    plt.figure(figsize=[6.5,4])
    plt.plot(x,dfdx_truth,'x')
    plt.plot(x,dfdx_WENO,'--')
    # plt.ylim(-1,1.1)
    # plt.xlim(-.01,.05)

    plt.savefig('./figs/WENO_flux.png')


def WENO_FVM_convection(V,h,flux='Gudonov'):
    """ Finite-volume approximation of momentum flux using WENO.


    Args:
        V: solution values on 1d periodic grid
        h: (uniform) grid size

    Returns:
        array of same size as V holding values of d(v**2)/dx
    """

    # periodify
    V = np.concatenate((V[-3:],V,V[0:3]),axis=0)

    # compute v+ and v- at the right boundaries of each cell 
    v1 = V[0:-5]
    v2 = V[1:-4]
    v3 = V[2:-3]
    v4 = V[3:-2]
    v5 = V[4:-1]
    v6 = V[5:]

    v_minus  = WENO_reconstruct(v1,v2,v3,v4,v5)
    v_plus   = WENO_reconstruct(v6,v5,v4,v3,v2)


    if flux is 'Gudonov':
        F = Gudonov_v2(v_minus,v_plus)
    else:
        raise NotImplementedError('flux method not known or implemented yet')


    # flux balance

    dFdx = (1/h)*(F[1:]-F[:-1])



    return dFdx


def Gudonov_v2(v_minus,v_plus):
    """Computing Gudonov method for flux f(v)=v^2/2."""


    F=np.zeros(v_minus.shape)

    # when v- is smaller than v+
    leftD = np.where(v_minus<=v_plus)

    F[leftD]=np.minimum(v_minus[leftD]**2,v_plus[leftD]**2)

    x_cross=np.where(v_minus*v_plus<0)
    F[x_cross]=0




    # when v- is larger than v+
    rightD= np.where(v_minus>v_plus)  # optimize this later

    F[rightD]=np.maximum(v_minus[rightD]**2,v_plus[rightD]**2)

    return F/2

def WENO_reconstruct(v1,v2,v3,v4,v5):
    """Reconstructing value of field v at boundary nodes.


    Args:
        v1: array holding values at node i-2
        v2: array holding values at node i-1
        v3: array holding values at node i
        v4: array holding values at node i+1
        v5: array holding values at node i+2

    Returns:
        array of values at (boundary) node i+1/2
    """



    # combining three stenccils
    phi1 = v1/3 - 7*v2/6 + 11*v3/6
    phi2 =-v2/6 + 5*v3/6 +    v4/3
    phi3 = v3/3 + 5*v4/6 -    v5/6

    #  measures of smoothness for each stencil (larger the S --> less smooth)
    S1 = (13/12)*(v1-2*v2+v3)**2+(1/4)*(v1-4*v2+3*v3)**2
    S2 = (13/12)*(v2-2*v3+v4)**2+(1/4)*(v2-v4)**2
    S3 = (13/12)*(v3-2*v4+v5)**2+(1/4)*(3*v3-4*v4+v5)**2

    # deciding the weights at each point
    V = np.stack((v1,v2,v3,v4,v5),axis=1)
    EPS = np.amax(V,axis=1)**2 * 1e-6 + 1e-99

    # non-normalized weights
    a1 = 0.1/ (S1+EPS)**2
    a2 = 0.6/ (S2+EPS)**2
    a3 = 0.3/ (S3+EPS)**2

    # combine the stencils
    v = (a1*phi1 + a2*phi2 + a3*phi3)/(a1+a2+a3)

 
    return v





def linear_adv_WENO(C,h,U):
    """Computing the advection term via WENO.
    
    Args:
        C: the field to be advected
        h: grid spacing
        U: velocity field
    
    Returns:
        np.array containing U* dC/dx
    
    (use this for advection by a given velocity field)
    see Osher & Fedkiw, Level Set Methods, 2003
    
    """

    N = C.shape[0]
    adv_term=np.zeros(N)

    
    if not isinstance(U, (np.ndarray)):
        U = U * np.ones(N)
    

    # index of points with left and right updwind direction
    LeftD =np.where(U>=0)
    RightD=np.where(U<0)

    # periodify
    C = np.concatenate((C[-3:],C,C[0:3]),axis=0)



    # divided difference at all points
    D = (C[1:]-C[:-1])/h

    v1 = D[0:-5]
    v2 = D[1:-4]
    v3 = D[2:-3]
    v4 = D[3:-2]
    v5 = D[4:-1]
    v6 = D[5:]



    # now reconstruct the actual derivative using WENO
    vx_left  = WENO_reconstruct(v1[LeftD],v2[LeftD],v3[LeftD],\
                                v4[LeftD],v5[LeftD])
    vx_right  = WENO_reconstruct(v6[RightD],v5[RightD],v4[RightD],\
                                v3[RightD],v2[RightD])

    adv_term[LeftD] =vx_left * U[LeftD]
    adv_term[RightD]=vx_right* U[RightD]


    return adv_term

def diffusion_term(V,h,nu=1):
    # central finite-difference to approximate the diffusion term
    # periodify
    V = np.concatenate((V[-1:],V,V[0:1]),axis=0)
    D2V = (-2*V[1:-1] + V[0:-2] + V[2:])  / (h**2)
    return D2V*nu

def test_diffusion():
    x = np.linspace(0,2*np.pi,num=200,endpoint=False)
    h = x[1]-x[0]


    c0 = np.sin(x) + np.cos(x)



    diff_cd= diffusion_term(c0,h,nu=1)
    diff_tru = -np.sin(x) - np.cos(x)


    plt.figure(figsize=[6.5,4])
    plt.plot(x,diff_cd,'x')
    plt.plot(x,diff_tru,'--')
    # plt.ylim(-1,1.1)
    # plt.xlim(-.01,.05)

    plt.savefig('./figs/diff_test.png')


def test_WENOderivative():
    x = np.linspace(0,2*np.pi,num=200,endpoint=False)
    h = x[1]-x[0]
    c0 = np.sin(x) + 2.1*np.cos(2*x)

    U0 = 1
    # U  = U0 *np.ones(x.shape) # constant velocity

    adv_WENO = linear_adv_WENO(c0,h,U0)
    adv_tru = (np.cos(x)-4.2*np.sin(2*x)) * U0

    plt.figure(figsize=[6.5,4])
    plt.plot(x,adv_WENO,'x')
    plt.plot(x,adv_tru,'--')
    # plt.ylim(-1,1.1)
    # plt.xlim(-.01,.05)

    plt.savefig('./figs/WENO_der_test.png')



def solve_ivp_TVD(F,tspan,y0,tval,max_step=.01):
    """Solving initial value problem with a total variation method.
    
    Args:
        F: callable that takes (t,y) and returns ydot
        tsapn: time span of integration
        u0: initial condition
        tval: values of t at which the solution is recorded

    
    Returns:
        Y: the array nt*n where n is the dimension of y0
    """
    nt = len(tval)
    Y = np.zeros((nt,y0.shape[0]))

    assert tspan[0] <= tval[0], 'tspan should contain tvals'
    assert tspan[-1] >= tval[-1], 'tspan should contain tvals'

    tstart = tspan[0]

    for j in range(nt):
        # re-compute the dt
        interval = tval[j]-tstart
        nstep = int(interval//max_step)

        if nstep !=0:
            dt = interval/nstep
            
            for k in range(nstep):
                y0 = TVD_RK3(F,tstart+k*dt,y0,dt)

        Y[j]=y0
        tstart=tval[j]

    return Y
    



def TVD_RK3(F,t,u,dt):
    """Total variation Runge-Kutta for time marching hyperbolic PDEs.
    
    Args:
        F: callable that takes (t,u) and returns udot
        t: time
        dt: time step size
        u: solution at time t0

    Returns:
        u_next: solution at time t0+dt
    
    """
    # Total Variation Diminishing (TVD) Runge-Kutta of order 3
    # for solving udot=F(t,u)
    # see Shu 2009, 
    # or Osher-Fedkiw 2003 for more detail

    u1 = u + dt*F(t,u)
    u2 = (3/4)*u + (1/4)*u1 + (1/4)*dt*F(t+dt,u1)
    u_next = (1/3)*u + (2/3) *u2 + (2/3)*dt*F(t+dt/2,u2)

    return u_next


def linear_advection_test():
    # testing WENO in linear advection

    x = np.linspace(0,2*np.pi,num=100,endpoint=False)
    h = x[1]-x[0]

    t = np.linspace(0,10,301)

    def WaveS(x):
        s=np.zeros(x.shape)
        s[x>2]=1
        s[x>3]=.5
        return s

    c0 = WaveS(x)
    U0 = .1
    U  = U0 *np.ones(x.shape) # constant velocity


    # ode RHS
    rhs = lambda t,y: -1.0*linear_adv_WENO(y,h,V0=U)


    Sol = solve_ivp(rhs,(np.amin(t),np.amax(t)),c0,method='RK45',\
        t_eval=t,max_step=.1 )


    V = np.transpose(Sol.y)
    dt = t[1]-t[0]

    plt.figure(figsize=[6.5,4])
    plt.subplot(2,2,1)
    j = 0
    Vtruth = WaveS(x-U0*j*dt)
    plt.plot(x,V[j,:])
    plt.plot(x,Vtruth,'--')
    plt.xlabel('$X$'),plt.ylabel('$V$')
    plt.title('initial condition')
    plt.ylim(-1,1)


    plt.subplot(2,2,2)
    j = 100
    Vtruth = WaveS(x-U0*j*dt)
    plt.plot(x,V[j,:])
    plt.plot(x,Vtruth,'--')
    plt.xlabel('$X$'),plt.ylabel('$V$')
    plt.title('t='+str(t[j]))
    plt.ylim(-1,1)

    plt.subplot(2,2,3)
    j = 200
    Vtruth = WaveS(x-U0*j*dt)
    plt.plot(x,V[j,:])
    plt.plot(x,Vtruth,'--')
    plt.xlabel('$X$'),plt.ylabel('$V$')
    plt.title('t='+str(t[j]))
    plt.ylim(-1,1)


    plt.subplot(2,2,4)
    j = 300
    Vtruth = WaveS(x-U0*j*dt)
    plt.plot(x,V[j,:])
    plt.plot(x,Vtruth,'--')
    plt.xlabel('$X$'),plt.ylabel('$V$')
    plt.title('t='+str(t[j]))
    plt.ylim(-1,1)

    plt.subplots_adjust(wspace=.4,hspace=.6)
    plt.savefig('./figs/WENO_lin_avd.png')

if __name__=='__main__':
    print('Jesus take the wheel')

    # V = np.arange(0,10)
    # V = np.concatenate((V[-3:],V,V[:3]),axis=0)
    # print(V)
    # linear_advection_test()
    # test_WENOderivative()
    # test_diffusion()
    # test_flux_WENO()
    # test_Burgers_WENO()
    test_WENOderivative()


