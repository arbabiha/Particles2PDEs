""" 
An implementaion of gap-tooth scheme for Burgers equation.

Based on "The Gap-Tooth Method in Particle Simulations" by Gear, Lu & Kevrekidis 2003
Hassan Arbabi, arbabiha@gmail.com 
April 4, 2020
"""




import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import timeit
import os
import scipy.fftpack as fftpack
from scipy.linalg import toeplitz
import scipy.integrate
import pickle



class BurgersGapToothSystem:
    """Class of gap tooth modeling of Burgers system.


    Attributes:
        teeth: list of arrays; i-th array is the particle positons in tooth i
        cavities: list of arrays; i-th array is the anti-particle positons in tooth i

        right_outflux_particle: list of arrays; i-th array is the extrusion of particles getting out of tooth i
        right_outflux_anti: list of arrays; i-th array is the extrusion of anti_particles getting out of tooth i
        left_outflux_particle: list of arrays; i-th array is the extrusion of particles getting out of tooth i
        left_outflux_anti: list of arrays; i-th array is the extrusion of anti_particles getting out of tooth i

        adjacency_mat: an array; the i-th row is [index of patch to the left of patch i, i, index of right patch]
        left_boundary: list of left boundary positions
        right_boundary: list of right boundary positions
        t: list of time stamps so far         
        .....
    """
    
    def __init__(self, N=40, alpha = 1, domain_bounds=[0,2*np.pi], nu=0.05, dt = None ):
        """Creating a uniform grid of teeth on a periodic domain.

        Args:
            N: number of teeth
            alpha: gap length; 1 no gap between teeth, 0 infinitely thin teeth
            domain_bounds: bounds of the domain duh
            nu: viscosity
        """
        self.N = N
        self.dt=dt
        self.alpha=alpha
        self.nu = nu
        self.t = [0]
        self.domain_bounds=domain_bounds
        D = (domain_bounds[1]-domain_bounds[0])/N
        self.tooth_center_x = np.arange(domain_bounds[0],domain_bounds[1],D)
        self.tooth_width = alpha*D
        self.left_boundaries = self.tooth_center_x - self.tooth_width/2
        self.right_boundaries = self.tooth_center_x + self.tooth_width/2

        # only 2 neighbors are required
        self.adjacency_mat = np.stack((np.arange(-1,N-1),np.arange(N),np.arange(1,N+1)),axis=1)
        self.adjacency_mat[N-1,2] = 0

        # flux resitriution coeffs
        self.a11 = 1- self.alpha**2
        self.a10 = - self.alpha *(1-self.alpha)/2
        self.a12 = self.alpha * (1+self.alpha)/2
 
    def initialize(self, Z: float, rho0=lambda x: 1-np.sin(np.pi*x)/2, inner_histogram_size = 10):
        """Lift density to particle positions.

        Args:
            Z: resolution, i.e., number of particles representing one unit of mass
            rho: initial density profile callable or numpy array
            inner_histogram_size: number of bins for tooth histograms

        Return:
            teeth
        """
        self.Z=Z 
        self.teeth = []

        if callable(rho0):
            rho_vals = rho0(self.tooth_center_x)
        else:
            rho_vals = rho0

        for xc,itooth in zip(self.tooth_center_x,range(self.N)):
            rho_tooth = rho_vals[itooth]
            tooth_bounds=[xc-self.tooth_width/2,xc+self.tooth_width/2]
            patch,_=lift_from_density(self.Z,rho_tooth,tooth_bounds)
            self.teeth.append(patch)
        
        self.cavities = [np.array([]) for k in range(self.N)]

        self.particle_history = [self.teeth]
        self.particle_history_t = [0]

        self.compute_density()
        self.rho_history = [self.rho]
        self.rho_history_t = [0]
        
        self.inner_histogram_size=inner_histogram_size
        self.compute_inner_histograms()
        self.inner_histograms_history = [self.inner_histograms]
        self.inner_histograms_t=[0]

    def compute_density(self):
        """Computing density of each tooth."""
        self.rho = np.array([(tooth.size/self.Z)/self.tooth_width for tooth in self.teeth])

    def compute_inner_histograms(self):
        """Computes the histogram of particles within each tooth."""
        self.inner_histograms=[]
        for tooth,lb,rb in zip(self.teeth,self.left_boundaries,self.right_boundaries):
            self.inner_histograms.append(np.histogram(tooth,bins=self.inner_histogram_size,range=[lb,rb])[0])

    @property 
    def how_many_redistributions(self):
        """Number of redistributions so that every (anti)-particle ends up inside teeth."""
        self.compute_density()
        u_max = np.max(self.rho)
        jump_max = max(u_max*self.dt,4*np.sqrt(2*self.nu*self.dt))
        return int(jump_max/(self.tooth_width/2)+1)

    @property
    def total_mass(self):
        self.compute_density()
        return np.sum(self.rho * (self.domain_bounds[1]-self.domain_bounds[0])/self.N)

    @property
    def particle_count(self):
        """No. of particles in teeth."""
        each_tooth=[p.size for p in self.teeth]
        return sum(each_tooth)
    
    @property
    def anti_count(self):
        """No of anti-particles in teeth."""
        each_tooth=[p.size for p in self.cavities]
        return sum(each_tooth)

    @property
    def particle_count_outflux(self):
        """No of particles in outflux."""
        right=[p.size for p in self.right_outflux_particle]
        left=[p.size for p in self.left_outflux_particle]
        return sum(right)+sum(left)
    

    @property
    def anti_count_ouflux(self):
        """No of particles in teeth."""
        right=[p.size for p in self.right_outflux_anti]
        left=[p.size for p in self.left_outflux_anti]
        return sum(right)+sum(left)


    def update_teeth(self):
        """Updating the state of particles within teeth via Burgers dynamics.
        
        This update takes each particle from x_t to x_{t+1} but does not 
        include the implementaion of influx or computation of outflux. 
        """
        
        self.teeth = [Burgers_tooth_update(tooth,self.nu,self.tooth_width,self.dt,self.Z)   for tooth in self.teeth]
    
    
    def compute_outflux(self):
        """Computes the outgoing particles and anti-particles from all teeth.
        
        From patches, done for all patches, by the patches  ;).
        It looks at teeth, identifies what particles have gone out of teeth,
        and puts their extrusion in right_outflux and left_outlfux."""

        # not efficient  duh  
        self.right_outflux_particle = [x[x>right_bdry]-right_bdry for  x,right_bdry in zip(self.teeth,self.right_boundaries)]
        self.left_outflux_particle = [x[x<left_bdry]-left_bdry for  x,left_bdry in zip(self.teeth,self.left_boundaries)]
        self.teeth=[patch[(patch<right_bdry)&(patch>left_bdry)] \
            for patch,left_bdry,right_bdry in zip(self.teeth,self.left_boundaries,self.right_boundaries)]
        



        self.right_outflux_anti = [x[x>right_bdry]-right_bdry for  x,right_bdry in zip(self.cavities,self.right_boundaries)]
        self.left_outflux_anti = [x[x<left_bdry]-left_bdry for  x,left_bdry in zip(self.cavities,self.left_boundaries)]
        self.cavities=[patch[(patch<right_bdry)&(patch>left_bdry)] \
            for patch,left_bdry,right_bdry in zip(self.cavities,self.left_boundaries,self.right_boundaries)]
        return



    def _interpolate_influx(self,outflux):
        """Interpolates one outflux into three influxes.
        
        IMPORTANT: indices show order in the flow direction: 0 upstream,
        1 current tooth, 2 downstream
        """

        n = outflux.size
        n10 = int(- self.a10 * n)
        n12 = int(self.a12 * n)

        # in order to preserve mass we compute n11 this way
        n11 = n - n12 + n10
        # print(str(n)+'--->'+str(n10)+','+str(n11)+','+str(n12))

        np.random.shuffle(outflux)
        I_upstream_anti = outflux[:n10]  # this one makes particles anti-particles and vice versa
        I_self = outflux[:n11]
        I_downstream = outflux[n-n12:n]  # avoiding special behavior when n12=0


        return I_upstream_anti,I_self,I_downstream

    def _distribute_outflux(self,O1,I0,I1,I2):
        """Breaking outflux O1 into influxes I0,I1,I2."""
        I_upstream_anti,I_self,I_downstream=self._interpolate_influx(O1)
        I0.append(I_upstream_anti)
        I1.append(I_self)
        I2.append(I_downstream)


    def compute_influx(self):
        """Computes the influx of particles and anti-particles into each tooth.
        
        Uses the quadratic interpolation formula to redistribute the outflux 
        into influxes.
        """
        self.left_influx_particle = [ [] for i in range(self.N) ]
        self.left_influx_anti =  [ [] for i in range(self.N) ]
        self.right_influx_particle =   [ [] for i in range(self.N) ]
        self.right_influx_anti = [ [] for i in range(self.N) ]


        # right-going fluxes
        upstream,downstream = self.adjacency_mat[:,0],self.adjacency_mat[:,2]

        for j in range(self.N):
            # particles
            self._distribute_outflux( self.right_outflux_particle[j],\
                self.right_influx_anti[upstream[j]],self.right_influx_particle[j],self.right_influx_particle[downstream[j]])

            self._distribute_outflux( self.right_outflux_anti[j],\
                self.right_influx_particle[upstream[j]],self.right_influx_anti[j],self.right_influx_anti[downstream[j]])



        # left-going fluxes
        upstream,downstream = self.adjacency_mat[:,2],self.adjacency_mat[:,0]
        
        for j in range(self.N):
            # particles
            self._distribute_outflux(self.left_outflux_particle[j],\
                self.left_influx_anti[upstream[j]],self.left_influx_particle[j], self.left_influx_particle[downstream[j]])



            # anti-particles
            self._distribute_outflux(self.left_outflux_anti[j],\
                self.left_influx_particle[upstream[j]],self.left_influx_anti[j], self.left_influx_anti[downstream[j]])



        # TODO: consider moving this to the loop
        self.left_influx_particle=[np.concatenate(x) for x in self.left_influx_particle]
        self.left_influx_anti=[np.concatenate(x) for x in self.left_influx_anti]
        self.right_influx_particle=[np.concatenate(x) for x in self.right_influx_particle]
        self.right_influx_anti=[np.concatenate(x) for x in self.right_influx_anti]

    

    def collect_influx(self):
        """Merge influxes with each tooth and cavity."""
        
        # loop seems more comprehensible than list comprehension
        for j in range(self.N):
            incoming_particles_l= self.left_influx_particle[j] + self.right_boundaries[j]
            incoming_particles_r= self.right_influx_particle[j] + self.left_boundaries[j]
            
            self.teeth[j] = np.concatenate((self.teeth[j],incoming_particles_l,incoming_particles_r)) 

            incoming_anti_l= self.left_influx_anti[j] + self.right_boundaries[j]
            incoming_anti_r= self.right_influx_anti[j] + self.left_boundaries[j]

            self.cavities[j] = np.concatenate((self.cavities[j],incoming_anti_l,incoming_anti_r))


    def fuse_cavity_tooth(self):
        """Annihilate anti-particles and particles within each tooth."""
        self.teeth=[annihilate_particles(tooth,cavity) for tooth,cavity in zip(self.teeth,self.cavities) ]
        self.cavities = [np.array([]) for k in range(self.N)]
        
            
    def redistribute_particles(self, max_iter = 5):
        """Redistributes particles and anti-particles till they land inside teeth.
        
        Repeates outflux, influx computation and merge with tooth until all particles or anti-particle are inside the teeth or max_iter is reached.
        """
        for j in range(max_iter):
            self.compute_outflux()
            if (self.anti_count_ouflux + self.particle_count_outflux)==0:
                break
            self.compute_influx()
            self.collect_influx()
            # self.visualize_teeth()

        if j==max_iter-1: 
            print('max iteration reached for redistribution process')


    def run_gap_tooth(self, dt: float,  Nt = 1, 
                      show_runtime=True, 
                      save_history=True, 
                      save_particle_history=False,
                      save_inner_histograms=False):
        """Continue simulating the system.
        
        Args:
            dt: time step
            Nt: number of time steps
            show_runtime: show how much time for whole run has elapsed
            save_history: saves the history of the rho evolution to self.rho_history
            save_particle_history: saves the history of particle position in teeth -- MEMORY EXPENSIVE
            tooth_histograms: the number of bins used for histograms within each tooth,
                if None, those histograms are not computed or saved.
        """

        self.dt = dt
        ttin = timeit.default_timer()

        n_redist = self.how_many_redistributions * 4

        for _ in range(Nt):
            self.t.append(self.t[-1]+self.dt)
            self.update_teeth()
            self.redistribute_particles(max_iter=n_redist)
            self.fuse_cavity_tooth()
            
            if save_history:
                self.compute_density()
                self.rho_history.append(self.rho)
                self.rho_history_t.append(self.t[-1])

            if save_inner_histograms:
                self.compute_inner_histograms()
                self.inner_histograms_history.append(self.inner_histograms)
                self.inner_histograms_t.append(self.t[-1])

            if save_particle_history:
                self.particle_history.append(self.teeth)
                self.particle_history_t.append(self.t[-1])
            
            


        if show_runtime:
            print('run took {} seconds'.format(timeit.default_timer() - ttin))



def lift_from_density(Z: int, rho: float, bin_edges):
    """Lifting density funcation to particle positions for a single tooth.

    This is Hassan's understanding of the paper; not optimized but understandable.

    Args:
        Z: resolution, i.e., number of particles representing one unit of mass
        rho: value of density within bin
        bin_edges: duh
        n_bin: how many bins put into the domain and used for lifting
    
    Returns:
        x_particles: array of particle positions
        Z_new: the adjusted value of Z
    """

    bin_edges = np.array(bin_edges)
    bin_width=bin_edges[1]-bin_edges[0]
    x_bin = (bin_edges[:-1]+bin_edges[1:])/2 # center of the bin

    mass_bin = rho *  bin_width
    n_particles_bin =  np.floor(mass_bin * Z )


    x_particles= []

    r = np.random.rand(int(n_particles_bin)) -.5 # n_particles_bin random numbers in [-.5,.5]
    x_particles = x_bin+ bin_width * r


    # we have slightly changed Z so we report back the new Z
    Z_new = x_particles.size / np.sum(mass_bin)


    return x_particles,Z_new


def annihilate_particles(tooth,cavity):
    """Cancel particles in tooth with anti-particles in cavity.
    
    Finds the closest particle in tooth to each anti-particle in cavity
    and deletes that particle.
    
    Args:
        tooth: array of particle positions
        cavity: array of anti-particle positions
        
    Returns:
        array of remaining particles
    """

    # not efficient
    for j in range(cavity.size):
        dist = np.abs(cavity[j]-tooth)
        ind_close = np.argmin(dist)
        tooth=np.delete(tooth,ind_close)

    return tooth

def Burgers_tooth_update(x_particles: np.ndarray, nu: float, tooth_width: float, dt: float, Z: float):
    """Updates particles positions according to viscous Burgers.
    
    Args:
        x_particles: array of particle positions
        nu: viscosity
        tooth_width: width of the teeth
        dt: time step
        Z: resolution factor, i.e., number of particles per unit mass
        """

    n_partciles = x_particles.size
    mass_tooth = n_partciles/Z 
    rho_tooth = mass_tooth/tooth_width
    u_drift = rho_tooth/2
    
    jump = u_drift*dt + np.random.randn(n_partciles)*np.sqrt(2*nu*dt)
    x_particles=x_particles + jump
    return x_particles




def compute_truth_Burgers(rho_0, nu: float = 0.05, t: float=0, 
                          ngrid = None):
    """Computing the truth density of Burgers via Cole-Hopf transform.

    Be careful: sometimes lead to artificially irregular approximations.
    
    Args:
        rho_0: the density profile at t=0, either callable or array
        nu: viscosity
        t: time
        ngrid: size of the grid on [0,2pi], overwritten if rho_0 is array
        
    Returns:
        the grid and density profile
    """
    
    if not callable(rho_0):
        density = rho_0
        ngrid=rho_0.shape[0]
        x = np.arange(0,1-.5/ngrid,1/ngrid) * 2* np.pi 
    else:
        x = np.arange(0,1-.5/ngrid,1/ngrid) * 2* np.pi 
        density = rho_0(x)

        

    cumdensity = np.cumsum(density) * 2 * np.pi / ngrid
    total_mass = cumdensity[-1]
    r = np.exp( total_mass/4/np.pi/nu * x - cumdensity / 2 / nu )
    a_k = fftpack.fft(r)

    N = a_k.size
    k = np.concatenate((np.arange(0,N/2),np.arange(-N/2,0)))
    f = np.exp(nu * (1j*k-total_mass/4/np.pi/nu)**2 * t) * a_k
    r = fftpack.ifft(f)
    rx = fftpack.ifft(1j*k*f)
    rho = total_mass / 2 / np.pi - 2 * nu * np.real(rx / r)

    return x,rho








def quick_run(rho0: callable,  N = 32, Z = 1000, nu = .05, alpha = .1, dt = .002, Nt = 1000):
    """Runing the Burgers example.
    
    This is to be used for quick data generation without the need to save the solver.

    Args:
        rho: the initial density profile on [0,2pi)
        N: no. of teeth
        Z: resolution factor, i.e., no. of particles representing each unit of mass
        nu: viscosity
        alpha: fraction of space covered by teeth
        dt: time step
        Nt: number of time steps

    Returns:
        the grid of tooth (x of center), the density evaluated at the end of simulation

    """


    tsys=BurgersGapToothSystem(alpha=alpha,N=N,nu=nu)
    tsys.initialize(Z,rho0=rho0)
    print('no. of particles:'+str(tsys.particle_count))

    tsys.run_gap_tooth(dt=dt,Nt=Nt)
    tsys.compute_density()
    return tsys.tooth_center_x,tsys.rho

def parameter_dependence():
    """Plots dependence of solution based on no. of teeth and resolution factor.

    This is fig. 2 in the paper.
    """

    rho0=lambda x: 1-np.sin(x)/2

    # vary the grid size 
    Z0 = 100000
    Ns = [16,32,64]

    grid_N,sol_N=[],[]


    # for N in Ns:
    #     grid,sol=quick_run(rho0,N=N,Z=Z0)
    #     grid_N.append(grid),sol_N.append(sol)


    # vary the resolution factors 
    Zs = [1000,10000,100000]
    N0 = 128

    grid_Z,sol_Z=[],[]


    for Z in Zs:
        grid,sol=quick_run(rho0,N=N0,Z=Z)
        grid_Z.append(grid),sol_Z.append(sol)

    # # # da truth
    x,rho=compute_truth_Burgers(rho0,t=2)
    x[x>np.pi]=x[x>np.pi]-2*np.pi
    ix = np.argsort(x)
    x_truth,rho_truth = x[ix],rho[ix]

    np.savez('../data/Burgers_benchmark13',\
            grid_N=grid_N, sol_N=sol_N, 
            Ns=Ns,grid_truth=x_truth,sol_truth=rho_truth,\
            grid_Z=grid_Z, sol_Z=sol_Z, Zs=Zs, N0=N0,Z0=Z0)



if __name__ == "__main__":
    ttin = timeit.default_timer()
    parameter_dependence()
    print('whole run took {} seconds'.format(timeit.default_timer() - ttin))


