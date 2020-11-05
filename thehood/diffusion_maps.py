"""A simple implementation of diffusion maps and geometric harmonics.

From Coifman, Ronald R., and Stéphane Lafon. 'Diffusion maps' 
Applied and computational harmonic analysis 21.1 (2006): 5-30.
and "Geometric harmonics: a novel tool 
for multiscale out-of-sample extension of empirical functions." Applied and 
Computational Harmonic Analysis 21.1 (2006): 31-52.


This code is not optimized for speed, but it can handel arbitrary distance function. 
For a more optimized implementation see
https://github.com/jmbr/diffusion-maps .



H. Arbabi, arbabiha@gmail.com.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import scipy.sparse


class Manifold_Model:
    """A simple object for computing diffusion maps and geometric harmonics."""

    def __init__(self, X, distance_function, alpha = 1 ):
        """Constructor.

        Args:
            X: data points (observaions x features)
            distance function: a callable that takes two data points and spits out their distance
            alpha: operator parameter (1:Laplace-Beltrami, 1/2:Fokker-Planck, 0: Normalized Graph Laplacian) 
        """
        self.X =X
        self.alpha=alpha
        self.distance_function= distance_function
        self.distance_matrix = squareform(pdist(self.X,metric=self.distance_function)) 

    def compute_diffusion_maps(self, epsilon=None, num_pairs = 10):
        """Computing the first 10 diffusion maps.

        Args:
            epsilon: the kernel width, if None the median of distance matrices is used
            num_pairs: number of eigen-vector -values for output
        
        Returns:
            matrix of diffusion map coordiantes and their eigenvalues
        """

        if epsilon is None:
            epsilon=np.median(self.distance_matrix)
        
        self.epsilon=epsilon

        self.W=np.exp(-(self.distance_matrix**2)/self.epsilon)

        d=np.sum(self.W, axis=1)**self.alpha
        d_inv = 1.0/d
        W_bar=d_inv[:,np.newaxis] * (self.W * d_inv[np.newaxis,:])
        
        row_sums = W_bar.sum(axis=1)
        W_hat = W_bar / row_sums[:, np.newaxis]
        
        eigenvalues, v_unsrtd = np.linalg.eig(W_hat)
        id_eigvalue=np.argsort(eigenvalues)[::-1]

        self.phis = np.real(v_unsrtd[:,id_eigvalue][:,:num_pairs])
        self.eigenvalues = np.real(eigenvalues[id_eigvalue][:num_pairs])

        return self.phis,self.eigenvalues

    def compute_geometric_harmonics(self,dim=None,epsilon_interp=None):
        """Computing the basis for geometric harmonic interpolation."""
        
        if epsilon_interp is None:
            epsilon_interp= 50 * np.median(self.distance_matrix)
            print('eps interp set to {:.1e}'.format(epsilon_interp))
        self.epsilon_interp = epsilon_interp

        W=np.exp(-(self.distance_matrix**2)/self.epsilon_interp)

        eigenvalues, v_unsrtd = np.linalg.eigh(W)
        id_eigvalue=np.argsort(eigenvalues)[::-1]

        if dim is None:
            ratio = eigenvalues/eigenvalues[0]
            dim = ratio[ratio>1e-6].shape[0]
            print('cut-off dim is '+str(dim))

        self.psis = v_unsrtd[:,id_eigvalue][:,:dim]
        self.psi_eigvales = eigenvalues[id_eigvalue][:dim]


    def interpolate_geometric_harmonics(self,X, f):
        """Using the Geometric Harmonics to interpolate the diffusion map coordinates.

        Args:
            X: new data points
            f: the function (values on self.X) to be interpolated

        Returns:
            the value of diffusion map coordinates for new data points.
        """
        if not hasattr(self, 'psis'):
            print('computing the basis ...')
            self.compute_geometric_harmonics()
        
        new_distance = cdist(X,self.X,metric=self.distance_function)
        W=np.exp(-(new_distance**2)/self.epsilon_interp) 

        a = self.psis.T @ f

        extended_psis = W @ (self.psis / self.psi_eigvales[np.newaxis,:]) 

        extended_f = extended_psis @ a

        return extended_f







def BasicDiffusionMaps(W,alpha=1):
    """Computing first 10 diffusion maps.
    
    Args:
        W: affinity matrix of data points
        alpha: operator parameter (1:Laplace-Beltrami, 1/2:Fokker-Planck, 0: Normalized Graph Laplacian)
    
    Returns:
        phis: the vector of diffusion maps
        lams: the corresponding eigenvalues
    """
    
    d=np.sum(W, axis=1)**alpha
    d_inv = 1.0/d
    W_bar=d_inv[:,np.newaxis] * (W * d_inv[np.newaxis,:])
    

    row_sums = W_bar.sum(axis=1)
    W_hat = W_bar / row_sums[:, np.newaxis]
    

    eigenvalues, v_unsrtd = np.linalg.eig(W_hat)
    id_eigvalue=np.argsort(eigenvalues)[::-1]
    phis = v_unsrtd[:,id_eigvalue]
    eigenvalues = eigenvalues[id_eigvalue]

    return phis[:,:10],eigenvalues[:10]



