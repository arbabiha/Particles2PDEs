"""
Library of neural nets for learning PDEs.

H. Arbabi, Aptil 2020, arbabiha@gmail.com.
"""




import numpy as np
import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Concatenate, Conv1D


import BarLegacy as BL

# tf.keras.backend.set_floatx('float32')
# tf.keras.backend.set_floatx('float64')

def Discretized_PDE_net(n_grid:int, n_stencil: int, n_conv=3, n_neurons=32):
    """A CNN model for learning Burgers PDE.

    E.g. the model with n_stencil=3 learns du_j/dt=f(u_j,u_{j-1},u_{j+1}).

    Args:
        n_grid: size of the (periodically-padded) input grid
        n_stencil: the kernel size of the first convolutional layer, 
            AND the stencil size of the finite difference
        n_conv: total number of convolutional layers
        n_neurons: size of hidden layers

    Returns:
        tf.keras.model that maps the field u (input) to u_t (output). 
    """
    u=tf.keras.Input(shape=(n_grid,),name="input_field")

    u_embeded=stencil_embedding(u,n_stencil)



    # 1st convolution+ activation layers
    clayer1= Conv1D(n_neurons,1,padding='valid',name='convolution_1')
    u_out = clayer1(u_embeded)
    u_out = tf.keras.activations.relu(u_out)

    for layer_ind in range(n_conv-2):
        clayer= Conv1D(n_neurons,1,padding='valid',name='convolution_'+str(layer_ind+2))
        u_out = clayer(u_out)
        u_out = tf.keras.activations.relu(u_out)
    
    clayer= Conv1D(1,1,padding='valid',name='convolution_'+str(n_conv))
    u_out = clayer(u_out)


    return tf.keras.Model(u,u_out)
    

def Functional_PDE_net(n_grid: int, dx: float, n_stencil: int, 
                       n_conv=3, n_neurons=32):
    """A functional model for learning PDEs.

    The model computes u_x,u_xx via finite difference and then
    at each x models u_t=f(u,u_x,u_xx) with trainable neural net.

    Args:
        n_grid: size of the (periodically-padded) input grid
        dx: uniform grid spacing
        n_stencil: the kernel size of the first convolutional layer, 
            AND the stencil size of the finite difference
        n_conv: total number of convolutional layers
        n_neurons: size of hidden layers

    Returns:
        tf.keras.model that maps the field u (input) to u_t (output). 
    """

    u=tf.keras.Input(shape=(n_grid,),name="input_field")

    # fixed layer for u_xx
    laplacian_layer = finite_diff_layer(dx,2,n_stencil)
    u_xx= laplacian_layer(u)

    # fixed layer for u_x
    laplacian_layer = finite_diff_layer(dx,1,n_stencil)
    u_x= laplacian_layer(u)

    # putting u,u_x and u_xx together
    us = tf.stack((u,u_x,u_xx),axis=-1,name='stack_u_ux_uxx')


    # 1st convolution+ activation layers
    clayer1= Conv1D(n_neurons,1,padding='valid',name='convolution_1')
    u_out = clayer1(us)
    u_out = tf.keras.activations.relu(u_out)

    for layer_ind in range(n_conv-2):
        clayer= Conv1D(n_neurons,1,padding='valid',name='convolution_'+str(layer_ind+2))
        u_out = clayer(u_out)
        u_out = tf.keras.activations.relu(u_out)
    
    clayer= Conv1D(1,1,padding='valid',name='convolution_'+str(n_conv))
    u_out = clayer(u_out)
    return tf.keras.Model(u,u_out)

def Burgers_PDE_greybox(n_grid: int, dx: float, nu: float, n_stencil: int, n_conv=3):
    """A CNN model for learning Burgers PDE only uux part.

    The model computes u_x,u_xx via finite difference and then
    at each x models u_t=f(u,u_x,u_xx) with trainable neural net.

    Args:
        n_grid: size of the (periodically-padded) input grid
        dx: uniform grid spacing
        n_stencil: the kernel size of the first convolutional layer, 
            AND the stencil size of the finite difference
        n_conv: total number of convolutional layers
        nu: viscosity

    Returns:
        tf.keras.model that maps the field u (input) to u_t (output). 
    """

    u=tf.keras.Input(shape=(n_grid,),name="input_field")

    

    # fixed layer for u_xx
    laplacian_layer = finite_diff_layer(dx,2,n_stencil)
    u_xx= laplacian_layer(u)
    # print(u_xx[...,tf.newaxis].shape)

    # fixed layer for u_x
    ux_layer = finite_diff_layer(dx,1,n_stencil)
    u_x= ux_layer(u)

    # putting u  u_x together
    us = tf.stack((u,u_x),axis=-1,name='stack_u_ux_uxx')


    # 1st convolution+ activation layers
    clayer1= Conv1D(32,1,padding='valid',name='convolution_1')
    u_out = clayer1(us)
    u_out = tf.keras.activations.relu(u_out)

    for layer_ind in range(n_conv-2):
        clayer= Conv1D(32,1,padding='valid',name='convolution_'+str(layer_ind+2))
        u_out = clayer(u_out)
        u_out = tf.keras.activations.relu(u_out)
    
    clayer= Conv1D(1,1,padding='valid',name='convolution_'+str(n_conv))
    u_out = clayer(u_out) + nu * u_xx[...,tf.newaxis]
    

    return tf.keras.Model(u,u_out)





def Burgers_PDE(n_grid: int, dx: float, n_stencil: int, nu: float):
    """A standard discretization of Burgers.

    The model computes u_t= - u*u_x+ nu*u_xx.

    Args:
        n_grid: size of the (periodically-padded) input grid
        dx: uniform grid spacing
        n_stencil: the kernel size of the first convolutional layer, 
            AND the stencil size of the finite difference
        ns: resolutaion rate of interpolation
        nu: viscosity

    Returns:
        tf.keras.model that maps the field u (input) to u_t (output). 
    """

    u=tf.keras.Input(shape=(n_grid,),name="input_field")

    # fixed layer for u_xx
    laplacian_layer = finite_diff_layer(dx,2,n_stencil)
    u_xx= laplacian_layer(u)

    # fixed layer for u_x
    laplacian_layer = finite_diff_layer(dx,1,n_stencil)
    u_x= laplacian_layer(u)

    # putting u,u_x and u_xx together
    u_t =- u*u_x + nu * u_xx 


    return tf.keras.Model(u,u_t)
 





class finite_diff_layer(tf.keras.layers.Layer):
    """A layer of frozen finite difference on uniform periodic grid."""

    def __init__(self, dx: float, derivative_order: int, stencil_size: int):
        """Constructor.

        Args:
            dx: spacing between grid points
            derivative_order: larger than 0
            stencil_size: at this point we only accept odd numbers 
        """
        super(finite_diff_layer, self).__init__()
        assert stencil_size % 2 ==1, "I accept only odd stencil size"
        self.stencil_size=stencil_size

        int_grid= np.arange(-stencil_size//2+1,stencil_size//2+1,1)
        local_grid = int_grid* dx   # local position of points

        damethod=BL.Method.FINITE_DIFFERENCES
        standard_coeffs= BL.coefficients(local_grid,damethod,derivative_order)
        
        # self.coeffs=tf.constant(standard_coeffs,dtype=tf.float64,name='df_coeffs_O'+str(derivative_order))
        self.coeffs=tf.constant(standard_coeffs,dtype=tf.float32,name='df_coeffs_O'+str(derivative_order))

    def build(self, input_shape):
        pass

    def call(self,u):
        u_embeded=stencil_embedding(u, self.stencil_size)
        return tf.einsum('s,bxs->bx', self.coeffs, u_embeded)









def stencil_embedding(inputs:tf.Tensor,stencil_width:int)-> tf.Tensor:
    """Embedding the input data with the size of stencil.

    Args:
        inputs: values of the field on a periodic 1d grid, shape=(...,x)
        stencil_width: width of the stencil

    Returns:
        tensor of shape (...,x,stencil_width) the values of stencil nodes
    """
    if stencil_width % 2 ==1:
        npad = stencil_width//2
    else:
        raise NotImplementedError('only accept odd stencil size')


    padded_inputs=tf.concat([inputs[:,-npad:],inputs,inputs[:,:npad]],axis=-1)

    # we add (y,depth) dimension to fake an image
    embedded=tf.image.extract_patches(padded_inputs[...,tf.newaxis,tf.newaxis],
                                      sizes=[1,stencil_width,1,1],
                                      strides=[1, 1, 1, 1],
                                      rates=[1, 1, 1, 1],
                                      padding='VALID',name='stencil_embeded')


    return tf.squeeze(embedded,axis=-2,name='squeeeze_me')  # remove (y,) dimension




def fdm_2nd_der(nx:int,dx:float,ns:int):
    """Computing the finite-diff ccoefficients for 2nd derivative.
    
    Args:
        nx: number of grid points
        dx: (uniform) grid spacing
        ns: (odd) stencil width
        
    Returns:
        alpha: tf.tensor of shape (nx,s) where alpha[ix,:] is the
            finite diff coefficients for estimating d2u/dx2 with O(h2) error.
    """
    
    assert ns % 2 ==1, "I accept only odd stencil width"

    int_grid= np.arange(-ns//2+1,ns//2+1,1)
    local_grid = int_grid* dx   # local position of points


    damethod=BL.Method.FINITE_DIFFERENCES

    a= BL.coefficients(local_grid,damethod,2)

    return tf.constant(a,dtype=tf.float32,name='a_xx')





if __name__=='__main__':

    print('pass?')



