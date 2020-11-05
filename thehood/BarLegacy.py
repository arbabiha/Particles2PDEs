# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Polynomial based models for finite differences and finite volumes.


Taken from Bar-Sinai et al., "Learning data-driven discretizations 
for partial differential equations", PNAS, 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum

import numpy as np
import scipy.special
import tensorflow as tf
from typing import Tuple



class GridOffset(enum.Enum):
  """Relationship between successive grids."""
  CENTERED = 1
  STAGGERED = 2


class Method(enum.Enum):
  """Discretization method."""
  FINITE_DIFFERENCES = 1
  FINITE_VOLUMES = 2


def regular_grid(
    grid_offset: GridOffset,
    derivative_order: int,
    accuracy_order: int = 1,
    dx: float = 1) -> np.ndarray:
  """Return the smallest grid on which finite differences can be calculated.

  Args:
    grid_offset: offset between input and output grids.
    derivative_order: integer derivative order to calculate.
    accuracy_order: integer order of polynomial accuracy to enforce. By default,
      only 1st order accuracy is guaranteed.
    dx: difference between grid points.

  Returns:
    1D numpy array giving positions at which to calculate finite differences.
  """
  min_grid_size = derivative_order + accuracy_order

  if grid_offset is GridOffset.CENTERED:
    max_offset = min_grid_size // 2  # 1 -> 0, 2 -> 1, 3 -> 1, 4 -> 2, ...
    grid = np.arange(-max_offset, max_offset + 1) * dx
  elif grid_offset is GridOffset.STAGGERED:
    max_offset = (min_grid_size + 1) // 2  # 1 -> 1, 2 -> 1, 3 -> 2, 4 -> 2, ...
    grid = (0.5 + np.arange(-max_offset, max_offset)) * dx
  else:
    raise ValueError('unexpected grid_offset: {}'.format(grid_offset))  # pylint: disable=g-doc-exception

  return grid


def constraints(
    grid: np.ndarray,
    method: Method,
    derivative_order: int,
    accuracy_order: int = None) -> Tuple[np.ndarray, np.ndarray]:
  """Setup the linear equation A @ c = b for finite difference coefficients.

  Args:
    grid: grid on which to calculate the finite difference stencil, relative
      to the point at which to approximate the derivative. The grid must be
      regular.
    method: discretization method.
    derivative_order: integer derivative order to approximate.
    accuracy_order: minimum accuracy order for the solution.

  Returns:
    Tuple of arrays `(A, b)` where `A` is 2D and `b` is 1D providing linear
    constraints. Any vector of finite difference coefficients `c` such that
    `A @ c = b` satisfies the requested accuracy order. The matrix `A` is
    guaranteed not to have more rows than columns.

  Raises:
    ValueError: if the linear constraints are not satisfiable.

  References:
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    Fornberg, Bengt (1988), "Generation of Finite Difference Formulas on
      Arbitrarily Spaced Grids", Mathematics of Computation, 51 (184): 699-706,
      doi:10.1090/S0025-5718-1988-0935077-0, ISSN 0025-5718.
  """
  if accuracy_order is None:
    # Use the highest order accuracy we can ensure in general. (In some cases,
    # e.g., centered finite differences, this solution actually has higher order
    # accuracy.)
    accuracy_order = grid.size - derivative_order

  if accuracy_order < 1:
    raise ValueError('cannot compute constriants with non-positive '
                     'accuracy_order: {}'.format(accuracy_order))

  deltas = np.unique(np.diff(grid))
  if (abs(deltas - deltas[0]) > 1e-8).any():
    raise ValueError('not a regular grid: {}'.format(deltas))
  delta = deltas[0]

  final_constraint = None
  zero_constraints = set()
  for m in range(accuracy_order + derivative_order):
    if method is Method.FINITE_DIFFERENCES:
      constraint = grid ** m
    elif method is Method.FINITE_VOLUMES:
      constraint = (1 / delta
                    * ((grid + delta/2) ** (m + 1)
                       - (grid - delta/2) ** (m + 1))
                    / (m + 1))
    else:
      raise ValueError('unexpected method: {}'.format(method))
    if m == derivative_order:
      final_constraint = constraint
    else:
      zero_constraints.add(tuple(constraint))

  assert final_constraint is not None

  num_constraints = len(zero_constraints) + 1
  if num_constraints > grid.size:
    raise ValueError('no valid {} stencil exists for derivative_order={} and '
                     'accuracy_order={} with grid={}'
                     .format(method, derivative_order, accuracy_order, grid))

  A = np.array(sorted(zero_constraints) + [final_constraint])  # pylint: disable=invalid-name

  b = np.zeros(A.shape[0])
  b[-1] = scipy.special.factorial(derivative_order)

  return A, b


def coefficients(
    grid: np.ndarray,
    method: Method,
    derivative_order: int) -> np.ndarray:
  """Calculate standard finite difference coefficients for the given grid.

  Args:
    grid: grid on which to calculate finite difference coefficients.
    method: discretization method.
    derivative_order: integer derivative order to approximate.

  Returns:
    NumPy array giving finite difference coefficients on the grid.
  """
  A, b = constraints(grid, method, derivative_order)  # pylint: disable=invalid-name
  return np.linalg.solve(A, b)


def zero_padded_coefficients(
    grid: np.ndarray,
    method: Method,
    derivative_order: int,
    padding: Tuple[int, int]) -> np.ndarray:
  """Calculate finite difference coefficients, but padded by zeros.

  These coefficients always hold on the given grid, but the result is guaranteed
  to have values on the left and right sides with indicated number of zeros.

  Args:
    grid: grid on which to calculate finite difference coefficients, which will
      be trimmed based on padding.
    method: discretization method.
    derivative_order: integer derivative order to approximate.
    padding: number of zeros to pad on the left and right sides of the result.

  Returns:
    NumPy array giving finite difference coefficients on the grid.
  """
  # note: need the "or" to avoid slicing with 0 as a right bound, because 0
  # is always interpretted as an offset from the start.
  pad_left, pad_right = padding
  trimmed_grid = grid[pad_left : (-pad_right or None)]
  trimmed_coefficients = coefficients(trimmed_grid, method, derivative_order)
  return np.pad(trimmed_coefficients, padding, mode='constant')


class PolynomialAccuracyLayer(object):
  """Layer to enforce polynomial accuracy for finite difference coefficients.

  Attributes:
    input_size: length of input vectors that are transformed into valid finite
      difference coefficients.
    bias: numpy array of shape (grid_size,) to which zero vectors are mapped.
    nullspace: numpy array of shape (input_size, output_size) representing the
      nullspace of the constraint matrix.
  """

  def __init__(self,
               grid: np.ndarray,
               method: Method,
               derivative_order: int,
               accuracy_order: int = 2,
               bias: np.ndarray = None,
               bias_zero_padding: Tuple[int, int] = (0, 0),
               out_scale: float = 1.0):
    """Constructor.

    Args:
      grid: grid on which to calculate finite difference coefficients.
      method: discretization method.
      derivative_order: integer derivative order to approximate.
      accuracy_order: integer order of polynomial accuracy to enforce.
      bias: np.ndarray of shape (grid_size,) to which zero-vectors will be
        mapped. Must satisfy polynomial accuracy to the requested order. By
        default, we calculate the standard finite difference coefficients for
        the given grid.
      bias_zero_padding: if a value for bias is not provided, ensure that the
        computed bias has the indicated number of zeros padded on the left and
        right sides. This is useful for initializing bias with upwinded
        coefficients.
      out_scale: desired multiplicative scaling on the outputs, relative to the
        bias.
    """
    A, b = constraints(grid, method, derivative_order, accuracy_order)  # pylint: disable=invalid-name
    print('polynomial accuracy order:'+str(accuracy_order))
    if bias is None:
      bias = zero_padded_coefficients(
          grid, method, derivative_order, bias_zero_padding)

    norm = np.linalg.norm(np.dot(A, bias) - b)
    if norm > 1e-8:
      raise ValueError('invalid bias, not in nullspace')  # pylint: disable=g-doc-exception

    # https://en.wikipedia.org/wiki/Kernel_(linear_algebra)#Nonhomogeneous_systems_of_linear_equations
    _, _, v = np.linalg.svd(A)
    input_size = A.shape[1] - A.shape[0]
    if not input_size:
      raise ValueError(  # pylint: disable=g-doc-exception
          'there is only one valid solution accurate to this order')

    # nullspace from the SVD is always normalized such that its singular values
    # are 1 or 0, which means it's actually independent of the grid spacing.
    nullspace = v[-input_size:]

    # ensure the nullspace is scaled comparably to the bias
    # TODO(shoyer): fix this for arbitrary spaced grids
    dx = grid[1] - grid[0]
    scaled_nullspace = nullspace * (out_scale / dx ** derivative_order)

    self.input_size = input_size
    self.grid_size = grid.size
    self.nullspace = scaled_nullspace
    self.bias = bias

  def apply(self, inputs: tf.Tensor) -> tf.Tensor:
    """Apply this layer to inputs.

    Args:
      inputs: float32 Tensor with dimensions [batch, x, input_size].

    Returns:
      Float32 Tensor with dimensions [batch, x, grid_size].
    """
    bias = self.bias.astype(np.float32)
    nullspace = tf.convert_to_tensor(self.nullspace.astype(np.float32))
    # print('nullspace shape='+str(nullspace.get_shape()))
    return bias + tf.einsum('bxi,ij->bxj', inputs, nullspace)



