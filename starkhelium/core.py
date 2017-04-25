#! python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:30:52 2017

@author: Adam Deller, UNIVERSITY COLLEGE LONDON.

Tools for calculating the Stark effect in Rydberg helium
using the Numerov method.

Based on:

    Stark structure of the Rydberg states of alkali-metal atoms
        M. L. Zimmerman et al. Phys. Rev. A, 20 2251 (1979)
        http://dx.doi.org/10.1103/PhysRevA.20.2251

    Rydberg atom diamagnetism
        M. M. KASH, PhD thesis, Massachusetts Institute of Technology.
        Dept. of Physics. (1988)
        http://hdl.handle.net/1721.1/14367

    Cold atoms and molecules by Zeeman deceleration and Rydberg-
    Stark deceleration
        S. D. Hogan, Habilitationsschrift, ETH Zurich (2012)
        http://dx.doi.org/10.3929/ethz-a-007577485

"""
from __future__ import print_function, division
from math import ceil, log, exp
import numpy as np
from numba import jit
from tqdm import trange
from scipy.constants import h, hbar, c, alpha, m_e, e, epsilon_0, atomic_mass, pi
from .drake1999 import quantum_defects

# constants
En_h = alpha**2.0 * m_e * c**2.0
a_0 = hbar/ (m_e * c * alpha)
mass_He = 4.002602 * atomic_mass
Z = 2
mu_me = (mass_He - m_e) / mass_He
mu_M = m_e / mass_He

@jit
def get_nl_vals(nmin, nmax, m):
    """ n and l vals for each matrix column, using range n_min to n_max.
    """
    n_rng = np.arange(nmin, nmax + 1)
    n_vals = np.array([], dtype='int32')
    l_vals = np.array([], dtype='int32')
    for n in n_rng:
        l_rng = np.arange(m, n)
        n_vals = np.append(n_vals, np.array(np.zeros_like(l_rng) + n))
        l_vals = np.append(l_vals, l_rng)
    return n_vals, l_vals

@jit
def get_J_vals(S, L_vals, diff):
    """ J = L + diff; unless L == 0, in which case J = S.
    """
    J_vals = L_vals + diff
    J_vals[L_vals == 0] = S
    return J_vals

@jit
def get_triplet_nLJ(nmin, nmax, m):
    """ n and L and J vals for each matrix column, using range n_min to n_max.
    """
    n_rng = np.arange(nmin, nmax + 1, dtype='int32')
    n_vals = np.array([], dtype='int32')
    L_vals = np.array([], dtype='int32')
    J_vals = np.array([], dtype='int32')
    for n in n_rng:
        l_rng = np.arange(m, n, dtype='int32')
        for l in l_rng:
            if l == 0:
                n_vals = np.append(n_vals, n)
                L_vals = np.append(L_vals, 0)
                J_vals = np.append(J_vals, 1)
            else:
                n_vals = np.append(n_vals, np.zeros(3, dtype='int32') + n)
                L_vals = np.append(L_vals, np.zeros(3, dtype='int32') + l)
                J_vals = np.append(J_vals, np.arange(l - 1, l + 2, dtype='int32'))
    return n_vals, L_vals, J_vals

@jit
def get_qd(S, n_vals, L_vals, J_vals):
    """ Calculate quantum defects.
    """
    iterations = int(10)
    num_cols = len(n_vals)
    qd = np.zeros(num_cols)
    for i in range(num_cols):
        n = n_vals[i]
        L = L_vals[i]
        J = J_vals[i]
        if L in quantum_defects[S]:
            if J in quantum_defects[S][L]:
                delta = quantum_defects[S][L][J]
                # calculate quantum defects
                qd_i = delta[0]
                for rep in range(iterations):
                    # repeat to get convergence
                    m = n - qd_i
                    defect = delta[0]
                    for j, d in enumerate(delta[1:]):
                        defect = defect + d*m**(-2.0*(j + 1))
                    qd_i = defect
                    qd[i] = defect
            else:
                qd[i] = np.nan
        else:
            qd[i] = 0.0
    return qd

@jit
def En_0(neff):
    """ Field-free energy. Ignores extra correction terms.

        -- atomic units --
    """
    energy = np.array([])
    for n in neff:
        en = -0.5 * n**-2.0
        energy = np.append(energy, en)
    return energy * mu_me

@jit
def W_n(S, n_vals, L_vals, J_vals):
    """ Field-free energy. Includes extra correction terms.

        -- atomic units --
    """
    neff = n_vals - get_qd(S, n_vals, L_vals, J_vals)
    energy = np.array([])
    for i, n in enumerate(n_vals):
        en = -0.5 * (neff[i]**-2.0 - 3.0 * alpha**2.0 / (4.0 * n**4.0) + \
                     mu_M**2.0 * ((1.0 + (5.0 / 6.0) * (alpha * Z)**2.0)/ n**2.0))
        energy = np.append(energy, en)
    return energy * mu_me

@jit
def wf_numerov(n, l, nmax, step=0.005, rmin=1.0):
    """ Use the Numerov method to find the wavefunction for state n*, l, where
        n* = n - delta.

        nmax ensures that wavefunctions from different values of n can be aligned.
    """
    W1 = -0.5 * n**-2.0
    W2 = (l + 0.5)**2.0
    rmax = 2 * nmax * (nmax + 15)
    r_in = n**2.0 - n * (n**2.0 - l*(l + 1.0))**0.5
    step_sq = step**2.0
    # ensure wf arrays will align using nmax
    if n == nmax:
        i = 0
        r_sub2 = rmax
    else:
        i = int(ceil(log(rmax / (2 * n * (n + 15))) / step))
        r_sub2 = rmax * exp(-i*step)
    i += 1

    # initialise
    r_sub1 = rmax * exp(-i*step)
    rvals = [r_sub2, r_sub1]
    g_sub2 = 2.0 * r_sub2**2.0 * (-1.0 / r_sub2 - W1) + W2
    g_sub1 = 2.0 * r_sub1**2.0 * (-1.0 / r_sub1 - W1) + W2
    y_sub2 = 1e-10
    y_sub1 = y_sub2 * (1.0 + step * g_sub2**0.5)
    yvals = [y_sub2, y_sub1]

    # Numerov method
    i += 1
    r = r_sub1
    while r >= rmin:
        ## next step
        r = rmax * exp(-i*step)
        g = 2.0 * r**2.0 * (-1.0 / r - W1) + W2
        y = (y_sub2 * (g_sub2 - (12.0 / step_sq)) + y_sub1 * \
            (10.0 * g_sub1 + (24.0 / step_sq))) / ((12.0 / step_sq) - g)

        ## check for divergence
        if r < r_in:
            dy = abs((y - y_sub1) / y_sub1)
            dr = (r**(-l-1) - r_sub1**(-l-1)) / r_sub1**(-l-1)
            if dy > dr:
                break

        ## store vals
        rvals.append(r)
        yvals.append(y)

        ## next iteration
        r_sub1 = r
        g_sub2 = g_sub1
        g_sub1 = g
        y_sub2 = y_sub1
        y_sub1 = y
        i += 1

    rvals = np.array(rvals)
    yvals = np.array(yvals)
    # normalisation
    yvals = yvals * (np.sum((yvals**2.0) * (rvals**2.0)))**-0.5
    return rvals, yvals

@jit
def find_first(arr, val):
    """ Index of the first occurence of val in arr.
    """
    i = 0
    while i < len(arr):
        if val == arr[i]:
            return i
        i += 1
    raise Exception('val not found in arr')

@jit
def find_last(arr, val):
    """ Index of the last occurence of val in arr.
    """
    i = len(arr) - 1
    while i > 0:
        if val == arr[i]:
            return i
        i -= 1
    raise Exception('val not found in arr')

@jit
def wf_align(r1, y1, r2, y2):
    """ Align two lists pairs (r, y) on r, assuming r array values overlap
        except at head and tail, and that arrays are reverse sorted.
    """
    if r1[0] != r2[0]:
        # trim front end
        if r1[0] > r2[0]:
            idx = find_first(r1, r2[0])
            r1 = r1[idx:]
            y1 = y1[idx:]
        else:
            idx = find_first(r2, r1[0])
            r2 = r2[idx:]
            y2 = y2[idx:]
    if r1[-1] != r2[-1]:
        # trim back end
        if r1[-1] < r2[-1]:
            idx = find_last(r1, r2[-1])
            r1 = r1[:idx + 1]
            y1 = y1[:idx + 1]
        else:
            idx = find_last(r2, r1[-1])
            r2 = r2[:idx + 1]
            y2 = y2[:idx + 1]
    if r1[0] == r2[0] and r1[-1] == r2[-1] and len(r1) == len(r2):
        return r1, y1, r2, y2
    else:
        raise Exception("Failed to align wavefunctions.")

@jit
def wf_overlap(r1, y1, r2, y2, p=1.0):
    """ Find the overlap between two radial wavefunctions (r, y).
    """
    r1, y1, r2, y2 = wf_align(r1, y1, r2, y2)
    return np.sum(y1 * y2 * r1**(2.0 + p))

@jit(cache=True)
def rad_overlap(n1, l1, n2, l2, p=1.0):
    """ Radial overlap for state n1, l1 and n2 l2.
    """
    nmax = max(n1, n2)
    r1, y1 = wf_numerov(n1, l1, nmax)
    r2, y2 = wf_numerov(n2, l2, nmax)
    return abs(wf_overlap(r1, y1, r2, y2, p))

@jit
def ang_overlap(l1, l2, m):
    """ Angular overlap <l1, m| cos(theta) |l2, m>.
    """
    dl = l2 - l1
    if dl == -1:
        return ((l1**2 - m**2) / ((2*l1 + 1) * (2*l1 - 1)))**0.5
    elif dl == +1:
        return (((l1 + 1)**2 - m**2) / ((2*l1 + 3) * (2*l1 + 1)))**0.5
    else:
        return 0.0

@jit
def stark_int(n_1, l_1, n_2, l_2, m):
    """ Stark interaction between states |n1, l1, m> and |n2, l2, m>.
    """
    if abs(l_1 - l_2) == 1:
        # Stark interaction
        return ang_overlap(l_1, l_2, m) * rad_overlap(n_1, l_1, n_2, l_2)
    else:
        return 0.0

@jit
def stark_matrix(neff_vals, l_vals, m):
    """ Stark interaction matrix.
    """
    num_cols = len(neff_vals)
    mat_I = np.zeros([num_cols, num_cols])
    for i in trange(num_cols, desc="calculate Stark terms"):
        n_1 = neff_vals[i]
        l_1 = l_vals[i]
        for j in range(i + 1, num_cols):
            n_2 = neff_vals[j]
            l_2 = l_vals[j]
            mat_I[i][j] = stark_int(n_1, l_1, n_2, l_2, m)
            # assume matrix is symmetric
            mat_I[j][i] = mat_I[i][j]
    return mat_I

def eig_sort(w, v):
    """ sort eignenvalues and eigenvectors by eigenvalue.
    """
    ids = np.argsort(w)
    return w[ids], v[:, ids]

@jit
def stark_map(H_0, mat_S, field):
    """ Calculate the eigenvalues for H_0 + H_S, where

         - H_0 is the field-free Hamiltonian,
         - H_S = D * F * mat_S
         - D is the electric dipole moment,
         - F is each value of the electric field (a.u.),
         - mat_S is the Stark interaction matrix.

        return eig_val [array.shape(num_fields, num_states)]
    """
    num_fields = len(field)
    num_cols = np.shape(H_0)[0]
    # initialise output arrays
    eig_val = np.empty((num_fields, num_cols), dtype=float)
    # loop over field values
    for i in trange(num_fields, desc="diagonalise Hamiltonian"):
        F = field[i]
        H_S = F * mat_S / mu_me
        # diagonalise, assuming matrix is Hermitian.
        eig_val[i] = np.linalg.eigh(H_0 + H_S)[0]
    return eig_val

@jit
def stark_map_vec(H_0, mat_S, field):
    """ Calculate eigenvalues and eigenvectors for H_0 + H_S. See stark_map().

        return eig_val [array.shape(num_fields, num_states)],
               eig_vec [array.shape(num_fields, num_states, num_states)]

         ------------------------------------------------------------------
         Note: A significant amount of memory may be required to hold the
               array of eigenvectors.
         ------------------------------------------------------------------
    """
    num_fields = len(field)
    num_cols = np.shape(H_0)[0]
    # initialise output arrays
    eig_val = np.empty((num_fields, num_cols), dtype=float)
    eig_vec = np.empty((num_fields, num_cols, num_cols), dtype=float)
    # loop over field values
    for i in trange(num_fields, desc="diagonalise Hamiltonian"):
        F = field[i]
        H_S = F * mat_S / mu_me
        # diagonalise, assuming matrix is Hermitian.
        eig_val[i], eig_vec[i] = np.linalg.eigh(H_0 + H_S)
    return eig_val, eig_vec