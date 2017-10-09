
# coding: utf-8

# # Calculate Stark map for triplet helium

# In[3]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from starkhelium import *
from tqdm import trange, tqdm
import os

au_to_ghz = 10**-9 * E_h /h
scl = au_to_ghz
au_to_cm = E_h / (100 * h * c)


# In[53]:

# User variables
  # Whether to import and save the Stark interaction matrix
IMPORT_MAT_S, CALC_MAT_S, SAVE_MAT_S = False, True, True
  # Whether to import and save the Diamagnetic interaction matrix
IMPORT_MAT_D, CALC_MAT_D, SAVE_MAT_D = False, True, True
  # Whether to save the eigenvalues and eigenvectors
SAVE_EIG_VALS = True
SAVE_EIG_VECS = True


# In[54]:

# Helper functions
def getDataDir():
    # Create data directoy if it doesn't exist
    directory = os.path.join(".", "data")
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory
def getImagesDir():
    # Create data directoy if it doesn't exist
    directory = os.path.join(".", "figures")
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def getFilenameInt(name, nmin, nmax):
    return name + "IntMatrix_n_" + str(nmin) + "-" + str(nmax)
    
def getFilenameEig(nmin, nmax, field, B_z):
    return "StarkMapData_n_" + str(nmin) + "-" + str(nmax) +     "_E_" + str(np.min(field)).replace('.', '-') + "_" + str(np.max(field)).replace('.', '-') + "_" + str(len(field)) +     "_B_" + str(B_z*1E3).replace('.', '-')

def saveIntMat(mat_I, name, nmin, nmax):
    # Create fileaname for interaction map
    filename = getFilenameInt(name, nmin, nmax) + '.npy'
    # Get data directoy, create it if it doesn't exist
    directory = getDataDir()
    # Save interaction matrix to file
    fileout = os.path.join(directory, filename)
    np.save(fileout, mat_I)
    
def importIntMat(name, nmin, nmax):
    filename = getFilenameInt(name, nmin, nmax) + ".npy"
    directory = getDataDir()
    filein = os.path.join(directory, filename)
    try:
        return np.load(filein)
    except:
        raise


# In[55]:

# quantum numbers
nmin = 4
nmax = 6
S = 1
n_vals, L_vals, m_vals = get_nlm_vals(nmin, nmax)
J_vals = get_J_vals(S, L_vals, 1)
# quantum defects
neff = n_vals - get_qd(S, n_vals, L_vals, J_vals)
# energy levels
En = W_n(S, n_vals, L_vals, J_vals)
#En = En_0(neff)
# field orientation
field_orientation = 'crossed'
# field-free Hamiltonian
H_0 = np.diag(En)
# Numerov step size
step_params = ['flat', 0.005]

if IMPORT_MAT_S: mat_S = importIntMat('Stark', nmin, nmax)
elif CALC_MAT_S: 
    mat_S = stark_matrix(neff, L_vals, m_vals, field_orientation, step_params=step_params)
    if SAVE_MAT_S: 
        saveIntMat(mat_S, 'Stark', nmin, nmax)
        #del mat_S

if IMPORT_MAT_D: mat_D = importIntMat('Diamagnetic', nmin, nmax)
elif CALC_MAT_D: 
    mat_D = diamagnetic_matrix(neff, L_vals, m_vals, step_params=step_params)
    if SAVE_MAT_D: 
        saveIntMat(mat_D, 'Diamagnetic', nmin, nmax)
        #del mat_D


# In[56]:

# specify the electric field
field = np.linspace(0.0, 10, 11) # V /cm
field_au = field * 100 / (En_h_He/(e*a_0_He)) 
# specify the magnetic field (in Telsa)
B_z = 1.0E-1
# (in atomic units)
B_z_au = B_z / (hbar/(e*a_0_He**2))
# Zeeman interaction Hamiltonian
H_Z = np.diag(E_zeeman(m_vals, B_z_au))
# Diamagnetic interaction Hamiltonian
if IMPORT_MAT_D or CALC_MAT_D:
    H_D = mat_D * (B_z_au**2)/8
else:
    H_D = 0

# diagonalise for each field
if SAVE_EIG_VECS:
    eig_vals, eig_vecs = stark_map_vec(H_0, mat_S, field_au, H_Z=H_Z, H_D=H_D)
else:
    eig_vals = stark_map(H_0, mat_S, field_au, H_Z=H_Z, H_D=H_D)

if SAVE_EIG_VALS:
    # Save eigenvalues to file
    filename = getFilenameEig(nmin, nmax, field, B_z) + "_eigval"
    fileout = os.path.join(getDataDir(), filename)
    np.save(fileout, eig_vals)
    
if SAVE_EIG_VECS:
    # Save eigenvectors to file
    filename = getFilenameEig(nmin, nmax, field, B_z) + "_eigvec"
    fileout = os.path.join(getDataDir(), filename)
    np.save(fileout, eig_vecs)


# In[ ]:



