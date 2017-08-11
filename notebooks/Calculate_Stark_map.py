
# coding: utf-8

# # Calculate Stark map for triplet helium

# In[1]:

import os
import numpy as np
#import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
from starkhelium import *
from tqdm import trange, tqdm
from scipy.constants import h, hbar, c, alpha, m_e, e, epsilon_0, atomic_mass, pi, physical_constants
a_0 = physical_constants['Bohr radius'][0]
En_h = alpha**2.0 * m_e * c**2.0;
scl = c*10**-9 * En_h /(h * c);


# In[2]:

# User variables
  # Whether to import and save the Stark interaction matrix
IMPORT_MAT_S, CALC_MAT_S, SAVE_MAT_S = False, True, False
  # Whether to import and save the Diamagnetic interaction matrix
IMPORT_MAT_D, CALC_MAT_D, SAVE_MAT_D = False, True, False
  # Whether to save the Stark map data
SAVE_STARK_MAP_DATA = True


# In[3]:

# Helper functions
def getDataDir():
    # Create data directoy if it doesn't exist
    directory = os.path.join(".", "data")
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory
def getImagesDir():
    # Create data directoy if it doesn't exist
    directory = os.path.join(".", "images")
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def getFilenameInt(name, nmin, nmax, step_params):
    return name + "IntMatrix_n_" + str(nmin) + "-" + str(nmax) +     "_step_" + str(step_params[0]).replace('.', '-') + "_" + str(step_params[1]).replace('.', '-') +     "_" + str(step_params[2]).replace('.', '-')
    
def getFilenameStarkMap(nmin, nmax, step_params, field, B_z):
    return "StarkMapData_n_" + str(nmin) + "-" + str(nmax) +     "_step_" + str(step_params[0]).replace('.', '-') + "_" + str(step_params[1]).replace('.', '-') +     "_" + str(step_params[2]).replace('.', '-') +     "_E_" + str(np.min(field)).replace('.', '-') + "_" + str(np.max(field)).replace('.', '-') + "_" + str(len(field)) +     "_B_" + str(B_z*1E3).replace('.', '-') + '.npy'

def saveIntMat(mat_I, name, nmin, nmax, step_params):
    # Create fileaname for interaction map
    filename = getFilenameInt(name, nmin, nmax, step_params)
    # Get data directoy, create it if it doesn't exist
    directory = getDataDir()
    # Save interaction matrix to file
    fileout = os.path.join(directory, filename)
    np.save(fileout, mat_I)
    
def importIntMat(name, nmin, nmax, step_params):
    filename = getFilenameInt(name, nmin, nmax, step_params) + ".npy"
    directory = getDataDir()
    filein = os.path.join(directory, filename)
    try:
        return np.load(filein)
    except:
        raise


# In[4]:

# quantum numbers
nmin = 10
nmax = 12
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
step_low = 0.005
step_high = 0.005
step_expo = 1.0
step_params = [step_low, step_high, step_expo]

if IMPORT_MAT_S: mat_S = importIntMat('Stark', nmin, nmax, step_params)
elif CALC_MAT_S: 
    mat_S = stark_matrix(n_vals, neff, L_vals, m_vals, field_orientation, step_params=step_params)
    if SAVE_MAT_S: saveIntMat(mat_S, 'Stark', nmin, nmax, step_params)
            
if IMPORT_MAT_D: mat_D = importIntMat('Diamagnetic', nmin, nmax, step_params)
elif CALC_MAT_D: 
    mat_D = diamagnetic_matrix(n_vals, neff, L_vals, m_vals, step_params=step_params)
    if SAVE_MAT_D: saveIntMat(mat_D, 'Diamagnetic', nmin, nmax, step_params)


# In[5]:

# specify the electric field
field = np.linspace(1.0, 2.0, 11) # V /cm
field_au = field * 100 / (En_h_He/(e*a_0_He)) 
# specify the magnetic field (in Telsa)
B_z = 1.6154E-3
# (in atomic units)
B_z_au = B_z / (hbar/(e*a_0_He**2))
# Zeeman interaction Hamiltonian
H_Z = np.diag(E_zeeman(m_vals, B_z_au))
# Diamagnetic interaction Hamiltonian
if IMPORT_MAT_D or CALC_MAT_D:
    H_D = mat_D * B_z_au**2 * (e**2/(8*m_e))
else:
    H_D = 0

# diagonalise for each field
stark_map = stark_map(H_0, mat_S, field_au, H_Z=H_Z, H_D=H_D)
if SAVE_STARK_MAP_DATA:
    # Save Stark map to file
    filename = getFilenameStarkMap(nmin, nmax, step_params, field, B_z)
    fileout = os.path.join(getDataDir(), filename)
    np.save(fileout, stark_map)


# In[ ]:



