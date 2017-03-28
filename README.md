# Rydberg_helium_Stark
Tools for calculating the Stark effect in Rydberg helium using the Numerov method.

## About
The code is based on:

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

And uses quantum defects from:

    High Precision Theory of Atomic Helium
    G. W. F. Drake
    Physica Scripta, Vol. T83 83-92 (1999)

The code hasn't been tested extensively. But it seems to give pretty accurate 
results for high-$n$ states ($n > 10$) and notably wrong ones for low-$n$.

## Install
Written using Python 3.6.0 :: Anaconda (64-bit).

Required packages are scipy, numpy, matplotlib, numba, and tqdm.

If you are using anaconda most of these are probably already installed, with 
the exceptions of numba and tqdm.

numba uses just-in-time compilation to significantly speed up python code.

```
conda install numba
```

tqdm provides a simple and efficient progress bar.

```
pip install tqdm
```

To install this project as a python package that can be imported into python 
as helium_stark,

```
git clone https://github.com/ad3ller/Rydberg_helium_Stark
cd ./Rydberg_helium_Stark
python setup.py develop
```

## Usage
See notebooks.
