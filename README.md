# Pyrus
Python implementation of redshift PDF close-pair analysis - specifically that of López-Sanjuan et al. (2014) - modified for stellar mass selected close-pairs in flux-limited surveys.

(Pyrus is the genus of trees from which Pears grow.)

## Requirements
Pyrus requires Python 2.7+ and the following modules:
* [numpy](http://www.numpy.org)
* [scipy](http://www.scipy.org)
* [astropy](http://www.astropy.org)

The following modules are optional if only performing simple, uncorrected pair/merger fraction analysis:
* [matplotlib](http://matplotlib.org)
* [photutils](http://photutils.readthedocs.org/en/latest/)
* [aplpy](https://aplpy.github.io/)

Pyrus, as presented here, is tested with the following module versions:
* Python (2.7.11), numpy (1.10.2), scipy (0.16.0), astropy (1.0.3), matplotlib (1.4.3), photutils (0.1), aplpy (1.0)

## Quick Start
Below is a quick start guide to using the script. This will calculate the major (mu = 4) pair/merger fraction for galaxies above a stellar mass of log(M*) > 11, at physical separations between 5 < r [kpc] < 30, between the redshift range 1.0 < z < 1.5. This example does not take into account spatial masks or mass completeness corrections. These are described in the functions below.

```python
from PyPairs import Pairs
import astropy.units as u
import numpy as np

# Set up pairs object
p = Pairs( zgrid, pzs, mzs,
           
           # Catalogue options
           photometry = './UDS_photometry.fits',
           catalogue_format = 'fits',
           idcol = 'id',
           racol = 'ra',
           deccol = 'dec',
           
           # Odds function arguments
           K = 0.05,
           dz = 0.001
           band = 'flux_tot_K',
           mags = False,
           abzp = 23.90,
           OSRlim = 0.3,
           mag_min = 16.0,
           mag_max = 25.0,
           mag_step = 0.2,
  )

# Set the intended stellar mass ratio, often denoted mu
p.massRatio = 4.
# Set the physical separation requirements
p.setSeparation( r_min = 5*u.kpc, r_max = 30*u.kpc )

# Provide initial indices of galaxies you wish to analyse
# In this case, use all galaxies already given to Pairs()
p.initialSample( np.arange(p.oddsCut.sum()) )

# Find initial pairs
p.findInitialPairs( z_min = 0.3, z_max = 2.5 )

# Make all necessary masks in mass, redshift, position
p.makeMasks( min_mass = 11.0,           # Minimum log(mass) for the primary sample
              max_mass = 12.0,          # Maximum log(mass) for any galaxy
              mass_ratio = p.massRatio
  )
  
# Retrieve the merger fraction for galaxies between z = 1.0 and z = 1.5
p.mergerFraction(1.0, 1.5)
fm = p.fm
```
## Class function descriptions

### `genCutOuts()`
Call this function after `mergerFraction()` to generate postage stamp cutouts of the identfied close pairs above some integrated pair probability. This functions required the [aplpy](https://aplpy.github.io/) python plotting library.

```python
p.genCutOuts(
        imagepath,                # path to the image used for cutouts
        outpath = './',           # path to dir in which figures will be saved
        Npair_lim = 0.01,         # limit of integrated weighted PPF above which to select pairs
        cutoutsize = 25*u.arcsec, # side length of cutout
        imghduid = 0,             # FITS HDU index corresponding to the image data
        outprefix = 'stamp',      # prefix of output file name
        z_mean = False            # redshift of pair required to draw search area
  )
```
Doing so will output single PDF files containing images such as this for each identified close-pair system.

![Alt](http://i.imgur.com/PNUT1Jr.jpg =300x "Example pair cutout plot")

### `bootstrapMergers()`
Call this function with or in place of `mergerFraction()` to perform a bootstrap error analysis of the pair/merger fraction. Random samples from the full galaxy list are drawn `nsamples` times and the analysis performed again. The mean and standard deviation in the returned pair/merger fractions are returned.

```python
p.bootstrapMergers(
        zmin = 1.0,         # Lower redshift of range in which to calculate fm
        zmax = 1.5,         # Higher redshift of range in which to calculate fm
        nsamples = 100      # Number of bootstrap samples (100+ suggested)
  )
```