# Pyrus
Python implementation of redshift PDF close-pair analysis - specifically that of López-Sanjuan et al. (2014) - modified for stellar mass selected close-pairs in flux-limited surveys.

(Pyrus is the genus of trees from which Pears grow.)

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

# Set the intended stellar mass ration, often denoted mu
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
