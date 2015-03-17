import time
# Matplotlib, numpy
import matplotlib.pyplot as plt, numpy as np
# Astropy
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
# Scipy
from scipy.spatial import cKDTree
from scipy.integrate import simps, trapz, cumtrapz, quad
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.constants import golden
# Photutils
from photutils import SkyCircularAnnulus, CircularAnnulus, aperture_photometry

class Pairs(object):
    """ Main class for photometric pair-count analysis
    
    Implementation of the Lopez-Sanjuan et al. (2014) redshift PDF pair-count method
    for photometric redshifts from EAZY (or other) and stellar mass estimates using
    Duncan et al. (2014) outputs.
        
    """

    def __init__(self, z, redshift_cube, mass_cube, band, z_best=False, 
                 photometry=False, catalog_format = 'fits',
                 idcol = 'ID', racol = 'RA', deccol = 'DEC', cosmology = False,
                 K = 0.05, dz = 0.01, OSRlim = 0.3, mags=True, abzp = False,
                 mag_min = 15, mag_max = 30.5, mag_step = 0.25, SNR = False, banderr='ERR',
                 maskpath=False):
        """ Load and format appropriately the necessary data for pair-count calculations
        
        Args:
            z (numpy.ndarray): Redshift steps - 1-d array of len(N)
            redshift_cube (numpy.ndarray): Photometric redshift probability
                distribution functions for galaxy sample. 2-d array of shape(M,N)
            mass_cube (numpy.ndarray): Stellar mass as a function of redshift for 
                galaxy sample. 2-d array of shape(M,N)

            Catalog Arguments:
                photometry (str): Path to photometry catalog
                catalog_format (str): Catalog format, e.g. 'fits' or 'ascii'
                idcol (str): Column name for galaxy IDs
                racol (str): Column name for galaxy RAs
                deccol (str): Column name for galaxy DECs
                maskpath (str): Path to mask file for survey
          
            Cosmology Arguments:
                cosmology (astropy.cosmology): Astropy.cosmology object
                
                    if cosmology == None: FlatLambdaCDM with H0=70, Om0=0.3 
                        assumed.

            Odds Function Arguments:
                K (float): parameter detailing limits to integrate around, such that integral
                    performed over z_best +/- K(1+z).
                dz (float): redshift range step to interpolate to.
                OSRlim (float): selected limit to calculate the odds sampling rate (OSR).
                band (str): column name in photometry catalogue one wishes to parametrise
                    the OSR as a function of band.
                mags (bool): True if catalogue contents are in mags; False if in fluxes
                abzp (float): AB magnitude zero-point of catalogue fluxes for conversion to
                    AB magnitudes.
                mag_min (float): Bright limit for OSR parametrisation
                mag_max (float): Faint limit for OSR parametrisation
                mag_step (float): Magnitude step-size for OSR parametrisation

                SNR (bool or float): If float, the SNR criteria to apply to all sources in catalog
                banderr (str): Suffix for 'band' to find error column in photometry catalog
        
        Returns:
            Public attributes created and appropriately formatted for later 
                use and access
        
        """
        # Cube containing the P(z) for each galaxy
        redshift_cube = np.array(redshift_cube)
        # Cube containing the M*(z) for each galaxy
        mass_cube = np.array(mass_cube)
        # Check for shape mismatch
        if redshift_cube.shape != mass_cube.shape:
            print("Redshift and Mass data-cube shapes do not match - please check")
        else:
            self._pz = redshift_cube
            self._mz = mass_cube

        # Set the redshift range array
        self.zr = z
        self.maskpath = maskpath

        # Get the peak of P(z) for every galaxy
        self.peakz_arg = np.argmax(self._pz, axis=1)
        self.peakz = self.zr[self.peakz_arg]

        if z_best:
            self._z_best = z_best
        else:
            self._z_best = self.peakz

        # Class photometry path
        self.photometry_path = photometry

        # Attempt to read the catalogue
        try:
            self.phot_catalog = Table.read( self.photometry_path )
        except:
            print("Cannot read photometry catalogue - please check")


        # Calculate Odds properties and make relevant masks
        print("Calculating Odds properties")
        self.calcOdds(band, K=K, dz=dz, OSRlim=OSRlim, mags=mags, abzp=abzp, 
                      mag_min = mag_min, mag_max = mag_max, mag_step = mag_step)
        self.oddsCut = np.array((self.odds > OSRlim))

        if SNR: # Apply additional SNR cut to full sample
            if mags: 
                # SNR Cut in magnitudes
                snr_mag =  2.5*np.log10(np.e)*(1/SNR)
                SNRcut = (np.abs(self.phot_catalog[band+banderr]) < snr_mag)
            else:
                # SNR cut in fluxes
                ferr = self.phot_catalog[band+banderr]
                fmin = (self.phot_catalog[band] > 0.)
                SNRcut = fmin*((self.phot_catalog[band]/self.phot_catalog[band+banderr]) > SNR)
            self.oddsCut = np.array(self.oddsCut * SNRcut)

        self.pz = self._pz[self.oddsCut,:]
        self.mz = self._mz[self.oddsCut,:]

        self.z_best = self._z_best[self.oddsCut]

        # Enforce P(z) normalisation
        factors = simps( self.pz, self.zr, axis=1)
        for gal in range(len(self.pz)):
            self.pz[gal,:] /= factors[gal]

        self.odds = self.odds[self.oddsCut]
        self.OSRmags = self.OSRmags[self.oddsCut]

        # Class ID, position and co-ordinate arrays
        self.IDs = np.array(self.phot_catalog[idcol],dtype=int)[self.oddsCut]
        self.RA = self.phot_catalog[racol][self.oddsCut]
        self.Dec = self.phot_catalog[deccol][self.oddsCut]
        
        # Ensure RA and Dec are in degrees
        try:
            if self.RA.unit.physical_type == 'angle':
                self.RA = self.RA.to(u.deg)
                self.Dec = self.Dec.to(u.deg)
            else:
                self.RA = self.RA.data * u.deg
                self.Dec = self.Dec.data * u.deg

        except AttributeError:
            # Need a fallback
            self.RA = self.RA.data * u.deg
            self.Dec = self.Dec.data * u.deg

        self.coords = SkyCoord(self.RA,self.Dec,frame='icrs')       
         
        # Set up cosmology
        if not cosmology:
            self.cosmo = FlatLambdaCDM( H0=70, Om0=0.3 )
        else:
            self.cosmo = cosmology

    # DEFINITION FUNCTIONS

    def setSeparation(self, r_min, r_max):
        """ Define the physical radius conditions of the class. Designed this way so 
                can change this on the fly if need be.

        Args:
            r_min (astropy.unit float): Minimum physical radius for close pairs.
            r_max (astropy.unit float): Maximum physical radius for close pairs.

        """
        # Set the class properties
        self.r_min = r_min.to(u.kpc)
        self.r_max = r_max.to(u.kpc)

    def initialSample(self,initial):
        """ Define and set up initial sample of galaxies in which to find pairs
        
        Initial sample definition is done OUTSIDE of class and then loaded to allow
        for different sample definition criteria for respective papers. Alleviate need
        for multiple different functions.
        
        Args:
            initial (1d array): Indexes of galaxies which satisfy the initial sample
                criteria.
        
        """
        self.initial = np.array(initial)

    # CALCULATION FUNCTIONS

    def findInitialPairs(self, z_max = 4.0, z_min = 0.3, tol = 0.02, exclusions = False):
        """ Find an initial list of potential close pair companions based on the already
            defined separations.

            Args:
                z_min (float): Minimum redshift being considered by the work. For calc-
                    ulation of maximum separation.
                z_max (float): Minimum redshift being considered by the work. For calc-
                    ulation of maximum separation.
                tol (float): Fractional tolerance in separation criteria for generating
                    initial list of potential pairs. Account for discrete nature of max
                    and min separation calculations.

        """

        # Convert Sky Coordinates to cartesian xyz for correct 3d distances
        cartxyz = self.coords.cartesian.xyz
        flatxyz = cartxyz.reshape((3, np.prod(cartxyz.shape) // 3))
        
        sample_tree = cKDTree(flatxyz.value.T[self.initial])

        # Calculate min and maximum angular diameter distance in redshift range
        # in case it spans angular diameter distance turnover.
        dAdist = self.cosmo.angular_diameter_distance(np.linspace(z_min, z_max, 1000))
        dAmin = dAdist.min()*(1-tol)
        dAmax = dAdist.max()*(1+tol)

        # Calculate separations
        maxsep = (self.r_max / dAmin.to(u.kpc))*u.rad
        minsep = (self.r_min / dAmax.to(u.kpc))*u.rad

        # Convert on-sky angular separation to matching cartesian 3d distance
        # (See astropy.coordinate documentation for seach_around_sky)
        # If changing function input to distance separation and redshift, MUST convert
        # to an angular separation before here.

        r_maxsep = (2 * np.sin(Angle(maxsep) / 2.0)).value
        r_minsep = (2 * np.sin(Angle(minsep) / 2.0)).value
        
        
        # Computed trees might be worth keeping, maybe not
        self.full_tree = cKDTree(flatxyz.value.T) 
        self.initial_pairs = sample_tree.query_ball_tree(self.full_tree, 
                                                         r_maxsep)
        self.initial_tooclose = sample_tree.query_ball_tree(self.full_tree, 
                                                            r_minsep)

        # Remove both self-matches and matches below min separation
        for i, primary in enumerate(self.initial):
            # Sort so it matches brute force output
            self.initial_pairs[i] = np.sort(self.initial_pairs[i])
            self.initial_tooclose[i] = np.sort(self.initial_tooclose[i])

            # Delete self-matches and matches within minsep
            self.initial_pairs[i] = np.delete(self.initial_pairs[i],
                                              np.searchsorted(self.initial_pairs[i],
                                                              self.initial_tooclose[i]))

        # Trim pairs
        self.trimmed_pairs = np.copy(self.initial_pairs)
        Nduplicates = 0

        for i, primary in enumerate(self.initial):
            for j, secondary in enumerate(self.initial_pairs[i]):
                if secondary in self.initial:
                    
                    Nduplicates += 1 # Counter to check if number seems sensible

                    primary_mass = self.mz[primary, self.peakz_arg[primary]]
                    secondary_mass = self.mz[secondary, self.peakz_arg[secondary]]
                    
                    if secondary_mass > primary_mass:
                        self.trimmed_pairs[i] = np.delete(self.initial_pairs[i], j)
                    else:
                        k = np.where(self.initial == secondary)[0][0]
                        index = np.where(self.initial_pairs[k] == primary)[0][0]
                        self.trimmed_pairs[k] = np.delete(self.initial_pairs[k],
                                                          index)

        Ntotal = np.sum([len(self.trimmed_pairs[gal]) for gal in range(len(self.initial))])
            
        print ('{0} duplicates out of {1} total pairs trimmed'.format(Nduplicates , Ntotal))

        self._max_angsep = maxsep
        self._min_angsep = minsep

    def makeMasks(self, min_mass, max_mass=13, mass_ratio = 4.):
        """ Make the various masks to enforce angular separation, stellar mass ratio and
            selection conditions. Also produce the Z(z)-function.

            Args:
                mass_cut (float or float-array): Defines the stellar mass cut to be included
                    in the primary sample. Units of log10(stellar mass).
                max_mass (float): Maximum stellar mass of a galaxy
                mass_ratio (float): Ratio of stellar masses to be considered a pair.

        """

        dA_z = self.cosmo.angular_diameter_distance(self.zr).to(u.kpc)
        z_msks, sep_msks, sel_msks, pri_msks, Nzpair = [], [], [], [], []
        pri_weights = []
        sec_weights = []
        
        for i, primary in enumerate( self.initial ):
            # Calculate Primary and Secondary galaxy weights
            pri = self.OSRweights(self.OSRmags[primary])
            sec = self.OSRweights(self.OSRmags[self.trimmed_pairs[i]])
            # if np.isnan(pri):
            #     pri = 1
            # sec[np.isnan(sec)] = 1
            # if np.sum(np.isnan(sec)) > 0:
            #     print 'ISNAN weight'+str(primary)+' '+str(secondary)
            pri_weights.append(pri)
            sec_weights.append(sec)

            primary_pz = self.pz[primary, :]
            primary_mz = self.mz[primary, :]
            Zz_arrays, Zpair_fracs = [], []
            sep_arrays, sel_arrays = [], []

            primary_pz /= simps(primary_pz, self.zr)

            # Get angular distances of all companions
            d2d = self.coords[primary].separation(self.coords[self.trimmed_pairs[i]]).to(u.rad)

            # Min/max angular separation as a function of redshift
            theta_min = ((self.r_min / dA_z)*u.rad)
            theta_max = ((self.r_max / dA_z)*u.rad)

            # Make a selection function mask
            pri_msks.append( np.logical_and(np.log10(self.mz[primary]) >= min_mass,
                                            np.log10(self.mz[primary]) < max_mass))

            for j, secondary in enumerate(self.trimmed_pairs[i]):
                
                # Redshift probability
                # -----------------------------------------
                secondary_pz = self.pz[ secondary, :]
                secondary_pz /= simps(secondary_pz, self.zr)
                
                Nz = (primary_pz + secondary_pz) * 0.5
                Zz = np.nan_to_num((primary_pz * secondary_pz) / Nz)
                Zz_arrays.append(Zz)
                Zpair_fracs.append(simps(Zz,self.zr))

                # Separation masks
                # -----------------------------------------
                # Sepration (in degrees) between primary and secondary
                # d2d = self.coords[primary].separation(self.coords[secondary]).to(u.deg)

                # Create boolean array
                sep_msk = np.logical_and(d2d[j] >= theta_min , d2d[j] <= theta_max)
                sep_arrays.append( sep_msk )

                # Selection masks
                # ----------------------------------------- 
                secondary_mz = self.mz[ secondary, :]
                # Create the boolean array enforcing conditions
                sel_msk = np.array((primary_mz/secondary_mz) <= mass_ratio, dtype=bool)
                sel_arrays.append( sel_msk )
            
            sel_msks.append( sel_arrays )
            z_msks.append( Zz_arrays )
            Nzpair.append( Zpair_fracs )
            sep_msks.append( sep_arrays )

        # Set class variables
        self.redshiftProbs = np.array( z_msks )
        self.separationMasks = np.array( sep_msks )
        self.pairMasks = np.array( sel_msks )
        self.selectionMasks = np.array( pri_msks )
        self.Nzpair = np.array( Nzpair )
        self.OSRweights_primary = np.array(pri_weights)
        self.OSRweights_secondary = np.array(sec_weights)
        self._massRatio = mass_ratio
        # Calc the PPF
        self.calcPPF()

    def calcPPF(self):
        """ Function to calculate the unweighted PPF

        """

        PPF_total = []
        PPF_pairs = []

        for i, primary in enumerate(self.initial):
            PPF_temp = []
            PPF_tot_temp = []
            for j, secondary in enumerate( self.trimmed_pairs[i] ):
                ppf_z = (self.redshiftProbs[i][j] * self.pairMasks[i][j] * 
                         self.selectionMasks[i] * self.separationMasks[i][j])
                PPF_temp.append( ppf_z )
                PPF_tot_temp.append( simps(ppf_z, self.zr) )

            PPF_pairs.append(PPF_temp)
            PPF_total.append(np.sum(PPF_tot_temp))

        self.PPF_pairs = np.array( PPF_pairs )
        self._PPF_total = np.array( PPF_total )

    def mergerFraction(self, zmin, zmax):
        """ Calculate the merger fraction as in Eq. 22 (Lopez-Sanjuan et al. 2014)
            
            Args:
                zmin (float):   minimum redshift to calculate f_m
                zmax (float):   maximum redshift to calculate f_m

        """
        fm = self.mergerIntegrator(zmin, zmax, self.initial, self.trimmed_pairs,
                                   self.selectionMasks, self.PPF_pairs, 
                                   self.OSRweights_primary,
                                   self.OSRweights_secondary) # Add pair weights when done
        self.fm = fm
        self._zrange = [zmin, zmax]
        return self.fm

    def mergerIntegrator(self, zmin, zmax, initial, trimmed_pairs,
                         selectionMasks, PPF_pairs, OSR_primary,
                         OSR_secondary):
        """ Function to integrate the total merger fraction.
        
            Generalised to receive any set of inputs rather than stored
            values so that both true value and error estimates can be 
            calculated through resampling methods, e.g. bootstrap
            (implemented below) or jackknife (yet to implement)

            Args:
                zmin (float):   minimum redshift to calculate f_m
                zmax (float):   maximum redshift to calculate f_m

                initial
                trimmed_pairs
                selectionMasks
                PPF_pairs
                #Pair weights (when calculated)

        """

        # Redshift mask we want to examine
        zmask = np.logical_and(self.zr >= zmin, self.zr <= zmax)

        # Integrate over pairs
        k_sum = 0.
        i_sum = 0.
        k_int = PPF_pairs # * self.pairWeights
        for i, primary in enumerate(initial):
            if PPF_pairs[i]: # Some are empty
                for j, secondary in enumerate(trimmed_pairs[i]):
                    k_sum += (simps(k_int[i][j][zmask], self.zr[zmask]) * 
                              OSR_secondary[i][j])

            # Integrate over the primary galaxies
            # Re-enforce Pz normalisation
            i_pz = self.pz[primary] / simps(self.pz[primary], self.zr)
            i_int = i_pz * selectionMasks[i] * OSR_primary[i]
            i_sum += simps( i_int[zmask], self.zr[zmask])

        # Set the merger fraction
        fm = k_sum / i_sum
        self._k_sum = k_sum
        self._i_sum = i_sum
        return fm
        
    def calcOdds(self, band, K=0.1, dz=0.01, OSRlim=0.3, mags=True, abzp=False,
                 mag_min = 15, mag_max = 30.5, mag_step = 0.25):
        """ Calculate the Odds parameter for each galaxy and the odds sampling rate (OSR)
            for the field.

            Args:
                K (float): parameter detailing limits to integrate around, such that integral
                    performed over z_best +/- K(1+z).
                dz (float): redshift range step to interpolate to.
                OSRlim (float): selected limit to calculate the odds sampling rate (OSR).
                band (str): column name in photometry catalogue one wishes to parametrise
                    the OSR as a function of band.
                mags (bool): True if catalogue contents are in mags; False if in fluxes
                abzp (float): AB magnitude zero-point of catalogue fluxes for conversion to
                    AB magnitudes.
                mag_min (float): Bright limit for OSR parametrisation
                mag_max (float): Faint limit for OSR parametrisation
                mag_step (float): Magnitude step-size for OSR parametrisation
        """

        if self._pz.shape[0] != self._z_best.shape[0]:
            print "Redshift array and P(z) cube not the same length - please check"

        odds = []
        zr_i = np.arange( self.zr.min(), self.zr.max()+dz, dz)

        for gal in range(len(self._pz)):

            gal_dz = K*(1.+self._z_best[gal])
            #gal_intmsk = np.logical_and(self.zr >= (self.z_best[gal]-gal_dz), 
            #                                    self.zr <= (self.z_best[gal]+gal_dz))

            # Interpolate only the section of self.zr,self.pz that we want to integrate
            gal_intmsk_i = np.logical_and(zr_i >= (self._z_best[gal]-gal_dz),
                                                zr_i <= (self._z_best[gal]+gal_dz))
            gal_pz = np.interp(zr_i, self.zr, self._pz[gal,:])
            gal_int_tot = simps(gal_pz, zr_i)
            gal_int_i = np.clip(simps(gal_pz[gal_intmsk_i], zr_i[gal_intmsk_i])/ gal_int_tot, 0., 1.)
            
            odds.append(gal_int_i)

        self.odds = np.array(odds)
        self._oddsK = K

        # Parameterise it as a function of detection magnitude
        if not mags:
            if not abzp:
                print 'No AB zero-point for flux conversion - please check'
            self.OSRmags = -2.5*np.log10( self.phot_catalog[band] ) + abzp
        else:
            self.OSRmags = self.phot_catalog[band]

        # Clip mag range in case bins extend past observed photometry
        if mag_min < np.nanmin(self.OSRmags):
            mag_min = np.around(np.nanmin(self.OSRmags),1)-mag_step
        if mag_max >= np.nanmax(self.OSRmags):
            mag_max = np.around(np.nanmax(self.OSRmags),1)+mag_step
        magbins = np.arange(mag_min, mag_max, mag_step)

        OSRmag = []

        for b in range(len(magbins)-1):
            umag, lmag = magbins[b], magbins[b+1]
            binmsk = np.logical_and( self.OSRmags >= umag, self.OSRmags <= lmag)
            binodds = self.odds[binmsk]

            # Calculating odds ratio in 'whole' galaxies rather than
            # integrated P(z)'s negates any normalisation issues.
            goododds = (self.odds >= OSRlim)
            goodsum = np.sum(binmsk*goododds, dtype=float )
            allsum = np.sum(binmsk, dtype=float )

            if allsum == 0.:
                OSRmag.append(0.)
            else:
                OSRmag.append(goodsum/allsum)

        self.OSR = np.nan_to_num(OSRmag)
        self.OSRmagbins = magbins[:-1] + 0.5*np.diff(magbins)

    def OSRweights(self, mag_input):
        """ OSR Weight function
        """
        weights = np.divide(1.0, griddata(self.OSRmagbins, self.OSR, mag_input,
                                          fill_value=1.))
        return weights

    def bootstrapMergers(self, zmin, zmax, nsamples  = 10):
        """ Estimate error on fm through bootstrap resampling of initial sample
            
            Efron (1979) - estimate of Bootstrap Standard Error
            
            Args:
                nsamples (int): Number of bootstrap resampling iterations.
                    Default = 10 (will make larger for final estimates, e.g. 50-100)
        
        """
        
        try:
            float(self.fm) # Double check merger fraction already calculated
        except AttributeError:
            # Calculate if not yet created
            self.mergerFraction(zmin, zmax)
        
        fm_array = []
        
        for iteration in range(nsamples):
            # Sample with replacement the observed sample
            newsample = np.random.randint(len(self.initial), size = len(self.initial))
            
            # Make relevant lists for new bootstrap sample
            initial = np.array([self.initial[gal] for gal in newsample])
            trimmed_pairs = np.array([self.trimmed_pairs[gal] for gal in newsample])
            selectionMasks = np.array([self.selectionMasks[gal] for gal in newsample])
            PPF_pairs = np.array([self.PPF_pairs[gal] for gal in newsample])
            OSR_primary = np.array([self.OSRweights_primary[gal] for gal in newsample])
            OSR_secondary = np.array([self.OSRweights_secondary[gal] for gal in newsample])
            #Add pair_weights when done
            
            # Calculate fm for sample
            fm_newsample = self.mergerIntegrator(zmin, zmax, initial, trimmed_pairs, 
                                                 selectionMasks,
                                                 PPF_pairs, OSR_primary, OSR_secondary)
            fm_array.append(fm_newsample)
            
        fm_array = np.array(fm_array)
        self.fm_mean = fm_array.sum() / nsamples
        
        # Estimate StDev for resampled values
        fm_ste = np.sqrt( np.sum((fm_array - self.fm_mean)**2) / (nsamples - 1) )
        self.fm_ste = fm_ste
        return self.fm_mean, self.fm_ste

    def _mergerFraction(self, zmin, zmax):
        """ Calculate the merger fraction as in Eq. 22 (Lopez-Sanjuan et al. 2014)
    
            Args:
                zmin (float):   minimum redshift to calculate f_m
                zmax (float):   maximum redshift to calculate f_m
    
        """
    
        # Redshift mask we want to examine
        zmask = np.logical_and( self.zr >= zmin, self.zr <= zmax )
    
        # Integrate over pairs
        k_sum = 0.
        k_int = self.PPF_pairs # * self.pairWeights
        for i, primary in enumerate(self.initial):
            if self.PPF_pairs[i]: # Some are empty
                for j, secondary in enumerate(self.trimmed_pairs[i]):
                    k_sum += np.sum( simps( k_int[i][j][zmask], self.zr[zmask], ) )
    
        # Integrate over the primary galaxies
        i_int = self.pz[ self.initial ] * self.selectionMasks # * galaxyweights
        i_sum = np.sum( simps( i_int[:,zmask], self.zr[zmask], axis = 1) )
    
        # Set the merger fraction
        self.fm = k_sum / i_sum
        self._zrange = [zmin, zmax]
        return self.fm

    def _calcArea(self, primary_index, maskimage_path=False, rmin=0*u.arcsec,
                    rmax=15*u.arcsec, ps=0.2684*u.arcsec/u.pixel, xy=False, maskdata=False):

        """ Calculate the area of each search annuli masked/unobserved to weight
            pairs.

            Args:

                primary_index (int or int array): indices of galaxies one wishes to calculate an
                    aperture for
                maskimage_path (str): path to mask image. If already opened, pass hdu via maskdata
                rmin (astropy unit float): inner angle search radius
                rmax (astropy unit float): outer angle search radius
                ps (astropy unit float): mask image pixel-scale in appropriate units
                xy (bool): if true, supplying (x,y) coords for apertures, not an astropy.coordinate
                    set. NOT IMPLEMENTED.
                maskdata (fits hdu): supply a astropy.io.fits hdu object so that we do not have to 
                    keep opening the large file.
                
        """

        # Load in FITS image
        if maskdata:
            image = maskdata
        else:
            image = fits.open(maskimage_path)[0]
        # Create apertures
        if xy:
            # NOT IMPLEMENTED YET
            apertures = CircularAnnulus( self.coords[primary_index], r_in=rmin, r_out=rmax,)
        else:
            apertures = SkyCircularAnnulus( self.coords[primary_index], r_in=rmin.to(u.arcsec),
                r_out=rmax.to(u.arcsec))
        # Perform sum
        photo_table = aperture_photometry(image, apertures, method='center')
        photo_table = photo_table['aperture_sum']
        # Calculate the fraction of area covered
        r_min_pix = (rmin / ps)
        r_max_pix = (rmax / ps)
        f_area = photo_table / (np.pi*(r_max_pix.value - r_min_pix.value)**2.)
        # f_area = np.clip(photo_table / (np.pi*(r_max_pix - r_min_pix)**2.), 0., 1.)

        if np.isinf(f_area).any():
            print 'Some f_area are inf - please check'

        return np.nan_to_num(np.divide(1.,f_area))

    def _areaWeights(self):
        """ Calculate the area weights of the galaxies in the initial sample.

        Args:
            None

        Requirements:
            photutils
        """

        if not self.maskpath:
            self.areaWeights = np.ones( (len(self.initial),len(self.zr)) )
            print 'no mask image provided. cannot calculate area weights. assumed unity.'
        else:
            maskdata = fits.open(self.maskpath)[0]
            weights = []
            zr = np.arange(0.01, self.zr.max()+0.2, 0.2)

            for zz in zr:
                theta_z_min = (self.r_min.to(u.kpc) / self.cosmo.angular_diameter_distance(zz).to(u.kpc))*u.rad
                theta_z_max = (self.r_max.to(u.kpc) / self.cosmo.angular_diameter_distance(zz).to(u.kpc))*u.rad
                area = self._calcArea(self.initial,rmin=theta_z_min.to(u.arcsec),rmax=theta_z_max.to(u.arcsec),
                            maskdata=maskdata, ps=0.2684*u.arcsec/u.pixel)
                weights.append(area)

            weights = np.array(weights)
            self.areaWeights = griddata( zr, weights, self.zr,).T

    def _calcMassCompWeights(self, mf_z, mf_params, mf_fn):
        """ Calculate the weights needed to ensure that any searches for companions
            that fall below the mass completeness are weighted appropriately. Following
            the work of Patton et al. (2000)

            Args:
                mf_z (float array): mass function redshift bins that correspond to the
                    parameters given in mf_params
                mf_params (float array): mass function parameters to pass to mf_fn in order
                    to generate the mass function  within each redshift bin
                mf_fn (function name): function to pass *mf_params[i,:] for redshift bin i
        """

        bin_indices = np.ones_like(self.zr)*-99.
        # What mass function bin does each z in self.zr correspond to?
        for bi in range(len(mf_z)-1):
            binl, binh = mf_z[bi], mf_z[bi+1]
            mask = np.logical_and( self.zr >= binl, self.zr < binh)
            bin_indices[mask] = bi

        # If it equals -99, assign a weight of 1 later on.
        if (bin_indices == -99.).any():
            print 'WARNING - mass function redshift bins do not match self.zr. Please check.'

        # log stellar mass array to act as integration x-axis
        mass_x = np.arange(7.,14.,0.05) # need to 10*x this later

        # test brute force way
        primaryFluxInts = []
        for i, primary in enumerate(self.initial):
            # Get the limiting stellar mass at every z
            m_lim = np.maximum( [self.mz[primary,:]/self._massRatio, self._massCompleteness(self.zr)] )

            primary_z = []
            for zi, z in enumerate(self.zr):
                # For each redshift, perform the integral
                if self.mz[primary,zi]/self._massRatio >= self._massCompleteness(self.zr)[zi]:
                    primary_z.append(1.)
                else:
                    mask = np.logical_and(mass_x >= m_lim[zi], mass_z <= self.mz[primary,zi])
                    mf_y = mf_fn(mass_z[mask], *mf_params[bin_indices[zi]])
                    top = simps(mf_y, mass_x[mask])

                    mask = np.logical_and(mass_x >= self.mz[primary,zi]/self._massRatio, 
                                mass_x < self.mz[primary,zi])
                    mf_y = mf_fn(mass_z[mask], *mf_params[bin_indices[zi]])
                    bottom = simps(mf_y, mass_x[mask])

                    primary_z.append( top/bottom )

            primaryFluxInts.append(primary_z)


        self.fluxWeights = np.array(primaryFluxInts)
        
        def setMassCompleteness(self, redshift, magnitude, magLim):
            """ Set up mass completeness function parameters
            
            Reads in magnitude for 1Msol normalised magnitude for
            desired M/L ratio for completeness calculations and sets up the
            appropriate variables.
            
            Args:
                redshift (array): Redshift array for which magnitudes have been
                    calculated
                magnitude (array): Corresponding magnitudes for the chosen M/L
                magLim (float): Magnitude completeness limit for the field
            
            """
            self._comp_zr = np.array(redshift)
            self._comp_magnitude = np.array(magnitude)
            self.magLim = float(magLim)

        def _massCompleteness(self, redshifts):
            """ Calculate Mass completeness limits 
            
                Args:
                    redshifts (array): Redshift array to calculate mass
                        completeness for.
                        
                Returns:
                    mass_limit (array): Mass completeness limit for the set magnitude
                        limit and M/L curve (set with setMassCompleteness).
                        
                        Manually changing self.magLim will change the calculated
                        mass completeness correspondingly, without the need to 
                        redefine the completeness curve seperately.
            
            """
            mass_limit = griddata(self._comp_zr, 
                                  0.4*(self._comp_magnitude - self.magLim),
                                  redshifts)

            return 10**mass_limit
            
        # primaryWeights = []
        # for i, primary in enumerate(self.initial):

        #     pweight = []
        #     for zi, z in enumerate(self.zr):

        #         if bin_indices[zi] < 0.:
        #             pweight.append(1.)
        #         else:
        #             zpmass = self.mz[primary]
        #             zpmasslim = zpmass / self._massRatio
        #             survey_masslim = massLimFn(z)

        #             if survey_masslim < zpmasslim:
        #                 # No need to integrate
        #                 pweight.append(1.)
        #             else:
        #                 # Perform integral from Patton et al. (2000) - integrate GSMF from the mass 
        #                 # ...of the primary galaxy to the survey mass limit and the
        #                 top, _e, _info = quad(mf_fn, survey_masslim, zpmass, args=mf_params[bin_indices[zi]])
        #                 bottom, _e, _info = quad(mf_fn, zpmasslim, zpmass, args=mf_params[bin_indices[zi]])
        #                 # Total weighting of companion is 1/ (top/bot)
        #                 pweight.append( np.divide(1., top/bottom ) )

        #     primaryWeights.append(pweight)
        # self.massCompWeights = np.array(primaryWeights)

    # SEPARATE MASKING FUNCTIONS

    def selectionMask(self, mass_cut):
        """ Create selection function masks for each of the primary galaxies

        Args:  
            mass_cut (float or float array): mass cut to use over self.zr

        """

        pri_msks = []

        for i, primary in enumerate( self.initial ):
            pri_msks.append( np.array( np.log10(self.mz[primary]) >= mass_cut, dtype=bool ) )

        return np.array( pri_msks )

    def separationMask(self):

        sep_msks = []

        for i, primary in enumerate( self.initial ):
            # Get angular distances of all companions
            d2d = self.coords[primary].separation(self.coords[self.trimmed_pairs[i]]).to(u.rad)
            sep_arrays = []

            for j, secondary in enumerate( self.trimmed_pairs[i] ):
                # Create boolean array
                sep_msk = np.logical_and( d2d[j] >= theta_min , d2d[j] <= theta_max )
                sep_arrays.append( sep_msk )

            sep_msks.append( sep_arrays )

        return np.array( sep_msks )

    def pairMask(self, mass_ratio = 4.):

        pair_msks = []

        for i, primary in enumerate(self.initial):
            pair_arrays = []
            primary_mz = self.mz[primary, :]

            for j, secondary in enumerate(self.trimmed_pairs[i]):
                secondary_mz = self.mz[secondary, :]
                # Create the boolean array enforcing conditions
                p_msk = np.array((primary_mz/secondary_mz) <= mass_ratio, dtype=bool)
                # sel_msk = np.logical_and(primary_mz >= 10.**mass_cut, (primary_mz/secondary_mz) <= mass_ratio)
                pair_arrays.append( p_msk )

            pair_msks.append( sel_arrays )

        return np.array( sel_msks )

    def redshiftProb(self):

        Zz_msks = []

        for i, primary in enumerate(self.initial):
            Zz_arrays = []
            primary_pz = self.pz[primary]
            # Re-inforce P(z) normalisation
            primary_pz /= simps(primary_pz, self.zr)

            for j, secondary in enumerate(self.trimmed_pairs[i]):
                secondary_pz /= simps(secondary_pz, self.zr)
                secondary_pz = self.pz[ secondary]
                Nz = (primary_pz + secondary_pz) * 0.5
                Zz = np.nan_to_num( (primary_pz * secondary_pz) / Nz )
                Zz_arrays.append( Zz )

            Zz_msks.append( Zz_arrays )

        return np.array( Zz_msks )

    # VISUALISATION FUNCTIONS

    def plotOSR(self, legend=True, draw_frame=False):
        Fig = plt.figure(figsize=(6,3.5))
        Ax = Fig.add_subplot(111)


        Ax.plot( self.OSRmagbins, self.OSR, 'o', color='w', mew=2, mec='dodgerblue')

        Ax.set_ylim(-0.1,1.1)
        Ax.set_xlabel('AB magnitude')  
        Ax.set_ylabel('OSR')

        plt.tight_layout()
        plt.show()

    def plotOdds(self, legend=True, draw_frame=False):
        Fig = plt.figure(figsize=(6,3.5))
        Ax = Fig.add_subplot(111)

        Ax.hist(self.odds, np.arange(0.,1.025,0.025), normed=1)

        Ax.set_xlabel('Odds')
        Ax.set_ylabel('counts')

        Ax.text(0.05,0.9, '{0} = {1:.3f} {2}'.format(r'$\Delta z$', self._oddsK, r'$\times (1+z)$'), 
                                transform=Ax.transAxes)

        plt.tight_layout()
        plt.show()

    def plotSample(self,galaxy_indices,legend=True,draw_frame=False):

        Fig = plt.figure(figsize=(6,6.5))
        Ax = Fig.add_subplot(111)

        for gal in np.array(galaxy_indices,ndmin=1):
            Ax.plot( self.RA[gal], self.Dec[gal], 'ow', mec='r', mew=2, ms=10)

        Ax.plot( self.RA, self.Dec, 'ok', ms=5)

        Ax.set_ylabel('Dec')
        Ax.set_xlabel('RA')
        Fig.subplots_adjust(right=0.95,top=0.95,bottom=0.14)
        plt.show()

    def plotPz(self,galaxy_indices,legend=True,draw_frame=False):
        """ Plot the redshift likelihood distribution for a set of galaxies in sample.
        
        Args:
            galaxy_indices (int or 1-d array): galaxy indices to plot
            legend (bool): Include figure legend
            draw_frame (bool): Draw legend frame
            
        Returns:
            Matplotlib Figure 
            
        """
        
        Fig = plt.figure(figsize=(6,3.5))
        Ax = Fig.add_subplot(111)

        for gal in np.array(galaxy_indices,ndmin=1):
            Ax.plot(self.zr,self.pz[gal,:],lw=2, 
                    label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[gal],'$z_{peak}$:',self.peakz[gal]))

        
        Ax.set_xlabel('Redshift, z')
        Ax.set_ylabel(r'$P(z)$')
        plt.tight_layout()
        plt.show()
    
    def plotMz(self,galaxy_indices, legend=True, draw_frame=False):
        """ Plot the stellar mass vs redshift for a set of galaxies in sample.
        
        Args:
            galaxy_indices (int or 1-d array): galaxy indices to plot
            legend (bool): Include figure legend
            draw_frame (bool): Draw legend frame
            
        Returns:
            Matplotlib Figure 
            
        """

        Fig = plt.figure(figsize=(6,3.5))
        Ax = Fig.add_subplot(111)
        
        for gal in np.array(galaxy_indices,ndmin=1):
            Ax.plot(self.zr,np.log10(self.mz[gal,:]),lw=2, 
                    label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[gal],'$z_{peak}$:',self.peakz[gal]))
        if legend:
            Leg = Ax.legend(loc='lower right', prop={'size':8})
            Leg.draw_frame(draw_frame)
        
        Ax.set_xlabel('Redshift, z')
        Ax.set_ylabel(r'$M_{*}(z)$')
        Ax.set_ylim(6.5,11.5)
        Fig.subplots_adjust(right=0.95,top=0.95,bottom=0.14)
        plt.show()
        
    def plotPairsMass(self,primary_index, legend=True, draw_frame=False):
        primary = self.initial[primary_index]
        secondaries = self.trimmed_pairs[primary_index]
        
        for j, secondary in enumerate(secondaries):
            Fig, Ax = plt.subplots(2,figsize=(4.5,6))
            Ax[0].plot(self.zr, self.pz[primary,:],'--',lw=2, color='dodgerblue',
                       label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[primary],
                                                                   '$z_{peak}$:',
                                                                   self.peakz[primary]))
                                                                   
            Ax[0].plot(self.zr, self.pz[secondary,:],':', lw=2, color='indianred',
                       label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[secondary],
                                                                   '$z_{peak}$:',
                                                                   self.peakz[secondary]))
            if legend:
                Leg1 = Ax[0].legend(loc='upper right', prop={'size':8})
                Leg1.draw_frame(draw_frame)
        
            Ax[0].set_xlabel('Redshift, z')
            Ax[0].set_ylabel(r'$P(z)$')
            Ax[0].text(0.9,0.25, r'{0:s} {1:.2f}'.format('$\mathcal{N}_{z} =$ ',self.Nzpair[primary_index][j]),
                       horizontalalignment='right',verticalalignment='center',
                       transform=Ax[0].transAxes)
            
            # Ax[1].plot(self.zr,np.log10(self.mz[primary,:]),'--',lw=2,color='dodgerblue',
            #            label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[primary],
            #                                                        '$z_{peak}$:',
            #                                                        self.peakz[primary]))
                                                                   
            # Ax[1].plot(self.zr,np.log10(self.mz[secondary,:]),':',lw=2,color='indianred',
            #            label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[secondary],
            #                                                        '$z_{peak}$:',
            #                                                        self.peakz[secondary]))
            
            
            print self.selectionMasks[primary_index][j]
            selmask = np.invert(self.selectionMasks[primary_index][j])
            print selmask
            zr_mask = np.ma.masked_where(selmask, self.zr)
            mz1 = np.ma.masked_where(selmask, np.log10(self.mz[primary,:]) )
            mz2 = np.ma.masked_where(selmask, np.log10(self.mz[secondary,:]) )

            print 'mz1', mz1.shape, 'mz2', mz2.shape, 'zr_mask', zr_mask.shape
            print zr_mask

            Ax[1].plot(zr_mask, mz1, '--', lw=5,color='dodgerblue',
                       label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[primary],
                                                                   '$z_{peak}$:',
                                                                   self.peakz[primary]))

            Ax[1].plot(zr_mask, mz2, ':', lw=5,color='indianred',
                       label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[secondary],
                                                                   '$z_{peak}$:',
                                                                   self.peakz[secondary]))


            Ax[1].set_xlabel('Redshift, z')
            Ax[1].set_ylabel(r'$\log_{10} \rm{M}_{\star}$')
            Ax[1].set_ylim(6.5,12.5)
            Fig.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.1,hspace=0.25)
            plt.show()

    def plotPairsPz(self,primary_index,legend=True,draw_frame=False):

        primary = self.initial[primary_index]
        secondaries = self.initial_pairs[primary_index]
        
        for j, secondary in enumerate(secondaries):
            Fig, Ax = plt.subplots(1,figsize=(4.*golden,4.))
            # Plot the primary P(z)
            # Ax.plot(self.zr,cumtrapz(self.pz[primary,:], self.zr, initial=0),'--',lw=2,color='b',
            #             label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[primary_index],
            #                                                        '$z_{peak}$:',
            #                                                        self.peakz[primary_index]))
            # # Plot the secondary P(z)
            # Ax.plot(self.zr,cumtrapz(self.pz[secondary,:], self.zr, initial=0),':',lw=2,color='r',
            #             label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[secondary],
            #                                                        '$z_{peak}$:',
            #                                                        self.peakz[secondary]))
            # # Plot the Z function
            # Ax.plot(self.zr,cumtrapz(self.redshiftProbs[primary_index][j], self.zr, initial=0), '--k', lw=2.5)

            Ax.plot(self.zr,self.pz[primary,:]/self.pz[primary,:].max(),'--',lw=2,color='b',
                        label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[primary_index],
                                                                   '$z_{peak}$:',
                                                                   self.peakz[primary_index]))
            # Plot the secondary P(z)
            Ax.plot(self.zr,self.pz[secondary,:]/self.pz[secondary,:].max(),':',lw=2,color='r',
                        label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[secondary],
                                                                   '$z_{peak}$:',
                                                                   self.peakz[secondary]))
            # Plot the Z function
            Ax.plot(self.zr,cumtrapz(self.redshiftProbs[primary_index][j], self.zr, initial=0), '-w', lw=3)
            Ax.plot(self.zr,cumtrapz(self.redshiftProbs[primary_index][j], self.zr, initial=0), '-k', lw=2)

            # Legend
            if legend:
                Leg1 = Ax.legend(loc='upper right', prop={'size':8,},)
                Leg1.draw_frame(draw_frame)
            # Labels, limits
            Ax.set_xlabel('Redshift, z')
            Ax.set_ylabel(r'$P(z)$')
            # Ax.set_ylabel(r'$\int P(z)$')
            Ax.set_xlim(0,3.5), Ax.set_ylim(0,1.1)
            # Plot the Number of pairs
            Ax.text(0.75,0.1, r'{0:s} {1:.3f}'.format('$\mathcal{N}_{z} =$ ', self.Nzpair[primary_index][j]), transform=Ax.transAxes)
            
            Fig.subplots_adjust(left=0.10,right=0.95,top=0.95,bottom=0.13)
        
        plt.show()

    # INFORMATION FUNCTIONS

    def printInfo(self):

        print
        print '#'*70
        print '\t Pair information:'
        print '\t \t - Separations: r_min = {0:1.1f}, r_max = {1:1.1f}'.format( self.r_min, self.r_max )
        print '\t \t - Maximum separation = {0:1.2f}'.format( self._max_angsep.to(u.arcsec) )
        print '\t \t - {0} galaxies in primary sample'.format( len( self.initial) )
        print '\t \t - {0} companion galaxies identified'.format( np.sum([ len(self.trimmed_pairs[i]) for i in range(len(self.initial)) ]) )
        print '\t \t - Sum over Nz = {0:.3f}'.format( np.sum(np.sum(self.Nzpair)) )
        print '\t \t - Unweighted sum over PPFs = {0:.3f}'.format( np.sum(self._PPF_total) )
        print '\t \t - Weighted f_m = {0:.2f}% over {1:.1f} < z < {2:.1f}'.format( self.fm*100., *self._zrange )
        print '#'*70
        print
        

class MockPairs(object):
    """ Pair-count analysis for SAM mock catalog data        
    """

    def __init__(self, catalog, field_width=False, field_height=False, catalog_format = 'fits',
                 redshift_col = 'z', mass_col = 'Mstar', 
                 racol = 'RA', deccol = 'DEC', cosmology = False):
        """ Load and format appropriately the necessary data for pair-count calculations
        
        Args:
            Catalog Arguments:
                photometry (str): Path to photometry catalog
                catalog_format (str): Catalog format, e.g. 'fits' or 'ascii'
                idcol (str): Column name for galaxy IDs
                racol (str): Column name for galaxy RAs
                deccol (str): Column name for galaxy DECs
          
            Cosmology Arguments:
                cosmology (astropy.cosmology): Astropy.cosmology object
                
                    if cosmology == None: FlatLambdaCDM with H0=70, Om0=0.3 
                        assumed.

        Returns:
            Public attributes created and appropriately formatted for later 
                use and access
        
        """

        self.catalog_path = catalog
        # Attempt to read the catalogue
        try:
            self.catalog = Table.read( self.catalog_path, format = catalog_format )
        except:
            print("Cannot read photometry catalogue - please check")


        self._RA = self.catalog[racol]
        self._Dec = self.catalog[deccol]
        self._z = self.catalog[redshift_col].data
        self._mass = self.catalog[mass_col].data
        
        # Ensure RA and Dec are in degrees
        if self._RA.unit.physical_type == 'angle':
            self._RA = self._RA.to(u.deg)
            self._Dec = self._Dec.to(u.deg)

        else:
            self._RA = self._RA.data * u.deg
            self._Dec = self._Dec.data * u.deg

        self._coords = SkyCoord(self._RA,self._Dec,frame='icrs')       
        
        
        # Set up initial sub-region
        
        self.setSubField(field_width, field_height, fixed = True)
        
        # Set up cosmology
        if not cosmology:
            self.cosmo = FlatLambdaCDM( H0=70, Om0=0.3 )
        else:
            self.cosmo = cosmology

    def setSubField(self, field_width=False, field_height=False, fixed = False):
        """ Create sub-region from mock.
        """
        RA_min, RA_max = self._RA.min(), self._RA.max()
        Dec_min, Dec_max = self._Dec.min(), self._Dec.max()
        dDec = Dec_max - Dec_min
        dRA = RA_max - RA_min
    
        if field_width:
            if fixed:
                self.subRA = RA_min + 0.5*dRA
                self.subDec = Dec_min + 0.5*dDec
            else:
                self.subRA = (RA_min + 0.5*field_width) + (np.random.rand() * (RA_max-RA_min-field_width))
                self.subDec = (Dec_min + 0.5*field_height) + (np.random.rand() * (Dec_max-Dec_min-field_height))

            self.AreaCut = np.logical_and(np.abs(self._RA - self.subRA) < 0.5*field_width,
                                     np.abs(self._Dec - self.subDec) < 0.5*field_height)
        
            self.RA = self._RA[self.AreaCut]
            self.Dec = self._Dec[self.AreaCut]
            self.coords = self._coords[self.AreaCut]
            self.z = self._z[self.AreaCut]
            self.mass = self._mass[self.AreaCut]
        else:
            RAcut = np.logical_and(self._RA < (RA_max- 0.05*dRA), self._RA > (RA_min + 0.05*dRA))
            Deccut = np.logical_and(self._Dec < (Dec_max - 0.05*dDec), self._Dec > (Dec_min + 0.05*dDec))
            
            self.AreaCut = np.logical_and(RAcut, Deccut)
            self.RA = self._RA[self.AreaCut]
            self.Dec = self._Dec[self.AreaCut]
            self.coords = self._coords[self.AreaCut]
            self.z = self._z[self.AreaCut]
            self.mass = self._mass[self.AreaCut]           

    def setSeparation(self, r_min, r_max):
        """ Define the physical radius conditions of the class. Designed this way so 
                can change this on the fly if need be.

        Args:
            r_min (astropy.unit float): Minimum physical radius for close pairs.
            r_max (astropy.unit float): Maximum physical radius for close pairs.

        """
        # Set the class properties
        self.r_min = r_min.to(u.kpc)
        self.r_max = r_max.to(u.kpc)

    def findPairs(self, z_min = 0.3, z_max = 4.0, ztol = 0.02, tol = 0.02,
                  min_mass = 9.5, max_mass = 13.,ratio = 4.):
        """ Find an initial list of potential close pair companions based on the already
            defined separations.

            Args:
                z_min (float): Minimum redshift being considered by the work. For calc-
                    ulation of maximum separation.
                z_max (float): Minimum redshift being considered by the work. For calc-
                    ulation of maximum separation.
                tol (float): Fractional tolerance in separation criteria for generating
                    initial list of potential pairs. Account for discrete nature of max
                    and min separation calculations.

        """

        # Convert Sky Coordinates to cartesian xyz for correct 3d distances

        self.sample_mr = np.array((self.mass >= (min_mass - np.log10(ratio))))
        self.sample_mr *= np.logical_and(z_min-ztol < self.z, self.z < z_max+ztol)
        
        sample_cut = np.logical_and(z_min < self.z[self.sample_mr], self.z[self.sample_mr] < z_max)
        sample_cut *= np.array(self.mass[self.sample_mr] >= min_mass)
        sample = np.where(sample_cut)[0]
        
        masses = self.mass[self.sample_mr]
        coords = self.coords[self.sample_mr]
        z_mr = self.z[self.sample_mr]

        cartxyz = coords.cartesian.xyz
        flatxyz = cartxyz.reshape((3, np.prod(cartxyz.shape) // 3))
                
        # Calculate min and maximum angular diameter distance in redshift range
        # in case it spans angular diameter distance turnover.
        dAdist = self.cosmo.angular_diameter_distance(np.linspace(z_min, z_max, 1000))
        dAmin = dAdist.min()*(1-tol)
        dAmax = dAdist.max()*(1+tol)

        # Calculate separations
        maxsep = (self.r_max / dAmin.to(u.kpc))*u.rad
        minsep = (self.r_min / dAmax.to(u.kpc))*u.rad

        # Convert on-sky angular separation to matching cartesian 3d distance
        # (See astropy.coordinate documentation for seach_around_sky)
        # If changing function input to distance separation and redshift, MUST convert
        # to an angular separation before here.

        r_maxsep = (2 * np.sin(Angle(maxsep) / 2.0)).value
        r_minsep = (2 * np.sin(Angle(minsep) / 2.0)).value
        
        dA_exact = self.cosmo.angular_diameter_distance(z_mr[sample]).to(u.kpc)
        max_exact = (self.r_max / dA_exact) * u.rad
        min_exact = (self.r_min / dA_exact) * u.rad
        
        # Computed trees might be worth keeping, maybe not
        sample_tree = cKDTree(flatxyz.value.T[sample])
        full_tree = cKDTree(flatxyz.value.T) 
        initial_pairs = sample_tree.query_ball_tree(full_tree, 
                                                    r_maxsep)
        self.trimmed = []

        self.exact_pairs = []
        self.exact_pairs_sum = []
        # Remove both self-matches and matches below min separation
        for i, primary in enumerate(sample):
            if i % 100 == 0:
                print i
                
            if initial_pairs[i]:
            # Sort so it matches brute force output
                initial_pairs[i] = np.sort(initial_pairs[i])

                z_sep = np.abs(z_mr[primary] - z_mr[initial_pairs[i]]) / (1 + z_mr[primary])
                r_sep = coords[primary].separation(coords[initial_pairs[i]]).to(u.rad).value
                pair = np.logical_and(r_sep >= min_exact[i].value,r_sep <= max_exact[i].value)
                pair *= np.logical_and(z_sep < 0.0017, 
                                       np.abs(masses[primary] - masses[initial_pairs[i]]) < np.log10(ratio))

                # if sum(pair) > 0:
                #     print i
                #     print r_sep, min_exact[i].value, max_exact[i].value
                #     print z_sep,
                #     print (len(initial_pairs[i][pair]))

                if sum(pair):
                    self.exact_pairs.append(initial_pairs[i][pair])
                    self.exact_pairs_sum.append(len(initial_pairs[i][pair]))
                    self.trimmed.append(primary)
            # Delete self-matches and matches within minsep
        # Trim pairs
        self.trimmed_pairs = np.copy(self.exact_pairs)
        
        Nduplicates = 0

        for i, primary in enumerate(self.trimmed):
            for j, secondary in enumerate(self.exact_pairs[i]):
                if secondary in self.trimmed:
                    
                    Nduplicates += 1 # Counter to check if number seems sensible

                    primary_mass = masses[primary]
                    secondary_mass = masses[secondary]
                    
                    if secondary_mass > primary_mass:
                        self.trimmed_pairs[i] = np.delete(self.exact_pairs[i], j)
                    else:
                        k = np.where(self.trimmed == secondary)[0][0]
                        index = np.where(self.exact_pairs[k] == primary)[0][0]
                        self.trimmed_pairs[k] = np.delete(self.exact_pairs[k], index)
                    
        self.Npairs = np.array([len(self.trimmed_pairs[gal]) for gal in range(len(self.trimmed))])
        self.Npairs_total = np.sum(self.Npairs)
        self.Ninitial = float(len(sample))
        
        self.fm = self.Npairs_total / self.Ninitial

        #Ntotal = np.sum([len(trimmed_pairs[gal]) for gal in range(len(sample))])
            
        #print ('{0} duplicates out of {1} total pairs trimmed'.format(Nduplicates , Ntotal))

        self._max_angsep = maxsep
        self._min_angsep = minsep
