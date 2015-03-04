import time
# Matplotlib, numpy
import matplotlib.pyplot as plt, numpy as np
# Astropy
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
# Scipy
from scipy.spatial import cKDTree
from scipy.integrate import simps, trapz, cumtrapz
from scipy.interpolate import interp1d

class Pairs(object):
    """ Main class for photometric pair-count analysis
    
    Implementation of the Lopez-Sanjuan et al. (2014) redshift PDF pair-count method
    for photometric redshifts from EAZY (or other) and stellar mass estimates using
    Duncan et al. (2014) outputs.
        
    """

    def __init__(self, z, redshift_cube, mass_cube, band, z_best=False, 
                 photometry=False, catalog_format = 'fits',
                 idcol = 'ID', racol = 'RA', deccol = 'DEC', cosmology = False,
                 K = 0.075, dz = 0.01, OSRlim = 0.3, mags=True, abzp = False,
                 mag_min = 17.5, mag_max = 30., mag_step = 0.5):
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

        self.peakz_arg = np.argmax(self._pz, axis=1)
        self.peakz = self.zr[self.peakz_arg]

        if z_best:
            self.z_best = z_best
        else:
            self.z_best = self.peakz

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
                      mag_min=mag_min, mag_max=mag_max, mag_step=mag_step)
                      
        self.oddsCut = np.array((self.odds > OSRlim))

        self.pz = self._pz[self.oddsCut,:]
        self.mz = self._mz[self.oddsCut,:]

        # Class ID, position and co-ordinate arrays
        self.IDs = np.array(self.phot_catalog[idcol],dtype=int)[self.oddsCut]
        self.RA = self.phot_catalog[racol][self.oddsCut]
        self.Dec = self.phot_catalog[deccol][self.oddsCut]
        
        # Ensure RA and Dec are in degrees
        if self.RA.unit.physical_type == 'angle':
            self.RA = self.RA.to(u.deg)
            self.Dec = self.Dec.to(u.deg)

        else:
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

    def findInitialPairs(self, z_max = 4.0, z_min = 0.3, tol = 0.02):
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
                mass_ratio (float): Ratio of stellar masses to be considered a pair.

        """

        dA_z = self.cosmo.angular_diameter_distance(self.zr).to(u.kpc)
        z_msks, sep_msks, sel_msks, pri_msks, Nzpair = [], [], [], [], []

        for i, primary in enumerate( self.initial ):

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
                                   self.redshiftProbs, self.pairMasks,
                                   self.separationMasks, self.selectionMasks,
                                   self.PPF_pairs) # Add pair weights when done
        self.fm = fm
        self._zrange = [zmin, zmax]
        return self.fm

    def mergerIntegrator(self, zmin, zmax, initial, trimmed_pairs,
                         selectionMasks, PPF_pairs):
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
                    k_sum += simps(k_int[i][j][zmask], self.zr[zmask])

            # Integrate over the primary galaxies
            # Re-enforce Pz normalisation
            i_pz = self.pz[primary] / simps(self.pz[primary], self.zr)
            i_int = i_pz * selectionMasks[i] # * galaxyweights
            i_sum += simps( i_int[zmask], self.zr[zmask])

        # Set the merger fraction
        fm = k_sum / i_sum
        self._k_sum = k_sum
        self._i_sum = i_sum
        return fm
        
    def calcOdds(self, band, K=0.1, dz=0.01, OSRlim=0.3, mags=True, abzp=False,
                 mag_min = 17.5, mag_max = 30., mag_step = 0.25):
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

        if self._pz.shape[0] != self.z_best.shape[0]:
            print "Redshift array and P(z) cube not the same length - please check"

        odds = []
        zr_i = np.arange( self.zr.min(), self.zr.max()+dz, dz)

        for gal in range(len(self._pz)):

            gal_dz = K*(1.+self.z_best[gal])
            #gal_intmsk = np.logical_and(self.zr >= (self.z_best[gal]-gal_dz), 
            #                                    self.zr <= (self.z_best[gal]+gal_dz))

            # Interpolate only the section of self.zr,self.pz that we want to integrate
            gal_intmsk_i = np.logical_and(zr_i >= (self.z_best[gal]-gal_dz),
                                                zr_i <= (self.z_best[gal]+gal_dz))
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
            mags = -2.5*np.log10( self.phot_catalog[band] ) + abzp
        else:
            mags = self.phot_catalog[band]

        magbins = np.arange(mag_min, mag_max, mag_step)
        OSRmag = []

        for b in range(len(magbins)-1):
            umag, lmag = magbins[b], magbins[b+1]
            binmsk = np.logical_and( mags >= umag, mags <= lmag)
            binodds = self.odds[binmsk]

            # Calculating odds ratio in 'whole' galaxies rather than
            # integrated P(z)'s negates any normalisation issues.
            goododds = (self.odds >= OSRlim)
            goodsum = np.sum(binmsk*goododds, dtype=float )
            allsum = np.sum(binmsk, dtype=float )

            OSRmag.append(goodsum/allsum)

        self.OSR = np.nan_to_num(OSRmag)
        self.OSRmags = np.array( [(magbins[i]+magbins[i+1])/2. for i in range(len(magbins)-1) ] )

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
        
        for iteration in arange(nsamples):
            # Sample with replacement the observed sample
            newsample = np.random.randint(len(self.initial), size = len(self.initial))
            
            # Make relevant lists for new bootstrap sample
            initial = np.array([self.initial[gal] for gal in newsample])
            trimmed_pairs = np.array([self.trimmed_pairs[gal] for gal in newsample])
            selectionMasks = np.array([self.selectionMasks[gal] for gal in newsample])
            PPF_pairs = np.array([self.PPF_pairs[gal] for gal in newsample])
            #Add pair_weights when done
            
            # Calculate fm for sample
            fm_newsample = self.mergerIntegrator(zmin, zmax, initial, trimmed_pairs, 
                                                 selectionMasks,
                                                 PPF_pairs)
            fm_array.append(fm_newsample)
            
        fm_array = np.fm_array
        fm_mean = fm_array.sum() / nsamples
        
        # Estimate StDev for resampled values
        fm_ste = np.sqrt( np.sum((fm_array - fm_mean)**2) / (nsamples - 1) )
        self.fm_ste = fm_ste
        return self.fm, self.fm_ste

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


        Ax.plot( self.OSRmags, self.OSR, 'o', color='w', mew=2, mec='dodgerblue')

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
            
            Ax[1].plot(self.zr,np.log10(self.mz[primary,:]),'--',lw=2,color='dodgerblue',
                       label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[primary],
                                                                   '$z_{peak}$:',
                                                                   self.peakz[primary]))
                                                                   
            Ax[1].plot(self.zr,np.log10(self.mz[secondary,:]),':',lw=2,color='indianred',
                       label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[secondary],
                                                                   '$z_{peak}$:',
                                                                   self.peakz[secondary]))
            
            selmask = np.invert(self.selectionMasks[primary_index][j])
            zr_mask = np.ma.masked_where(selmask, self.zr)
            mz1 = np.ma.masked_where(selmask, np.log10(self.mz[primary,:]) )
            mz2 = np.ma.masked_where(selmask, np.log10(self.mz[secondary,:]) )

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
            Ax[1].set_ylim(6.5,11.5)
            Fig.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.1,hspace=0.25)
            plt.show()

    def plotPairsPz(self,primary_index,legend=True,draw_frame=False):

        primary = self.initial[primary_index]
        secondaries = self.initial_pairs[primary_index]
        
        for j, secondary in enumerate(secondaries):
            Fig, Ax = plt.subplots(2,figsize=(6,9))
            # Plot the primary P(z)
            Ax[0].plot(self.zr,cumtrapz(self.pz[primary,:], self.zr, initial=0),'--',lw=2,color='b',
                        label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[primary_index],
                                                                   '$z_{peak}$:',
                                                                   self.peakz[primary_index]))
            # Plot the secondary P(z)
            Ax[0].plot(self.zr,cumtrapz(self.pz[secondary,:], self.zr, initial=0),':',lw=2,color='r',
                        label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[secondary],
                                                                   '$z_{peak}$:',
                                                                   self.peakz[secondary]))
            # Plot the Z function
            Ax[0].plot(self.zr,cumtrapz(self.redshiftProbs[primary_index][j], self.zr, initial=0), '--k', lw=2.5)
            # Legend
            if legend:
                Leg1 = Ax[0].legend(loc='upper right', prop={'size':8})
                Leg1.draw_frame(draw_frame)
            # Labels, limits
            Ax[0].set_xlabel('Redshift, z')
            Ax[0].set_ylabel(r'$\int P(z)$')
            Ax[0].set_xlim(0,2), Ax[0].set_ylim(0,1.1)
            # Plot the Number of pairs
            Ax[0].text(0.75,0.1, r'{0:s} {1:.3f}'.format('$\mathcal{N}_{z} =$ ', self.Nzpair[primary_index][j]), transform=Ax[0].transAxes)
            
            Fig.subplots_adjust(right=0.95,top=0.95,bottom=0.14)
        
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
