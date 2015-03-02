# Matplotlib, numpy
import matplotlib.pyplot as plt, numpy as np, time
# Astropy
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
# Scipy
from scipy.spatial import cKDTree
from scipy.integrate import simps

class Pairs(object):
    """ Main class for photometric pair-count analysis
    
    Implementation of the Lopez-Sanjuan et al. (2014) redshift PDF pair-count method
    for photometric redshifts from EAZY (or other) and stellar mass estimates using
    Duncan et al. (2014) outputs.
        
    """
    
    def __init__(self, z, redshift_cube, mass_cube, photometry=False, catalog_format = 'fits',
                 idcol = 'ID', racol = 'RA', deccol = 'DEC', cosmology = False):
        """ Load and format appropriately the necessary data for pair-count calculations
        
        Args:
            z (numpy.ndarray): Redshift steps - 1-d array of len(N)
            redshift_cube (numpy.ndarray): Photometric redshift probability
                distribution functions for galaxy sample. 2-d array of shape(M,N)
            mass_cube (numpy.ndarray): Stellar mass as a function of redshift for 
                galaxy sample. 2-d array of shape(M,N)
            photometry (str): Path to photometry catalog
            catalog_format (str): Catalog format, e.g. 'fits' or 'ascii'
            idcol (str): Column name for galaxy IDs
            racol (str): Column name for galaxy RAs
            deccol (str): Column name for galaxy DECs
            H0 (float): Hubble constant in km/Mpc/s.
            OM (float): Dark matter density in units of critical density.
        
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
            print "Redshift and Mass data-cube shapes do not match - please check"
        else:
            self.pz = redshift_cube
            self.mz = mass_cube
        # Set the redshift range array
        self.zr = z
        # Get the most probable redshift solution
        self.peakz = np.array([self.zr[galx] for galx in np.argmax(self.pz,axis=1)])
        self.peakz_arg = np.argmax(self.pz,axis=1)
        # Class photometry path
        self.photometry_path = photometry
        # Attempt to read the catalogue
        try:
            self.phot_catalog = Table.read( self.photometry_path )
        except:
            print "Cannot read photometry catalogue - please check"
        # Class ID, position and co-ordinate arrays
        self.IDs = self.phot_catalog[idcol]
        self.RA = self.phot_catalog[racol] * u.deg
        self.Dec = self.phot_catalog[deccol] * u.deg
        self.coords = SkyCoord(self.RA,self.Dec,frame='icrs')
        # Set up a cosmology for the class to use
        if not cosmology:
            self.cosmo = FlatLambdaCDM( H0=70, Om0=0.3 )
        else:
            self.cosmo = cosmology

    def defineSeparation(self, r_min, r_max):
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
        
    def findInitialPairs(self, z_min = 0.1):
        """ Find an initial list of potential close pair companions based on the already
            defined separations.

            Args:
                z_min (float): Minimum redshift being considered by the work. For calc-
                    ulation of maximum separation.

        """

        # Calculate the maximum separation (in degrees) on the sky
        maxsep = (self.r_max / self.cosmo.angular_diameter_distance( z_min ).to(u.kpc) )*u.rad
        # Set up the binary table of the initial sample's RA and D
        primary_tree = cKDTree( zip(self.RA.value[self.initial], self.Dec.value[self.initial]) )
        # Set up the binary table of the full catalogue's RA and Dec
        self.full_tree = cKDTree(zip(self.RA.value, self.Dec.value)) # Might be worth keeping, maybe not
        # Search for objects within maxsep separation
        self.initial_pairs = primary_tree.query_ball_tree(self.full_tree, maxsep.to(u.deg).value)
        # Remove self-matches
        # Returns an array of length = len(self.initial), each element is a list of potential companion indices.
        for p, i_idx in enumerate(self.initial):
            self.initial_pairs[p].remove(i_idx)
        # Now search for duplicate values
        # ... For each galaxy in the primary sample,
        for i, primary in enumerate(self.initial):
            # Go through its matches...
            for j, secondary in enumerate(self.initial_pairs[i]):
                # If the potential companion is in the primary sample
                if secondary in self.initial:
                    # What are their massses at z(min(chi^2))
                    primary_mass = self.mz[primary, self.peakz_arg[primary]]
                    secondary_mass = self.mz[secondary, self.peakz_arg[secondary]]
                    # Delete things
                    if secondary_mass > primary_mass:
                        # If the companion is more massive, remove the companion from this
                        # ... primary galaxy's initial_pairs list
                        self.initial_pairs[i].remove(secondary)

        self.initial_pairs = np.array(self.initial_pairs)

    def makeMasks(self, mass_cut, mass_ratio = 4.):
        """ Make the various masks to enforce angular separation, stellar mass ratio and
            selection conditions. Also produce the PPF(z).

            Args:
                mass_cut (float or float-array): Defines the stellar mass cut to be included
                    in the primary sample.
                mass_ratio (float): Ratio of stellar masses to be considered a pair.

        """

        dA_z = self.cosmo.angular_diameter_distance(self.zr).to(u.kpc)

        z_msks, sep_msks, sel_msks, Nzpair = [], [], [], []

        for i, primary in enumerate( self.initial ):

            primary_pz = self.pz[ primary, :]
            primary_mz = self.mz[ primary, :]
            Zz_arrays, Zpair_fracs = [], []
            sep_arrays, sel_arrays = [], []

            for j, secondary in enumerate(self.initial_pairs[i]):

                # Redshift probability
                # -----------------------------------------
                secondary_pz = self.pz[ secondary, :]
                Nz = (primary_pz + secondary_pz) * 0.5
                Zz = (primary_pz * secondary_pz) / Nz
                Zz_arrays.append( Zz )
                Zpair_fracs.append( simps(Zz,self.zr) )

                # Separation masks
                # -----------------------------------------
                # Sepration (in degrees) between primary and secondary
                d2d = self.coords[primary].separation(self.coords[secondary]).to(u.deg)
                # Min/max angular separation as a function of redshift
                theta_min = ((self.r_min / dA_z)*u.rad).to(u.deg)
                theta_max = ((self.r_max / dA_z)*u.rad).to(u.deg)
                # Create boolean array
                sep_msk = np.logical_and( d2d >= theta_min , d2d <= theta_max )
                sep_arrays.append( sep_msk )

                # Selection masks
                # ----------------------------------------- 
                secondary_mz = self.mz[ secondary, :]
                # Create the boolean array enforcing conditions
                sel_msk = np.logical_and(primary_mz >= mass_cut, (primary_mz/secondary_mz) <= mass_ratio)
                sel_arrays.append( sel_msk )
            
            sel_msks.append( sel_arrays )
            z_msks.append( Zz_arrays )
            Nzpair.append( Zpair_fracs )
            sep_msks.append( sep_arrays )

        # Set class variables
        self.redshiftProbs = np.array( z_msks )
        self.separationMasks = np.array( sep_msks )
        self.selectionMasks = np.array( sel_msks )

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
        if legend:
            Leg = Ax.legend(loc='upper right', prop={'size':8})
            Leg.draw_frame(draw_frame)
        
        Ax.set_xlabel('Redshift, z')
        Ax.set_ylabel(r'$P(z)$')
        plt.tight_layout()
        plt.show()
    
    def plotMz(self,galaxy_indices,legend=True,draw_frame=False):
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