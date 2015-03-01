import matplotlib.pyplot as plt, numpy as np, time
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import cKDTree

class Pairs(object):
    """ Main class for photometric pair-count analysis
    
    Implementation of the Lopez-Sanjuan et al. (2014) redshift PDF pair-count method
    for photometric redshifts from EAZY (or other) and stellar mass estimates using
    Duncan et al. (2014) outputs.
        
    """
    
    def __init__(self, z, redshift_cube, mass_cube, photometry=False, catalog_format = 'fits',
                 idcol = 'ID', racol = 'RA', deccol = 'DEC', H0 = 70., OM = 0.3):
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
        # Class photometry path
        self.photometry_path = photometry
        # Attempt to read the catalogue
        try:
            self.phot_catalog = Table.read(self.photometry_path)
        except:
            print "Cannot read photometry catalogue - please check"
        # Class ID, position and co-ordinate arrays
        self.IDs = self.phot_catalog[idcol]
        self.RA = self.phot_catalog[racol] * u.deg
        self.Dec = self.phot_catalog[deccol] * u.deg
        self.coords = SkyCoord(self.RA,self.Dec,frame='icrs')
        # Set up a cosmology for the class to use
        self.cosmo = FlatLambdaCDM(H0=H0, Om0=OM)

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
        self.initial = initial
        
    def findPairs(self,maxsep,units=u.arcsecond):
        """ 
        Docs
        """
        sample_tree = cKDTree( zip(self.RA.value[self.initial], self.Dec.value[self.initial]) )
        
        self.full_tree = cKDTree(zip(self.RA.value, self.Dec.value)) # Might be worth keeping, maybe not
        
        self.initial_pairs = sample_tree.query_ball_tree(self.full_tree, (maxsep*units).to('degree').value)
        # Individual sets of matches need sorting before comparing to findPairs brute force

        print self.initial_pairs

    def findPairsAP(self, maxsep):

        maxsep = maxsep*u.arcsecond
        maxsep = maxsep.to(u.deg)

        primary_cat = SkyCoord( ra=self.RA[self.initial], dec=self.Dec[self.initial])

        seps = []
        for gal in primary_cat:
            tmpseps = self.coords.separation( gal )
            seps.append( np.where((tmpseps < maxsep))[0] )


        print seps



    def redshiftProb(self,p1x,p2x):
        """ Generate the redshift probability function, Z(z), for two galaxies

        Args:
            p1x (int): Index of the central galaxy.
            p2x (int): Index of the companion galaxy.

        """

        top = self.pz[p1x,:] * self.pz[p2x,:]
        bot = 0.5 * (self.pz[p1x,:] + self.pz[p2x,:])

        return top/bot

    def genAngularMask(self,thetas):
        """

        """
        theta_minz = self.r_min / self.cosmo.angular_diameter_distance(self.zr)
        theta_maxz = self.r_max / self.cosmo.angular_diameter_distance(self.zr)
        thetas_cube = thetas * np.ones( (len(thetas), len(self.zr) ) )
        angularMask = np.array( (thetas_cube <= theta_maxz) * (thetas_cube >= theta_minz) )
        return angularMask

    def genMassMask(self,cidx,pidx):
        """ Generate the stellar mass mask that defines the merger ratio of interest.

        Args:
            cidx (int or 1-d array): Central galaxy indices.
            pidx (int or 1-d array): Pair/companion galaxy indices.

        """

        if cidx.shape != pidx.shape:
            print 'Cannot generate mass mask! Index arrays not the same shape - please check'
        cmz, pmz = self.mz[cidx,:], self.mz[pidx,:]


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