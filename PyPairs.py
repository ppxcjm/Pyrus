import matplotlib.pyplot as plt
import numpy as np
import atpy


class Pairs(object):
    """ Main class for photometric pair-count analysis
    
    Implementation of the Lopez-Sanjuan et al. (2014) redshift PDF pair-count method
    for photometric redshifts from EAZY (or other) and stellar mass estimates using
    Duncan et al. (2014) outputs.
        
    """
    
    def __init__(self, z, redshift_cube, mass_cube, photometry, catalog_format = 'fits',
                 idcol = 'ID', racol = 'RA', deccol = 'DEC'):
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
        
        Returns:
            Public attributes created and appropriately formatted for later 
                use and access
        
        """
                 
        redshift_cube = np.array(redshift_cube)
        mass_cube = np.array(mass_cube)
        if redshift_cube.shape != mass_cube.shape:
            print "Redshift and Mass data-cube shapes do not match - please check"
        else:
            self.pz = redshift_cube
            self.mz = mass_cube
        
        self.zr = z
        self.peakz = np.array([self.zr[gal] for gal in np.argmax(self.pz,axis=1)])
        

        self.photometry_path = photometry
        self.phot_catalog = atpy.Table(self.photometry_path,type='fits')
        self.IDs = self.phot_catalog[idcol]
        self.RA = self.phot_catalog[racol]
        self.Dec = self.phot_catalog[deccol]
        
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
        Fig.subplots_adjust(right=0.95,top=0.95,bottom=0.14)
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