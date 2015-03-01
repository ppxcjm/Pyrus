import matplotlib.pyplot as plt
import numpy as np
import atpy
import astropy.unit as u
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
from scipy.integrate import trapz, simps

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
        self.coords = SkyCoord(self.RA,self.Dec,frame='icrs')
        
    def initialSample(self,intitial):
        """ Define and set up initial sample of galaxies in which to find pairs
        
        Initial sample definition is done OUTSIDE of class and then loaded to allow
        for different sample definition criteria for respective papers. Alleviate need
        for multiple different functions.
        
        Args:
            initial (1d array): Indexes of galaxies which satisfy the initial sample
                criteria.
        
        """
        self.initial  = initial
        
        
    def findPairs(self,maxsep,minsep=0,units=u.arcsecond):
        """ 
        Docs
        """
        
        self.initial_pairs = []
        
        
        for i, gal in enumerate(self.coords[self.initial]):
            d2d = gal.separation(self.coords)
            catalogmsk = (minsep*units < d2d)*(d2d < maxsep*units)
            idxcatalog = np.where(catalogmsk)[0]
            self.initial_pairs.append(idxcatalog)

    def findPairs2(self,maxsep,minsep=0,units=u.arcsecond):
        """ 
        Docs
        """
        sample_tree = cKDTree( zip(self.RA.value[self.initial], self.Dec.value[self.initial]) )
        
        self.full_tree = cKDTree(zip(self.RA.value, self.Dec.value)) # Might be worth keeping, maybe not
        
        self.initial_pairs = sample_tree.query_ball_tree(self.full_tree, 
                                                         (maxsep*units).to('degree').value)
        initial_tooclose = sample_tree.query_ball_tree(self.full_tree, 
                                                       (minsep*units).to('degree').value)
        # Individual sets of matches need sorting before comparing to findPairs brute force

        # Remove both self-matches and matches below min separation
        for i, primary in enumerate(self.initial):
            for gal in initial_tooclose[i]:
                del self.initial_pairs[i][self.initial_pairs[i] == gal]
            
            self.initial_pairs[i] = numpy.sort(self.initial_pairs[i])
            # Sort so it matches brute force output
    
    def trimPairs(self):
        """ Remove duplicate pairs for initial target sample
            
            Steps being taken:
            ------------------
            If secondary in initial:
                - Check which is the highest mass of the pair
                - If primary is lower mass partner
                     - remove secondary from pair array
                - Else if primary is higher mass (i.e. still primary)
                        Find pairs array for secondary
                            - remove primary from pair array
        """

        for i, primary in enumerate(self.initial):
            for j, secondary in enumerate(self.initial_pairs[i]):
                if secondary in self.initial:
                    primary_mass = self.mz[self.peakz_arg[primary],:]
                    secondary_mass = self.mz[self.peakz_arg[secondary],:]
                    
                    if secondary_mass > primary_mass:
                        del self.initial_pairs[i][j]
                    else:
                        k = numpy.where(self.initial == secondary)
                        del self.initial_pairs[k][self.initial_pairs[k] == primary]

    def redshiftMask(self):
        msks = []
        Nzpair = []
        
        for i, primary in enumerate(self.initial):
            primary_pz = self.pz[primary,:]
            Zz_arrays = []
            Zpair_fracs = []
            for j, secondary in enumerate(self.initial_pairs[i]):
                secondary_pz = self.pz[secondary,:]
                Nz = (primary_pz + secondary_pz) * 0.5
                Zz = (primary_pz * secondary_pz) / Nz
                Zz_arrays.append(Zz)
                Zpair_fracs.append(simps(Zz,self.zr))
            
            msks.append(Zz_array)
            Nzpair.append(Zpair_fracs)
        self.redshiftMasks = msks
        self.Nzpair = Nzpair

    def separationMask(self,rmax=50,rmin=5,runits = u.kpc):
        msks = []
        
        for i, primary in enumerate(self.initial):
            sep_arrays = []
            for j, secondary in enumerate(self.initial_pairs[i]):
                dA_z = cosmo.angular_diameter_distance(self.zr).to(runits)
                d2d = self.coords[primary].separation(self.coords[secondary]).value
                sep_msk = np.logical_and( (rmin*u.kpc / dA_z) <= d2d , d2d <= (rmax*u.kpc / dA_z))
                sep_arrays.append(sep_msk)
            msks.append(sep_arrays)
        self.separationMasks = msks        
        
    def selectionMask(self,mass_cut,mass_ratio=4):
        msks = []
        
        for i, primary in enumerate(self.initial):
            sel_msks = []
            primary_mz = self.mz[primary,:]
            for j, secondary in enumerate(self.initial_pairs[i]):
                secondary_mz = self.mz[secondary,:]
                sel_msk = np.logical_and(primary_mz >= mass_cut, (primary_mz/secondary_mz) <= mass_ratio)
                sel_msks.append(sel_msk)
                
            msks.append(sel_msks)
        self.selectionMasks = msks
        
    def makeMasks(self,mass_cut,mass_ratio=4,rmax=50,rmin=5,runits=u.kpc):
        """ If individual mask functions above work correctly. Will add into
            single function for efficiency since they follow the same loop structure
            and don't necessarily need to be separate.
        
        """
        return None 
       
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
        
    def plotPairs(self,primary_index,legend=True,draw_frame=False):
        primary = self.initial[primary_index]
        secondaries = self.initial_pairs[primary_index]
        
        for j, secondary in enumerate(secondaries):
            Fig, Ax = plt.subplots(2,figsize=(4.5,8))
            Ax[0].plot(self.zr,self.pz[primary,:],'--',lw=2,color='dodgerblue')
            Ax[0].plot(self.zr,self.pz[secondary,:],':',lw=2,color='indianred')
            if legend:
                Leg1 = Ax[0].legend(loc='upper right', prop={'size':8})
                Leg1.draw_frame(draw_frame)
        
            Ax[0].set_xlabel('Redshift, z')
            Ax[0].set_ylabel(r'$P(z)$')
            Ax[0].text(0.1,0.9, r'{0:s} {1:.2f}'.format('$\mathcal{N}_{z} =$ ',self.Nzpair[primary_index][j]))
            
            Ax[1].plot(self.zr,np.log10(self.mz[primary,:]),'--',lw=2,color='dodgerblue',
                       label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[primary_index],
                                                                   '$z_{peak}$:',
                                                                   self.peakz[primary_index]))
                                                                   
            Ax[1].plot(self.zr,np.log10(self.mz[secondary,:]),':',lw=2,color='indianred'
                       label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[secondary],
                                                                   '$z_{peak}$:',
                                                                   self.peakz[secondary]))
            
            selmask = self.selectionMasks[primary_index][j]
            
            Ax[1].plot(self.zr,np.log10(self.mz[primary,:])[selmask],lw=4,color='dodgerblue',
                       label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[primary_index],
                                                                   '$z_{peak}$:',
                                                                   self.peakz[primary_index]))
            
            Ax[1].plot(self.zr,np.log10(self.mz[secondary,:])[selmask],lw=4,color='indianred',
                       label = r'ID: {0:.0f} {1:s} {2:.2f}'.format(self.IDs[secondary],
                                                                   '$z_{peak}$:',
                                                                   self.peakz[secondary]))
            if legend:
                Leg2 = Ax[1].legend(loc='lower right', prop={'size':8})
                Leg2.draw_frame(draw_frame)
        
            Ax[1].set_xlabel('Redshift, z')
            Ax[1].set_ylabel(r'$\log_{10} \rm{M}_{\star}$')
            Ax[1].set_ylim(6.5,11.5)
            Fig.subplots_adjust(right=0.95,top=0.95,bottom=0.14)
            plt.show()