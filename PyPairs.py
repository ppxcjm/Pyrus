import time
import matplotlib.pyplot as plt
import numpy as np
import atpy
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle

from scipy.spatial import cKDTree
from scipy.integrate import trapz 
from scipy.integrate import simps

class Pairs(object):
    """ Main class for photometric pair-count analysis
    
    Implementation of the Lopez-Sanjuan et al. (2014) redshift PDF pair-count method
    for photometric redshifts from EAZY (or other) and stellar mass estimates using
    Duncan et al. (2014) outputs.
        
    """
    
    def __init__(self, z, redshift_cube, mass_cube, photometry, cosmology, catalog_format = 'fits',
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
        self.peakz_arg = np.argmax(self.pz, axis=1)
        self.peakz = self.zr[self.peakz_arg]
        

        self.photometry_path = photometry
        self.phot_catalog = atpy.Table(self.photometry_path, type='fits')
        self.IDs = self.phot_catalog[idcol]
        self.RA = self.phot_catalog[racol] * u.degree
        self.Dec = self.phot_catalog[deccol] * u.degree
        self.coords = SkyCoord(self.RA, self.Dec, frame='icrs')
        self.cosmo = cosmology
        
    def initialSample(self,initial):
        """ Define and set up initial sample of galaxies in which to find pairs
        
        Initial sample definition is done OUTSIDE of class and then loaded to allow
        for different sample definition criteria for respective papers. Alleviate need
        for multiple different functions.
        
        Args:
            initial (1d array): Indexes of galaxies which satisfy the initial sample
                criteria.
        
        """
        self.initial  = initial
        

    def findPairs(self, maxsep, minsep=0, units=u.arcsecond):
        """ Find close pairs with an angular separation between minsep and maxsep
        
        Makes use of cKDTree to vastly speed up computation - output is identical
        to that of brute force method.
        
        """
        start = time.time()
        
        # Convert Sky Coordinates to cartesian xyz for correct 3d distances
        cartxyz = self.coords.cartesian.xyz
        flatxyz = cartxyz.reshape((3, np.prod(cartxyz.shape) // 3))
        
        sample_tree = cKDTree( flatxyz.value.T[self.initial] ) 
        
        # Convert on-sky angular separation to matching cartesian 3d distance
        # (See astropy.coordinate documentation for seach_around_sky)
        # If changing function input to distance separation and redshift, MUST convert
        # to an angular separation before here.
        r_maxsep = (2 * np.sin(Angle(maxsep * units) / 2.0)).value
        r_minsep = (2 * np.sin(Angle(minsep * units) / 2.0)).value
        
        
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

        print( 'Time taken: {0:.2f} s'.format(time.time() - start) )


    def findPairs2(self,maxsep,minsep=0,units=u.arcsecond):
        """ Find close pairs with an angular separation between minsep and maxsep
        
        Uses brute force loop to find all galaxies in target separation range
        for each galaxy in the initial sample. Of order ~500-1000x slower than cKDTree
        """
        start = time.time()
        self.initial_pairs = []
        
        
        for i, gal in enumerate(self.coords[self.initial]):
            d2d = gal.separation(self.coords).to(u.arcsecond)
            catalogmsk = (minsep*units < d2d) * (d2d < maxsep*units)
            idxcatalog = np.where(catalogmsk)[0]
            self.initial_pairs.append(idxcatalog)

        print('Time taken: {0:.2f} s'.format(time.time() - start))
    
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

    def redshiftMask(self):
        
        msks = []
        Nzpair = []
        
        for i, primary in enumerate(self.initial):
            # Re-enforce normalisation of the P(z)
            primary_pz = self.pz[primary,:]
            primary_pz /= simps(primary_pz, self.zr)
            
            Zz_arrays = []
            Zpair_fracs = []
            
            for j, secondary in enumerate(self.trimmed_pairs[i]):
                secondary_pz = self.pz[secondary,:]
                secondary_pz /= simps(secondary_pz, self.zr)
                
                # Calculate convolved redshift distributions
                Nz = (primary_pz + secondary_pz) * 0.5
                Zz = (primary_pz * secondary_pz) / Nz
                Zz_arrays.append(Zz)
                Zpair_fracs.append(simps(Zz,self.zr))
            
            msks.append(Zz_arrays)
            Nzpair.append(Zpair_fracs)
        self.redshiftMasks = msks
        self.Nzpair = Nzpair

    def separationMask(self,rmax=50,rmin=5,runits = u.kpc):
        msks = []
        dA_z = self.cosmo.angular_diameter_distance(self.zr).to(runits)
        print 'dA calculated \n'
        print 'Calculating separation masks... ',
        
        for i, primary in enumerate(self.initial):
            print primary, ': ',
            sep_arrays = []
            d2d = self.coords[primary].separation(self.coords[self.trimmed_pairs[i]]).to(u.rad)
            # Calculate angular separation for all pairs around primary
            theta_min = (rmin*runits / dA_z)*u.rad
            theta_max = (rmax*runits / dA_z)*u.rad
            
            for j, secondary in enumerate(self.trimmed_pairs[i]):
                print secondary,
                # Apply upper and lower criteria along redshift array
                sep_msk = np.logical_and(theta_min <= d2d[j], d2d[j] <= theta_max)
                sep_arrays.append(sep_msk)
                
            msks.append(sep_arrays)
            print ''
        
        print 'Done'
        self.separationMasks = msks        
        
    def selectionMask(self, mass_cut, mass_ratio=4):
        msks = []
        
        for i, primary in enumerate(self.initial):
            print primary, ': ',
            sel_msks = []
            primary_mz = self.mz[primary, :]
            
            for j, secondary in enumerate(self.trimmed_pairs[i]):
                print secondary,
                secondary_mz = self.mz[secondary, :]
                sel_msk = np.logical_and(primary_mz >= mass_cut, (primary_mz / secondary_mz) <= mass_ratio)
                sel_msks.append(sel_msk)
                
            msks.append(sel_msks)
            print ''
        self.selectionMasks = msks
        
    def makeMasks(self, mass_cut, mass_ratio=4, rmax=50, rmin=5, runits=u.kpc):
        """ If individual mask functions above work correctly. Will add into
            single function for efficiency since they follow the same loop structure
            and don't necessarily need to be separate.
        
        """
        return None
        
    def PPF(self):
        """ Function to calculate un-weighted PPF
        
        Temporary function to calculate integrated PPF without area or redshift
        quality weights.
        
        """
        self.PPF_total = [] # Sum over all pairs for each primary
        self.PPF_pairs = [] # Arrays of PPF for each pair
        
        for i, primary in enumerate(self.initial):
            PPF_temp = []
            for j, secondary in enumerate(self.trimmed_pairs[i]):
                ppf_z = (self.redshiftMasks[i][j] * self.selectionMasks[i][j] * 
                         self.separationMasks[i][j])
                         
                PPF_temp.append(simps(ppf_z, self.zr))
                
            self.PPF_pairs.append(PPF_temp)
            self.PPF_total.append(np.sum(PPF_temp))
            
    def denominator(self,mass_cut):
        """ Temporary function
        
        """
        msks = []
        
        for i, primary in enumerate(self.initial):
            sel_msks = []
            primary_mz = self.mz[primary, :]
            primary_pz = self.pz[primary, :]
            sel_msk = np.array((primary_mz >= mass_cut))
                
            msks.append(simps(primary_pz * sel_msk, self.zr))
        self.bottom = msks  
       
    def plotPz(self,galaxy_indices, legend=True, draw_frame=False):
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
        
    def plotPairs(self,primary_index, legend=True, draw_frame=False):
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