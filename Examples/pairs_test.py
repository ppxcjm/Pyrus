import numpy as np
import PyPairs as p

with np.load('masses/gs_weighted.mass_z.prob.npz') as data:
    zr = data['z']
    masses = data['mass_array']
    
with np.load('masses/gs_weighted.z.prob.npz') as data:
    zr = data['z']
    redshifts = data['z_array']
    
catalog_path = '/data/candels/catalogs/GOODS-S/gs_all_tf_h_130213a_multi.mergers.fits'

GS = p.Pairs(zr, redshifts, masses, catalog_path)

GS.plotPz([10000,30000,4050])
GS.plotMz([10000,30000,4050])