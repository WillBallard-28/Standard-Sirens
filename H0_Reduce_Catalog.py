import numpy as np
import healpy as hp
from astropy.io import fits
from math import pi
import sys
from astropy.table import Table
sys.stderr=open('error.log','w',buffering=1)

z_min=0.001
z_max=0.5
njobs=10
systerr=0.001
outlabel='reducedTable.fits'

ra_column_name='TARGET_RA'
dec_column_name='TARGET_DEC'
z_column_name='Z'
zerr_column_name='ZERR'
mag_column_name='OMAG'
absmag_column='AMAG'
deltachi2_column='DELTACHI2'
flux_column_name='FLUX_R'

DIR_SKYMAP='/hildafs/projects/phy220048p/share/Dark_sirens/'
skymap_name='GW190412_skymap'
DIR_CATALOG='/hildafs/projects/phy220048p/share/DESI/'
glxcat='zall-tilecumulative-iron.fits'


NSIDE=1024
nevents=1
nside_cat=8


filename = DIR_CATALOG+glxcat
hdul= fits.open(filename,memmap=True)
h=hdul[1].data

print("Reading in fits data")
deltachi2=h[deltachi2_column]
mask_deltachi=deltachi2>40
zwarn=h['ZWARN']
zwarn_mask=(zwarn==0)
spectype=h['SPECTYPE']
spectype_mask=(spectype=='GALAXY')
z=h[z_column_name]
mask_z=np.logical_and(z>z_min, z<z_max)
bgs=h['BGS_TARGET']
bgs_mask=bgs>0

ra_g=h[ra_column_name][mask_z & mask_deltachi & zwarn_mask & spectype_mask & bgs_mask]
dec_g=h[dec_column_name][mask_z & mask_deltachi & zwarn_mask & spectype_mask & bgs_mask]
z=z[mask_z & mask_deltachi & zwarn_mask & spectype_mask & bgs_mask]
zerr_g=h[zerr_column_name][mask_z & mask_deltachi & zwarn_mask & spectype_mask & bgs_mask]
flux_g=h[flux_column_name][mask_z & mask_deltachi & zwarn_mask & spectype_mask & bgs_mask]
ebv_g=h['EBV'][mask_z & mask_deltachi & zwarn_mask & spectype_mask & bgs_mask]




hdul.close()

pix_g= hp.pixelfunc.ang2pix(NSIDE, ra_g, dec_g,lonlat=True)    

t=Table([ra_g,dec_g,z,zerr_g,pix_g,flux_g,ebv_g],
        names=('TARGET_RA','TARGET_DEC','Z','ZERR', 'hpix1024','FLUX_R','EBV'))
t.write(outlabel,format='fits',overwrite=True)
print('Done writing '+outlabel)