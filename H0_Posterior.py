import healpy as hp
import numpy as np
from astropy.cosmology import FlatLambdaCDM,z_at_value
from astropy.io import fits
import astropy.units as u
import posterior_utils as pos
import sys 
import matplotlib.pyplot as plt
from joblib import Parallel, delayed



#Set up file name imports and column names
DIR_SKYMAP='/global/homes/w/wballard/standard-sirens'
skymap='GW190412_skymap.fits.gz'
DIR_CATALOG='/global/homes/w/wballard/standard-sirens'
DIR_OUT='/global/homes/w/wballard/standard-sirens/out/'
outlabel='H0n_output'
glxcat='reducedTable.fits'
ra_column_name='TARGET_RA'
dec_column_name='TARGET_DEC'
z_column_name='Z'
zerr_column_name='ZERR'
pix_column_name='hpix1024'
ebv_column_name='EBV'
flux_column_name='FLUX_R'
skymap_name=DIR_SKYMAP+skymap
catalog_name = DIR_CATALOG+glxcat

#Assign global variables
H0_min=20
H0_max=140
Omega_m=0.3
z_min=0.001
z_max=0.5
cl=0.68
extinction_coeff=2.165
n_H0_maxlike=[]
n_H0_err_p=[]
n_H0_err_m=[]
n_H0_median=[]
mpi=False

#Open up skymap and read in pixel data
pb,distmu,distsigma,distnorm=hp.read_map(skymap_name,field=range(4),memmap=True)
NSIDE=hp.npix2nside(len(pb))
pixarea=hp.nside2pixarea(NSIDE)
pixarea_deg=hp.nside2pixarea(NSIDE,degrees=True)
pb_frac=0.95

#Open up reduced galaxy catalog and assign catalogs
h=fits.open(catalog_name,memmap=True)[1].data
z_g=h[z_column_name]
ra_g=h[ra_column_name]
dec_g=h[dec_column_name]
zerr_g=h[zerr_column_name]
pix_g=h[pix_column_name]
flux_g=h[flux_column_name]
ebv_g=h[ebv_column_name]
weight_g=np.ones_like(ra_g)

#Correct flux with extinction and calculate absolute magnitude
transmission_mw=ebv_g*extinction_coeff
corrected_flux=flux_g/transmission_mw
omag_g=22.5-2.5*np.log10(corrected_flux)
abs_mag=omag_g-pos.dist_mod(z_g)

#Create credible level map, where each pixel is assigned a cumulative probablity that varies 
#inversely with probability (most likely is 0, least likely is 1)
i = np.flipud(np.argsort(pb))
sorted_credible_levels = np.cumsum(pb[i])
credible_levels = np.empty_like(sorted_credible_levels)
credible_levels[i] = sorted_credible_levels
mask_pix=(credible_levels<=pb_frac)
cred_cut=credible_levels[mask_pix]
cut_idx=i[:len(cred_cut)]
roi_area=np.sum(mask_pix) * hp.nside2pixarea(NSIDE, degrees=True)
ramax,decmax=hp.pix2ang(NSIDE,np.argmax(pb),lonlat=True)
distmu_avg=np.average(distmu[mask_pix],weights=pb[mask_pix])
distsig_avg=np.average(distsigma[mask_pix],weights=pb[mask_pix])

#Set up cosmological parameters and z_min/z_max
cosmomin=FlatLambdaCDM(H0=H0_min,Om0=Omega_m)
cosmomax=FlatLambdaCDM(H0=H0_max,Om0=Omega_m)
nsigma=1.65
H0_array = np.linspace(H0_min,H0_max,num=101)
d_min=distmu_avg-(nsigma*distsig_avg)
d_max=distmu_avg+(nsigma*distsig_avg)
z_min=max(z_min,z_at_value(cosmomin.luminosity_distance,d_min*u.Mpc,zmin=1e-5,zmax=2,method='bounded'))
z_max=min(z_max,z_at_value(cosmomax.luminosity_distance,d_max*u.Mpc,zmin=1e-5,zmax=2,method='bounded'))

#Remove galaxies outside of pixel measurement and absolute magnitude
mask_z = np.logical_and((z_g>z_min),(z_g<z_max))
mask_mag=abs_mag<-21.5
ra_glx=ra_g[mask_z & mask_mag]
dec_glx=dec_g[mask_z & mask_mag]
z_glx=z_g[mask_z & mask_mag]
zerr_glx=zerr_g[mask_z & mask_mag]
pix_glx=pix_g[mask_z & mask_mag]
weight_glx=weight_g[mask_z & mask_mag]/(weight_g[mask_z & mask_mag].sum())

#Find the pixels stored in the most likely region, and mask data galaxies not in region
in_region=np.where(np.isin(pix_glx,cut_idx))
z_gal = z_glx[in_region]
zerr_gal = zerr_glx[in_region]
ra_gal = ra_glx[in_region] 
dec_gal = dec_glx[in_region] 
pix_gal = pix_glx[in_region]
weight_gal=weight_glx[in_region]

# Need some insight on what these do
pb_gal =  np.zeros_like(ra_gal)
distmu_gal =  np.zeros_like(ra_gal)
distsigma_gal =  np.zeros_like(ra_gal)
distnorm_gal = np.zeros_like(ra_gal)
for i in cut_idx:
    this_pix=(pix_gal==i)
    pb_gal[this_pix]=pb[i]
    distmu_gal[this_pix]=distmu[i]
    distsigma_gal[this_pix]=distsigma[i]
    distnorm_gal[this_pix]=distnorm[i]
ngals = ra_gal.shape[0]
print("There are ", str(ngals), " galaxies within ", str(pb_frac*100.), "%, and z between ", z_min, z_max)

zarr = np.linspace(z_min,z_max,num=1+2**10)
gzarr = np.zeros((ngals,zarr.size))
z_norm = np.linspace(z_min,z_max,num=200)
for i in range(ngals):
    g = pos.gauss(z_gal[i],zerr_gal[i],zarr)
    g_norm = pos.gauss(z_gal[i],zerr_gal[i],z_norm)
    norm = 1.e-300+np.trapz(g_norm,z_norm)
    gzarr[i] = g/norm


#Horribly slow, but MPI isn't working
def lnpost(H0value):
        return pos.lnprob(H0value, z_gal, zerr_gal, weight_gal, pb_gal, distmu_gal, distsigma_gal, distnorm_gal, 
                          pixarea, H0_min, H0_max, z_min, z_max, zerr_use=True, cosmo_use=True ,omegam=Omega_m,
                          gauss_z_array=gzarr,z_s=zarr,hpix=pix_gal,d50=638,use_zPDF=False,bias_marginalize=False,
                          bias_mean=0, z_bias_bins_edges=0)
if mpi:
    print("njobs = ", 10)
    lnposterior=Parallel(n_jobs=10)(delayed(lnpost)(H0value) for H0value in H0_array)
else:
    lnposterior=[]
    for H0value in H0_array: 
        lnp = lnpost(H0value)
        lnposterior.append(lnp)

posterior=np.exp(lnposterior)
#Output stuff
blindpath = DIR_OUT+"/blinding_file.p"
H0_blinded_array=pos.make_blind(H0_array,H0_min,H0_max,blindpath)
print("Applying blinding factor. Saving value on ", blindpath)
H0_array_out = H0_blinded_array
H0_maxlike, H0_err_p , H0_err_m  = pos.cl_range(H0_array_out,posterior,cl=cl)
H0_median=pos.percentile(0.50, posterior, H0_array_out)
n_H0_maxlike.append(H0_maxlike)
n_H0_err_p.append(H0_err_p)
n_H0_err_m.append(H0_err_m)
n_H0_median.append(H0_median)
header = "H0_Blinded"
fmt = "%10.5f"


norm = 1.e-300+np.trapz(posterior, H0_array_out)
posterior = posterior/norm 
dl=int(distmu_avg)
plt.plot(H0_array_out, posterior, label="Event"+str(1)+": "+str(dl)+" Mpc")
fmt=fmt+" %10.6e"
header = header+" Posterior_"+str(1)

cols = np.column_stack((H0_array_out,posterior))
header = header.split()[0]+" Posterior"
plt.clf()
dl=int(distmu_avg)
plt.plot(H0_array_out, posterior, color='k', label=str(dl)+" Mpc")

plt.xlabel('$H_0$ [km/s/Mpc]',fontsize=20)
plt.ylabel('$p$',fontsize=20)
plt.title(outlabel)
plt.tight_layout()

plt.savefig(DIR_OUT+outlabel+'_posterior.png')


