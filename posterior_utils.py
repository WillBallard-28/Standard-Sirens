import numpy as np
from scipy.constants import c #m/s
from math import pi
from scipy.stats import norm
from astropy.cosmology import FlatLambdaCDM, z_at_value
import scipy.integrate as integrate
from scipy.integrate import romb
import pickle
import os 
import astropy.units as u 
from scipy import interpolate
import sklearn.gaussian_process as gp
import matplotlib.pyplot as plt
import time
#################################
########### H0 utils ############
#################################

def gauss(mu, std, x):
    return np.exp(-(x-mu)*(x-mu)/(2*std*std))/(std*(2.*pi)**0.5)

def percentile(perc, pdf, xarray): #May not be perfect due to binning... Valid only for regularly spaced xarrays
    return xarray[perc_idx(perc,pdf)]

def perc_idx(perc, pdf): #May not be perfect due to binning... Valid only for regularly spaced xarrays
    if perc == 1. : return len(pdf)-1
    if perc == 0. : return 0
    sum_pdf = 0.
    idx = 0
    norm = pdf.sum()
    if norm == 0 : return idx
    pdf = pdf/norm
    while (sum_pdf<perc):
        idx=idx+1
        if idx == len(pdf): return idx
        sum_pdf = pdf[:idx].sum()
    return max(idx,0)


def cl_range(x,post,cl=0.68):
    cdf=np.cumsum(post)
    cdf /= cdf[-1]
    interp=interpolate.interp1d(cdf,x)
    l = (1.0-cl)*0.5
    u = 1.0-l
    lower = interp(l)[()]
    upper = interp(u)[()]
    peak = x[np.argmax(post)]
    lower_err=abs(peak-lower)
    upper_err=abs(upper-peak)
    
    return peak, upper_err, lower_err


def make_blind(H0_true, H0_min, H0_max, outpath):
    '''
    This function check if the binary blinding factor file is already
    created. If it is: read and apply the blinding factor on H0. If not:
    generate the factor, save and apply.
    '''

    if os.path.isfile(outpath):
        #alpha, c = pickle.load(open(outpath,encoding='bytes')) #"rb"
        #Changed to work with Python3:
        with open(outpath,'rb') as f: 
            alpha,c=pickle.load(f,encoding='bytes')          
    else:
        # Random blinding factors
        c = np.random.uniform(-0.5, 0.5)
        alpha = np.random.uniform(0.75, 1.25)
        
        #Saving the blinding factors in a binary file    
        outarr = np.array([alpha, c])
        pickle.dump(outarr, open(outpath, "wb"))

    # transform to (-1,1)
    x = 2.*(H0_true-H0_min)/(H0_max-H0_min)-1.

    # transform to -inf,inf
    x = np.arctanh(x)

    # rescale
    x = alpha*x + c
        
    # transform back to (-1,1)
    x = np.tanh(x)

    #transform back to full parameter range
    H0_blinded = 0.5*(1.+x)*(H0_max-H0_min)+H0_min
    return H0_blinded

def selection_function(x,mu=638.,sigma=50.,lowlim=0.,highlim=1000.):
    return np.heaviside(x-lowlim, 0.5) * np.heaviside(highlim-x, 0.5) * 1/(np.exp((x-mu)/sigma)+1)

#### The likelihood has the option to have delta functions or Gaussians for redshifts, and the luminosity distance can be computed from Flat lambdaCDM or simple Hubble constant. All options are left so that the user can choose how quick the likelihood will be computed, depending also on the redshift range

def lnlike(H0, z, zerr, weight, pb_gal, distmu, diststd, distnorm, H0_min, H0_max, z_min, z_max, zerr_use, cosmo_use, omegam, gauss_z_array, z_s, hpix,d50,use_zPDF, bias_marginalize, mean_bias, z_bias_bins_edges):

    if zerr_use or use_zPDF: 
        pem = gauss_z_array
    else:
        pem = np.ones_like(gauss_z_array)

    #A couple of choices to make
    approx_pgw= False
    use_evidence = False
    testing= False

    cosmo = FlatLambdaCDM(H0=H0, Om0=omegam)

    dv = cosmo.differential_comoving_volume(z_s).value 
    ngals=z.size

    zmax_gw = z_at_value(cosmo.luminosity_distance, d50 * u.Mpc,zmin=1e-5,zmax=2,method='bounded')  
    vmax_gw = (cosmo.comoving_volume(zmax_gw).value)

    if approx_pgw:

        zgrid = np.logspace(np.log10(z_min),np.log10(z_max), 50)
        dgrid = cosmo.luminosity_distance(zgrid).value
        
        zmu = np.interp(distmu, dgrid, zgrid)
        zstd = (diststd / (1+zmu)) * (1000./c) * cosmo.H(zmu).value 
        pgw = pb_gal * distnorm * gauss(zmu,zstd,z) * weight
        #Tesing here
        #pgw = np.zeros((ngals, z_s.shape[0]))
        #for k in range(ngals):
        #    pgw[k,:] = pb_gal[k] * distnorm[k] * gauss(zmu[k],zstd[k],z_s) * weight[k]


    else:    
        pgw = np.zeros((ngals, z_s.shape[0]))
        d_s = cosmo.luminosity_distance(z_s).value
        for k in range(ngals):
            pgw[k,:] = pb_gal[k] * distnorm[k] * gauss(distmu[k],diststd[k],d_s) * weight[k] #*(H0/100)

    
    #AP this is older
    #d_gal = cosmo.luminosity_distance(z).value
#    pgw = np.zeros((ngals,zgrid.size))
#    z_norm = np.logspace(0.,2.,num=50)
#    for i in range(ngals):
#        g = gauss(zmu[i],zstd[i],z_norm)
#        norm = np.trapz(g,z_norm)
#        g = gauss(zmu[i],zstd[i],zgrid)/norm
#        pgw[i] = np.interp(z_s,zgrid,g)

#    w = pb_gal * distnorm * weight 
#    w_pem = np.sum(w[:,None] * pem, axis=0) 

#    w_pem_pgw_dv = w_pem * pgw * dv

#    like_gals = integrate.simps(w_pem_pgw_dv,z_s) / vmax_gw #* (H0/100.)

#    lnlike_sum = np.log(like_gals)

    savedir = '/data/des60.b/data/palmese/Dark_sirens/GW_standard_sirens/'

    beta = vmax_gw

    #Use evidence
    if (use_evidence==True):
        evidence = np.zeros(ngals)
        if zerr_use or use_zPDF:
            for i in range(ngals): #pem_new.shape[0]):
                if (pem[i,:].sum()>1e-12):
                    evidence[i] = integrate.simps(pem[i,:]*dv, z_s) #np.trapz(pem[i,:] * dv,x=z_s)
                    pem[i,:] = pem[i,:]/evidence[i]
        else:
            for i in range(ngals):
                evidence[i] = cosmo.differential_comoving_volume(z[i]).value
                pem[i,:] = pem[i,:]/evidence[i]


    #Marginalization over photoz bias 
    if bias_marginalize:
        print("Marginalizing over photoz bias")
        dz_s = np.arange(-0.2,0.2,0.001)      
        #Want to get 1 array in z per galaxy which is the new pem
        pem_bias = np.zeros((ngals,z_s.shape[0],dz_s.shape[0]))
        #If mean_bias is an array with z, we pick the galaxies based on ZMEAN 
        #and shift their pdf by the corresponding amount
        if (len(mean_bias)>1): #isinstance(mean_bias, list):
            nbins_zbias = len(mean_bias)
            mean_bias=np.array(mean_bias)
            idx_bin = np.digitize(z,z_bias_bins_edges)-1
            bias_gal = mean_bias[idx_bin]
            print("Photo-z bias assigned in z bins")
        else:
            bias_gal = np.full(ngals,mean_bias)    
            print("Photo-z bias not dependant on z")
        sigma_bias_gal = abs(bias_gal) #this is the same as the mean for now, distance from zspec is effectively both a bias and an error
        for igal in range(ngals): 
           
            bias_arr = gauss(bias_gal[igal], sigma_bias_gal[igal], dz_s)  
            
            for idz in range(dz_s.shape[0]):
                pem_new = np.interp(z_s,z_s-dz_s[idz],pem[igal]) #Shift pem by -deltaz
                pem_bias[igal,:,idz] = pem_new*bias_arr[idz]
            #Commenting out to bring the integral outside of the marginalization over galaxies
            #for k in range(z_s.shape[0]):
            #    pem[igal] = integrate.simps(pem_bias[k,:],dz_s)           
        pgw_pem_bias = pgw[:, None, None] * pem_bias

        pgw_pem_bias_sum = np.sum(pgw_pem_bias,axis=0)

        pgw_pem_sum = np.zeros(z_s.shape[0])
        for k in range(z_s.shape[0]):
            pgw_pem_sum[k] = integrate.simps(pgw_pem_bias_sum[k,:],dz_s)  

    else:
        if approx_pgw:
            pgw_pem = pgw[:, None] * pem
            #pgw_pem = pgw * pem
        else:
            pgw_pem = pgw * pem

        pgw_pem_sum = np.sum(pgw_pem,axis=0)


    pgw_pem_sum_dv = pgw_pem_sum  * dv 

    like_gals = integrate.simps(pgw_pem_sum_dv,z_s) / beta 

    lnlike_sum = np.log(like_gals)

    if testing and ((H0>30.)&(H0<45))|((H0>65.)& (H0<85.)):
        savedir = '/hildafs/projects/phy220048p/wballard/GW_standard_sirens/out/'
        np.savetxt(savedir+'pgw'+str(H0)+'.txt',pgw)
        np.savetxt(savedir+'pem.txt',pem)
    #    if (use_evidence==True): np.savetxt(savedir+'evidence.txt',evidence)
        #print pgw_pem_sum
        np.savetxt(savedir+'pgw_pem_sum'+str(H0)+'.txt',pgw_pem_sum)

# distgal = cosmo.luminosity_distance(z).value
#    like_gals = pb_gal * distnorm * gauss(distmu,diststd,distgal) * dV
#
#    if zerr_use:
#        z_s_bins = np.zeros(z_s.shape[0]+1)
#        z_s_step = (z_s[1]-z_s[0])
#        z_s_bins[0] = z_s[0]-z_s_step/2.
#        for i in range(1,z_s_bins.shape[0]): z_s_bins[i] = z_s_bins[i-1]+z_s_step
#        z_idx=np.digitize(z,z_s_bins)
#        f=np.zeros_like(z_s)
#        for p in np.unique(hpix):
#            hpix_idx=np.where(hpix==p)[0]
#            like_gals_hpix=like_gals[hpix_idx]
#            z_idx_hpix=z_idx[hpix_idx]
#            for j in np.unique(z_idx_hpix):
#                f[j-1]=np.median(like_gals_hpix[z_idx_hpix==j])
#            for i in hpix_idx:
#                like_gals[i]=np.convolve(f,gauss_z_array[i],mode='full')[2*(z_idx[i]-1)] 
#
#    try:
#        if (weight==0): weight=0
#    except:
#        like_gals = like_gals * weight 
#
#    zmax = z_at_value(cosmo.luminosity_distance, d50 * u.Mpc)  
#    #f = 1./(1+zmax)**3 
#    f = 1./(cosmo.comoving_distance(zmax).value)**3
#
#    lnlike_sum = np.log(1.e-300+np.sum(like_gals)*f)

    #return only lnlikesum after  tests
    return lnlike_sum

#### Flat prior ####

def lnprior(H0, H0_min, H0_max):
	if H0_min < H0 < H0_max:
		return 0.0
	return -np.inf

##### Posterior ####

def lnprob(H0, z, zerr, weight, pb_gal, distmu, diststd, distnorm, pixarea, H0_min, H0_max, z_min, z_max, zerr_use, cosmo_use,omegam,gauss_z_array,z_s, hpix,d50,use_zPDF,bias_marginalize,bias_mean, z_bias_bins_edges):
	lp = lnprior(H0, H0_min, H0_max)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(H0, z, zerr, weight, pb_gal, distmu, diststd, distnorm, H0_min, H0_max, z_min, z_max, zerr_use, cosmo_use,omegam,gauss_z_array,z_s, hpix,d50,use_zPDF, bias_marginalize,bias_mean, z_bias_bins_edges)


# absolue mag calculation, from Will Hartley
def dist_mod(z_vals,d='/global/homes/w/wballard/standard-sirens/'):
   z = np.load(d+'dist_mod_kcorr.npz')['z']
   dm = np.load(d+'dist_mod_kcorr.npz')['dm']
   f_dm = interpolate.interp1d(z, dm, kind='cubic', bounds_error=False, fill_value=0)
   return f_dm(z_vals)

### Simplified absolute mag calculation
def lumn_dist(z):
    H0 = 67.74
    h = H0/100.
    WV = 0.6911
    WR = 4.165E-5/(h*h)
    WM = 1. - WV - WR
    c = 299792.458
    n = 10000
    az = 1.0/(1.+z)
    DCMR = 0.
    DTT = 0.
    for i in range(n):
        a = az+(1-az)*(i+0.5)/n
        adot = np.sqrt((WM/a)+(WR/(a*a))+(WV*a*a))
        DTT = DTT + 1./adot
        DCMR = DCMR + 1./(a*adot)
    DCMR = (1.-az)*DCMR/n
    DA = az*DCMR
    DL = DA/(az*az)
    DL_Mpc = (c/H0)*DL
    # convert to distance modulus and return
    DL_mag = 5.*np.log10(DL_Mpc*1.e5) - 2.5*np.log10(1.+z)
    return DL_mag

def emu_z(cat, zcol='DNF_ZMEAN_SOF', sig_col='DNF_ZSIGMA_SOF',
          d='/data/des70.a/data/marcelle/GW170814/catalogs/',
          filename='PIT_corr.npz'):

    # Get the spread factors and arrays
    z = np.load(d+filename)['z']
    sig_factor = np.load(d+filename)['sig_factor']

    # define the covariance kernel
    len_scl = [0.1] # z
    kern = gp.kernels.Matern(length_scale=len_scl) + gp.kernels.WhiteKernel(noise_level=0.05)

    # generate the model
    mod = gp.GaussianProcessRegressor(kernel=kern, alpha=1.e-2, optimizer='fmin_l_bfgs_b', 
            n_restarts_optimizer=100, normalize_y=True, copy_X_train=True, random_state=10)

    # fit the GP to the computed spread factors
    mod.fit(z.reshape(-1, 1), sig_factor)

    # get the new redshift error from the GP emu 
    zsig_new = mod.predict(cat[zcol].reshape(-1, 1)) * cat[sig_col]

    return zsig_new

def emu_z_mag(cat, zcol='DNF_ZMEAN_SOF', mcol='MAG_AUTO_R', sig_col='DNF_ZSIGMA_SOF',
              d='/data/des70.a/data/marcelle/GW170814/catalogs/',
              filename='PIT_corr_mag.npz'):
    
    # Get the spread factors and arrays
    z = np.load(d+filename)['z']
    m = np.load(d+filename)['m']
    sig_factor = np.load(d+filename)['sig_factor']

    # define the covariance kernel
    len_scl = [0.1, 1.5] # z, m
    kern = gp.kernels.Matern(length_scale=len_scl) + gp.kernels.WhiteKernel(noise_level=0.05)

    # generate the model
    mod = gp.GaussianProcessRegressor(kernel=kern, alpha=1.e-2, optimizer='fmin_l_bfgs_b', 
        n_restarts_optimizer=100, normalize_y=True, copy_X_train=True, random_state=10)

    # sig_factor[z,m], in this case we want m tiled and z repeated
    # fit the GP to the computed spread factors
    mod.fit(np.vstack((np.repeat(z,len(m)),np.tile(m,len(z)))).T, sig_factor.flatten())
    
    # get the new redshift error from the GP emu 
    zsig_new = mod.predict(np.vstack((cat[zcol],cat[mcol])).T) * cat[sig_col]
    
    return zsig_new





#########################################
########### Time delay utils ############
#########################################

def sfh(t, tau):  #For now exp SFH
    return np.exp(-(t/tau))/tau

def lnlike_taud(taud, z, pb_gal, distmu, diststd, distnorm, H0, age_gal, tau_gal, norm_sfh_gal):
    distgal = (c/1000.)*z/H0 #THIS NEEDS TO BECOME A DISTANCE WITH FULL COSMOLOGY!!! #cosmo = FlatLambdaCDM(H0=H0, Om0=0.3, Tcmb0=2.725)
    t_gal = age_gal-taud
    ix_formed_gals =(t_gal>0)
    like_gals = norm_sfh_gal[ix_formed_gals] * sfh(t_gal[ix_formed_gals],tau_gal[ix_formed_gals]) * pb_gal[ix_formed_gals] * distnorm[ix_formed_gals] * norm(distmu[ix_formed_gals], diststd[ix_formed_gals]).pdf(distgal[ix_formed_gals])*z[ix_formed_gals]**2
    normalization = H0**3
    print(np.log(np.sum(like_gals)/normalization))
    return np.log(np.sum(like_gals)/normalization)

def lnprior_taud(taud, age_gal, tau_gal, norm_sfh_gal , taud_min, taud_max):
    if taud_min < taud < taud_max:
        return 0.0 #  Eventually this should be either norm_sfh_gal*sfh(age_gal-taud,tau_gal) for each galaxy or the prior on cosmological parameters
    return -np.inf

def lnprob_taud(taud, z, pb_gal, distmu, diststd, distnorm, H0, age_gal, tau_gal, norm_sfh_gal, taud_min, taud_max):
    lp = lnprior_taud(taud, age_gal, tau_gal, norm_sfh_gal, taud_min, taud_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_taud(taud, z, pb_gal, distmu, diststd, distnorm, H0, age_gal, tau_gal, norm_sfh_gal)
