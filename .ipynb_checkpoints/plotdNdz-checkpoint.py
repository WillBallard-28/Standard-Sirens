import numpy as np
import sys
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
sys.stderr=open('error.log','w',buffering=1)

def plt_style():
    plt.rcParams.update({
                        'lines.linewidth':1.0,
                        'lines.linestyle':'-',
                        'lines.color':'black',
                        'font.family':'serif',
                        'font.weight':'normal',
                        'font.size':13.0,
                        'text.color':'black',
                        'text.usetex':False,
                        'axes.edgecolor':'black',
                        'axes.linewidth':1.0,
                        'axes.grid':False,
                        'axes.titlesize':'x-large',
                        'axes.labelsize':'x-large',
                        'axes.labelweight':'normal',
                        'axes.labelcolor':'black',
                        'axes.formatter.limits':[-4,4],
                        'xtick.major.size':7,
                        'xtick.minor.size':4,
                        'xtick.major.pad':8,
                        'xtick.minor.pad':8,
                        'xtick.labelsize':'medium',
                        'xtick.minor.width':1.0,
                        'xtick.major.width':1.0,
                        'ytick.major.size':7,
                        'ytick.minor.size':4,
                        'ytick.major.pad':8,
                        'ytick.minor.pad':8,
                        'ytick.labelsize':'medium',
                        'ytick.minor.width':1.0,
                        'ytick.major.width':1.0,
                        'legend.numpoints':1,
                        #'legend.fontsize':'x-large',
                        'legend.shadow':False,
                        'legend.frameon':False})

table20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

filename="/hildafs/projects/phy220048p/wballard/scanner/reducedTable.fits"
event=''

h=fits.open(filename,memmap=True)[1].data
cosmo=FlatLambdaCDM(H0=70,Om0=0.3)

mean=722.3
sigma=74.6
nsigma=1.65
zmin = z_at_value(cosmo.luminosity_distance,(mean-nsigma*sigma)*u.Mpc,zmin=1e-5,zmax=2,method='bounded')
zmax = z_at_value(cosmo.luminosity_distance,(mean+nsigma*sigma)*u.Mpc,zmin=1e-5,zmax=2,method='bounded')
bins=np.linspace(0,zmax,num=30)
zbins = np.zeros(bins.shape[0]-1)
for i in range(bins.shape[0]-1): 
    zbins[i]=(bins[i+1]+bins[i])/2.

mask=(h['ZWARN']==0) & (h['SPECTYPE']=='GALAXY')
z=h['Z']
mask_z = (z>zmin)& (z<zmax) & mask
z=z[mask_z]

mean_z,zbins_e=np.histogram(z,bins=bins)

plt.clf()
plt_style()

dV = cosmo.differential_comoving_volume(zbins).value

norm_mean=mean_z.sum()
dV_n=dV/dV.sum()*norm_mean
print(len(zbins))
print(len(mean_z))
plt.fill_between(zbins,mean_z-dV_n, np.full(len(mean_z),0.), label='Mean Z',color='aqua',alpha=0.3)

plt.xlabel('z')
plt.ylabel(r'${\rm d} N/{\rm d}z - ({\rm d}N/{\rm d}z)_{\rm com}$')
plt.plot([0,zmax],[0.,0.], 'k--')
plt.ylim(-200,200)
plt.title(event)
plt.tight_layout()
plt.savefig('GW190412.png',dpi=200)