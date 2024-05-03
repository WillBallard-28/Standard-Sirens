# custom functions used frequently


import numpy as np
from astropy.coordinates import Distance
from astropy.cosmology import Planck18
from astropy import units as u
import astropy.constants as asc
from astropy.table import Table, hstack, vstack
from astropy import table


import psycopg2


omega_sky = 41253 #

# Calculating luminosities in different bands
# However, since we only look at the z-band we need to correct the solar luminosity
# Calculation from here: https://astronomy.stackexchange.com/questions/25126/how-to-calculate-luminosity-in-g-band-from-absolute-ab-magnitude-and-luminosity
lmbda_z = 920*10**(-9) #in  m
del_lambda_z = 160*10**(-9) #in m
del_v_z = (asc.c*lmbda_z/(del_lambda_z**2)).value # this is in /s now

lmbda_r = 640*10**(-9) #in  m 5600A-7200A -> 6400A median
del_lambda_r = 160*10**(-9) #in m
del_v_r = (asc.c*lmbda_r/(del_lambda_r**2)).value # this is in /s now


m_sun_z = -27.56 # in z-band: http://mips.as.arizona.edu/~cnaw/sun.html (DES filter)
m_sun_r = -27.12 # in r-band: http://mips.as.arizona.edu/~cnaw/sun.html (DES filter)

f_v_z = 10**((-48.6-m_sun_z)/2.5) # in erg/(cm^2 s Hz)
f_v_r = 10**((-48.6-m_sun_r)/2.5) # in erg/(cm^2 s Hz)

dist_sun = 1.496*10**13 # in cm

L_sun_z = f_v_z*del_v_z*4*np.pi*dist_sun**2 *10**(-7) # this is  in Watts now
L_sun_r = f_v_r*del_v_r*4*np.pi*dist_sun**2 *10**(-7)
L_sun_bol = 3.828*10**26

L_sun_bands = {"r": L_sun_r*u.W, "z": L_sun_z*u.W, "bol": L_sun_bol*u.W}

std_names = ['TARGETID', 'TARGET_RA', 'TARGET_DEC', 'LASTNIGHT', 'Z', 'ZERR', 'ZWARN', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'SPECTYPE', 'BGS_TARGET', 'EBV', 'SERSIC']

ext_coeffs =  [3.995, 3.214, 2.165, 1.592, 1.211, 1.064] # for the DECam 𝑢, 𝑔, 𝑟, 𝑖, 𝑧, 𝑌 

"""
The first two functions following here are actually copied from https://github.com/desihub/desiutil/blob/a15f4f98214d14ad05e740d68f333994dfa1f1f8/py/desiutil/dust.py#L22
"""

def extinction_total_to_selective_ratio(band, photsys, match_legacy_surveys=False):
    """Return the linear coefficient R_X = A(X)/E(B-V) where
    A(X) = -2.5*log10(transmission in X band),
    for band X in 'G','R' or 'Z' when
    photsys = 'N' or 'S' specifies the survey (BASS+MZLS or DECALS),
    or for band X in 'G', 'BP', 'RP' when photsys = 'G' (when gaia dr2)
    or for band X in 'W1', 'W2', 'W3', 'W4' when photsys is either 'N' or 'S'
    E(B-V) is interpreted as SFD.
    Args:
        band : 'G', 'R', 'Z', 'BP', 'RP', 'W1', 'W2', 'W3', or 'W4'
        photsys : 'N' or 'S'
    Returns:
        scalar, total extinction A(band) = -2.5*log10(transmission(band))
    """
    if match_legacy_surveys:
        # Based on the fit from the columns MW_TRANSMISSION_X and EBV
        # for the DR8 target catalogs and propagated in fibermaps
        # R_X = -2.5*log10(MW_TRANSMISSION_X) / EBV
        # It is the same value for the N and S surveys in DR8 and DR9 catalogs.
        R = {"G_N": 3.2140,
             "R_N": 2.1650,
             "Z_N": 1.2110,
             "G_S": 3.2140,
             "R_S": 2.1650,
             "Z_S": 1.2110,
             "G_G": 2.512,
             "BP_G": 3.143,
             "RP_G": 1.663}
    else:
        # From https://desi.lbl.gov/trac/wiki/ImagingStandardBandpass
        # DECam u  3881.6   3.994
        # DECam g  4830.8   3.212
        # DECam r  6409.0   2.164
        # DECam i  7787.5   1.591
        # DECam z  9142.7   1.211
        # DECam Y  9854.5   1.063
        # BASS g  4772.1   3.258
        # BASS r  6383.6   2.176
        # MzLS z  9185.1   1.199
        # Consistent with the synthetic magnitudes and function dust_transmission

        R = {"G_N": 3.258,
             "R_N": 2.176,
             "Z_N": 1.199,
             "G_S": 3.212,
             "R_S": 2.164,
             "Z_S": 1.211,
             "G_G": 2.197,
             "BP_G": 2.844,
             "RP_G": 1.622}

    # Add WISE from
    # https://github.com/dstndstn/tractor/blob/main/tractor/sfd.py#L23-L35
    R.update({'W1_N': 0.184,
              'W2_N': 0.113,
              'W3_N': 0.0241,
              'W4_N': 0.00910,
              'W1_S': 0.184,
              'W2_S': 0.113,
              'W3_S': 0.0241,
              'W4_S': 0.00910})

    assert band.upper() in ["G", "R", "Z", "BP", "RP", 'W1', 'W2', 'W3', 'W4']
    assert photsys.upper() in ["N", "S", "G"]
    return R["{}_{}".format(band.upper(), photsys.upper())]


def mwdust_transmission(ebv, band, photsys, match_legacy_surveys=False):
    """Convert SFD E(B-V) value to dust transmission 0-1 for band and photsys
    Args:
        ebv (float or array-like): SFD E(B-V) value(s)
        band (str): 'G', 'R', 'Z', 'W1', 'W2', 'W3', or 'W4'
        photsys (str or array of str): 'N' or 'S' imaging surveys photo system
    Returns:
        scalar or array (same as ebv input), Milky Way dust transmission 0-1
    If `photsys` is an array, `ebv` must also be array of same length.
    However, `ebv` can be an array with a str `photsys`.
    Also see `dust_transmission` which returns transmission vs input wavelength
    """
    if isinstance(photsys, str):
        r_band = extinction_total_to_selective_ratio(band, photsys, match_legacy_surveys=match_legacy_surveys)
        a_band = r_band * ebv
        transmission = 10**(-a_band / 2.5)
        return transmission
    else:
        photsys = np.asarray(photsys)
        if np.isscalar(ebv):
            raise ValueError('array photsys requires array ebv')
        if len(ebv) != len(photsys):
            raise ValueError('len(ebv) {} != len(photsys) {}'.format(
                len(ebv), len(photsys)))

        transmission = np.zeros(len(ebv))
        for p in np.unique(photsys):
            ii = (photsys == p)
            r_band = extinction_total_to_selective_ratio(band, p, match_legacy_surveys=match_legacy_surveys)
            a_band = r_band * ebv[ii]
            transmission[ii] = 10**(-a_band / 2.5)

        return transmission

def mw_transmission_from_data_table(data, band):
    
    """
    Automatically calculate the MW_TRANSMISSION values from normal data table I use.
    
    Parameters:
    -----------
        data: astropy.table
            the normal astropy table I use (needs TARGET_DEC and EBV values
        band: string ("g", "r", "z")
            the photometric band to use
    
    Returns:
    --------
        MW_TRANSMISSION: np.array
            returns a numpy array that contains the milky way transmission values"""
    
    photsys = np.array(["N" for q in range(len(data))])
    ii = np.where(data["TARGET_DEC"] < 0)
    photsys[ii] = "S"
    
    return mwdust_transmission(data["EBV"], band, photsys, match_legacy_surveys=False)

def connect_to_time_db():
    """
    connect to the desi daily SQL database
    
    Returns
    -------
        cursor: cursor object
            cursor for the database
    """
    try:
        db = psycopg2.connect(host='decatdb.lbl.gov', database='desidb', user='desi', password = "5kFibers!", port="5432")
        cursor = db.cursor()
        return cursor
    except (Exception, psycopg2.Error) as error:
        print("FAILED TO ESTABLISH CONNECTION TO DATABASE")
        print("------------------------------------------")
        print("Error log: ")
        print(error)
        
def make_table(rows, names = std_names):
    """
    Creates an astropy data table with only good redshifts and only unique entries (takes the first)
    
    Parameters
    ----------
        rows: row object from cursor.fetchall() for example
            contains the actual data from the sql database 
        names: list
            list of names for the columns (e.g. ["TARGET_RA", "TARGET_DEC"])
            
    Returns
    -------
        table: astropy table
            returns an astropy table with the specified column names and only good redshifts; also ensures only unique entries exist and can return a table with no data, only column names
    """
    if rows:
        data = Table(list(map(list, zip(*rows))),
                             names=names)
        
        if len(data) > 0:
            data = data[data['ZWARN']==0]
            data = data[data['Z']>=0]
            data = table.unique(data, keys = "TARGETID")
        return data
    else:
        return Table(names = names)

def make_list(rows, names = std_names):
    """
    Creates a numpy list from the sql rows
    
    Parameters
    ----------
        rows: row object from cursor.fetchall() for example
            contains the actual data from the sql database
            
    Returns
    -------
        data: python list
            returns a python list without any data clean up
    """
    
    if rows:
        data = list(map(list, zip(*rows)))
        return data
    

def get_data_rad_search(ra, dec, radius, cursor, query=None):
    
    """
    Queries data from the database; a standard query is implemented, but can be altered
    Parameters
    ----------
        ra: int/float
            right ascension of the center for the circular search
        dec: int/float
            declination of the center of the circular search
        radius: int/float
            search radius
        cursors: db.cursor()
            cursors object from the database
        query (optional): str
            the query string (i.e. what data to retrieve from the database
    
    Returns
    -------
        rows: row object from database
            contains the actual information. use make_table() to convert to astropy table
            
    """
    redux = "daily"
    if query == None:
        query = 'SELECT f.targetid, f.target_ra, f.target_dec, c.night, r.z, r.zerr, r.zwarn, f.flux_g,  f.flux_r, f.flux_z, r.spectype, f.bgs_target, f.ebv, f.sersic\n' \
                    f'FROM {redux}.tiles_fibermap f\n' \
                    f'INNER JOIN {redux}.cumulative_tiles c ON f.cumultile_id=c.id\n' \
                    f'INNER JOIN {redux}.tiles_redshifts r ON r.cumultile_id=c.id AND r.targetid=f.targetid\n' \
                    f'WHERE q3c_radial_query( f.target_ra, f.target_dec, {ra}, {dec}, {radius});'
    cursor.execute(query)
    rows = cursor.fetchall()
    return rows
       
def db_doall(ra, dec, radius, names = std_names, query=None):
    
    """
    Do all steps to retrieve data from the database. A standard query is provided, but can be changed
    
    Parameters
    ----------
        ra: int/float
            right ascension of the center for the circular search
        dec: int/float
            declination of the center of the circular search
        radius: int/float
            search radius
        names (optional): list
            list of names for the columns (e.g. ["TARGET_RA", "TARGET_DEC"])
        query (optional): str
            the query string (i.e. what data to retrieve from the database
    Returns
    -------
        table: astropy table
            returns the data table with only good redshift objects and unique entries
    """
    
    cursor = connect_to_time_db()
    rows = get_data_rad_search(ra, dec, radius, cursor, query)
    table = make_table(rows, names)
    return table

def app_mag(flux, mw_transmission=1):
    """
    Calculate the apparent magnitude from the flux as defined by https://www.legacysurvey.org/dr10/description/, including a mw transmission correction, if provided
    Input:
    - flux as numpy array in nanomaggies
    """
    
    return 22.5-2.5*np.log10(flux/mw_transmission)
    
    
    

def abs_mag(m, z, band, color_name, color):
    """
    Calculate the absolute magnitude, for now only uses m and z, no K-correction yet
    Input:
    - m: apparent magnitude
    - z: redshift 
    - band: band name, like "g"
    - color name: name of the color, e.g. "g - r"
    - color: value of the color
    """
    #return m - 5*np.log10(Distance(z=z, cosmology=Planck18)/u.Mpc*10**6)+5 - calc_kcor(band, z, color_name, color)
    return m - 5*np.log10(Distance(z=z, cosmology=Planck18)/u.Mpc*10**6)+5

def abs_mag_ldist(m, l_dist):
    
    return m-5*np.log10(l_dist)+5

def M(L, band = "bol"):
    
    """
    Calculates the absolute magnitude from the luminosity
    One can specify a band or just use the bolometric value without specifiyng a band
    Input:
    - L: luminosity in Watts (u.W)
    - band: "g", "r", "z" for example or specify nothing for bolometric
    """
    
    L_sun = L_sun_bands[band]
    
    return -2.5*np.log10(L/(L_sun))


def lum(M, band = "bol"):
    """
    Get the luminosity in watts
    Input:
    - absolute magnitude M in this band
    """
    L_sun = L_sun_bands[band]
    return L_sun*10**(-0.4*M)

def lum_wiggle(D_gal, m):
    """
    Calculate a luminositiy proxy from the magnitude
    PArams:
    - D_gal: Distance to the galaxy, measured from redshift
    - m: magnitude (for us in z-band)
    
    returns: D_gal**2*10**(-0.4*m)
    """
    return D_gal**2*10**(-0.4*m)


def V_max(omega_s, z_min, z_max_data, z_max_lum):
    """
    Description: Calculates the maximum Volume in which the source could have been detected in: corrects for the so-called "Malmquist-bias"
    (faint objects, which usually also means low mass objects, will only be covered in a survey within a smaller volume than bright and high-mass
    objects)
    
    Params:
    omega_s: surface area covered by the complete data
    z_min: lower redshift limit of sample
    z_max_lum: maximum redshift determined for for object based on luminosity and its completeness 
    (i.e. maximum redshift at which the source with luminosity l would no longer be part of the sample)
    
    """
    z_max = np.minimum(z_max_data, z_max_lum)
    return 4/3*np.pi * omega_s/omega_sky * (Planck18.comoving_distance(z_max)**3-Planck18.comoving_distance(z_min)**3)

def V_max_dist(omega_s, dist_min, dist_max_data, dist_max):
    """
    Description: Calculates the maximum Volume in which the source could have been detected in: corrects for the so-called "Malmquist-bias"
    (faint objects, which usually also means low mass objects, will only be covered in a survey within a smaller volume than bright and high-mass
    objects)
    
    Params:
    omega_s: surface area covered by the complete data
    z_min: lower redshift limit of sample
    z_max_lum: maximum redshift determined for for object based on luminosity and its completeness 
    (i.e. maximum redshift at which the source with luminosity l would no longer be part of the sample)
    
    """
    dist_max = np.minimum(dist_max_data, dist_max)
    return 4/3*np.pi * omega_s/omega_sky * (dist_max**3-dist_min**3)


def lum_lim(lum, m_lim, m):
    """
    Calculate the luminosity a source would have if its magnitude was equal to the DESI magnitude limit (have to research the value)
    
    Params:
    - lum: luminosity of an object
    - m_lim: magnitude limit of the survey
    - m: apparent magnitude of an object
    
    """
    return lum*10**(0.4*(m-m_lim))



def calc_kcor(filter_name, redshift, colour_name, colour_value):
    """
    K-corrections calculator in Python. See http://kcor.sai.msu.ru for the 
    reference. Available filter-colour combinations must be present in the 
    `coeff` dictionary keys.

    @type   filter_name: string    
    @param  filter_name: Name of the filter to calculate K-correction for, e.g. 
                         'u', 'g', 'r' for some of the SDSS filters, or 'J2', 
                         'H2', 'Ks2' for 2MASS filters (must be present in 
                         `coeff` dictionary)
    @type      redshift: float    
    @param     redshift: Redshift of a galaxy, should be between 0.0 and 0.5 (no
                         check is made, however)
    @type   colour_name: string    
    @param  colour_name: Human name of the colour, e.g. 'u - g', 'g - r', 
                         'V - Rc', 'J2 - Ks2' (must be present in `coeff` dictionary)
    @type  colour_value: float    
    @param colour_value: Value of the galaxy's colour, specified in colour_name    
    @rtype:              float
    @return:             K-correction in specified filter for given redshift and 
                         colour
    @version:            2012
    @author:             Chilingarian, I., Melchior. A.-L., and Zolotukhin, I.
    @license:            Simplified BSD license, see http://kcor.sai.msu.ru/license.txt

    Usage example:
    
        >>> calc_kcor('g', 0.2, 'g - r', 1.1)
        0.5209713975999992
        >>> calc_kcor('Ic', 0.4, 'V - Ic', 2.0)
        0.310069919999993
        >>> calc_kcor('H', 0.5, 'H - K', 0.1)
        -0.14983142499999502
        
    """
    coeff = {

        'r_gr': [
            [0,0,0,0],
            [1.83285,-2.71446,4.97336,-3.66864],
            [-19.7595,10.5033,18.8196,6.07785],
            [33.6059,-120.713,-49.299,0],
            [144.371,216.453,0,0],
            [-295.39,0,0,0],
        ], 
        'g_gz': [
            [0,0,0,0],
            [2.37454,-4.39943,7.29383,-2.90691],
            [-28.7217,-20.7783,18.3055,5.04468],
            [220.097,-81.883,-55.8349,0],
            [-290.86,253.677,0,0],
            [-73.5316,0,0,0],
        ],
        'z_gz': [
            [0,0,0,0],
            [0.30146,-0.623614,1.40008,-0.534053],
            [-10.9584,-4.515,2.17456,0.913877],
            [66.0541,4.18323,-8.42098,0],
            [-169.494,14.5628,0,0],
            [144.021,0,0,0],
        ]

    }

    c = coeff[filter_name + '_' + colour_name.replace(' - ', '')]
    kcor = 0.0

    for x, a in enumerate(c):
        for y, b in enumerate(c[x]):
            kcor += c[x][y] * redshift**x * colour_value**y

    return kcor