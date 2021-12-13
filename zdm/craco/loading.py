""" CRACO FRBs: This may move out of tests someday """

######
# first run this to generate surveys and parameter sets, by 
# setting NewSurveys=True NewGrids=True
# Then set these to False and run with command line arguments
# to generate *many* outputs
#####

# It should be possible to remove all the matplotlib calls from this
# but in the current implementation it is not removed.
import numpy as np
import os
from pkg_resources import resource_filename

from astropy.cosmology import Planck15, Planck18

from zdm import survey
from zdm import parameters
from zdm import cosmology as cos
from zdm import misc_functions

from IPython import embed

def set_state(alpha_method=1, cosmo=Planck18):

    ############## Initialise parameters ##############
    state = parameters.State()

    # Variable parameters
    vparams = {}
    
    vparams['FRBdemo'] = {}
    vparams['FRBdemo']['alpha_method'] = alpha_method
    vparams['FRBdemo']['source_evolution'] = 0
    
    vparams['beam'] = {}
    vparams['beam']['thresh'] = 0
    vparams['beam']['method'] = 2
    
    vparams['width'] = {}
    vparams['width']['logmean'] = 1.70267
    vparams['width']['logsigma'] = 0.899148
    vparams['width']['Wbins'] = 10
    vparams['width']['Wscale'] = 2
    
     # constants of intrinsic width distribution
    vparams['MW']={}
    vparams['MW']['DMhalo']=50
    
    vparams['host']={}
    vparams['energy'] = {}
    
    if vparams['FRBdemo']['alpha_method'] == 0:
        vparams['energy']['lEmin'] = 30
        vparams['energy']['lEmax'] = 41.7
        vparams['energy']['alpha'] = 1.55
        vparams['energy']['gamma'] = -1.09
        vparams['FRBdemo']['sfr_n'] = 1.67
        vparams['FRBdemo']['lC'] = 3.15
        vparams['host']['lmean'] = 2.11
        vparams['host']['lsigma'] = 0.53
    elif  vparams['FRBdemo']['alpha_method'] == 1:
        vparams['energy']['lEmin'] = 30
        vparams['energy']['lEmax'] = 41.4
        vparams['energy']['alpha'] = 0.65
        vparams['energy']['gamma'] = -1.01
        vparams['FRBdemo']['sfr_n'] = 0.73
        # NOTE: I have not checked what the best-fit value
        # of lC is for alpha method=1
        vparams['FRBdemo']['lC'] = 1 #not best fit, OK for a once-off
        
        vparams['host']['lmean'] = 2.18
        vparams['host']['lsigma'] = 0.48
        
    state.update_param_dict(vparams)
    state.set_astropy_cosmo(cosmo)

    # Return
    return state


def survey_and_grid(survey_name:str='CRAFT/CRACO_1_5000',
            state_dict=None,
               alpha_method=1, NFRB:int=100, lum_func:int=0):
    """ Load up a survey and grid for a CRACO mock dataset

    Args:
        cosmo (str, optional): astropy cosmology. Defaults to 'Planck15'.
        survey_name (str, optional):  Defaults to 'CRAFT/CRACO_1_5000'.
        NFRB (int, optional): Number of FRBs to analyze. Defaults to 100.
        lum_func (int, optional): Flag for the luminosity function. 
            0=power-law, 1=gamma.  Defaults to 0.
        state_dict (dict, optional):
            Used to init state instead of alpha_method, lum_func parameters

    Raises:
        IOError: [description]

    Returns:
        tuple: Survey, Grid objects
    """
    # Init state
    state = set_state(alpha_method=alpha_method)

    # Addiitonal updates
    if state_dict is None:
        state_dict = dict(cosmo=dict(fix_Omega_b_h2=True))
        state.energy.luminosity_function = lum_func
    state.update_param_dict(state_dict)
    
    # Cosmology
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    # get the grid of p(DM|z)
    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic',
        datdir=resource_filename('zdm', 'GridData'))
    
    ############## Initialise surveys ##############
    sdir = os.path.join(resource_filename('zdm', 'craco'), 'MC_Surveys')
    isurvey = survey.load_survey(survey_name, state, dmvals,
                                 NFRB=NFRB, sdir=sdir, Nbeams=5)

    # generates zdm grid
    grids = misc_functions.initialise_grids(
        [isurvey], zDMgrid, zvals, dmvals, state, wdist=True)
    print("Initialised grid")

    # Return Survey and Grid
    return isurvey, grids[0]