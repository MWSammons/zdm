""" 
This script creates zdm grids and plots localised FRBs

It can also generate a summed histogram from all CRAFT data

"""
import os

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm.craco import loading
from zdm import io
from magnificationMapper import normalisedLensFuncsAcrossBeam
from astropy.io import fits
from astropy import wcs
import astropy
from astropy import units as u
from astropy import constants as const
from multiprocessing import Pool

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

#        state = parameters.State()
#        state.energy.lEmax = 41.63
#        state.energy.gamma = -0.948
#        state.energy.alpha = -1.03
#        state.FRBdemo.sfr_n = 1.15
#        state.host.lsigma = 0.57
#        state.host.lmean = 2.22
#        state.FRBdemo.lC = 1.443

def beamPositionInstance(i):
    # in case you wish to switch to another output directory
    #opdir = "Localised_FRBs/"
    opdir = "CHORD/"
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)

    # Initialise surveys and grids

    # The below is for private, unpublished FRBs. You will NOT see this in the repository!
    sdir = "../data/Surveys/"
    
    # specifies state, updates variables according to H0
    # best fit, but with Emax extended as per Ryder et al
    state = parameters.State()
    state.energy.lEmax = 41.38
    state.energy.gamma = -1.3
    state.energy.alpha = -1.39
    state.FRBdemo.sfr_n = 1.0
    state.host.lsigma = 0.57
    state.host.lmean = 2.22
    state.FRBdemo.lC = 4.86
    state.energy.luminosity_function=4

    magni = fits.getdata('hlsp_frontier_model_macs0717_bradac_v1_z01-magnif.fits')
    info = fits.getheader('hlsp_frontier_model_macs0717_bradac_v1_z01-magnif.fits')
    proj = wcs.WCS(info)

    xMagni = np.meshgrid(np.arange(0,len(magni[:,0]),1), np.arange(0,len(magni[0,:]),1))
    tempCoords = proj.array_index_to_world_values(xMagni[0], xMagni[1])
    relBeamPositions = np.load('relBeamPos.npy')
    ratesArr=np.zeros(len(relBeamPositions[:,0]))
    print('---Beam Pos:', i)
    surveyName = 'CHORD_BeamPos_'+str(i)
    mux, pmux = normalisedLensFuncsAcrossBeam(48*u.m, 900*u.MHz, 1e-3, 10, np.array([np.mean(tempCoords[0])+relBeamPositions[i,0], np.mean(tempCoords[1])+relBeamPositions[i,1]]), proj, magni, 'CHORD/'+surveyName)
    np.save('mux_BP_'+str(i), np.log10(mux))
    np.save('pmux_BP_'+str(i), pmux)
    np.save(str(surveyName)+'mus', np.log10(mux))
    np.save(str(surveyName)+'pmus', pmux)
    fig, ax = plt.subplots()
    ax.plot(np.log10(mux), np.log10(pmux))
    ax.set_xlabel('$\log_{10}\\mu$')
    ax.set_ylabel('$\log_{10}p(\\mu)$')
    fig.savefig('beamMagDist/probs'+format(i, '02d'))
    plt.close()
    del(pmux)
    del(mux)
    
    s,g = loading.survey_and_grid(survey_name=surveyName,
        NFRB=None,sdir=sdir,init_state=state)
    
    np.save('ratesUnlensed_BP_'+str(i), g.rates)
    
    
    FRB_rate_per_day = np.sum(g.rates) * 10**g.state.FRBdemo.lC
    print("Rate of FRBs per day is ",FRB_rate_per_day)
    FRB_rate_per_day = np.sum(g.rates[g.zvals>1.0,:]) * 10**g.state.FRBdemo.lC
    print("Rate of FRBs per day with z > 1.0 is ",FRB_rate_per_day)
    
    ratesArr[i] = FRB_rate_per_day
    np.save('CHORD/'+surveyName+'RatesArr', ratesArr)

    misc_functions.plot_grid_2(
            g.rates,
            g.zvals,
            g.dmvals,
            name=opdir + surveyName + ".pdf",
            norm=3,
            log=True,
            label="$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]",
            project=False,
            zmax=4,
            Aconts=[0.01, 0.1, 0.5],
            DMmax=4000
        ) #


if __name__ == "__main__":
    relBeamPositions = np.load('relBeamPos.npy')
    p = Pool(10)
    mapArgs = []
    for i in range(len(relBeamPositions[:,0])):
        mapArgs.append(i)
    
    p.map(beamPositionInstance, mapArgs)

