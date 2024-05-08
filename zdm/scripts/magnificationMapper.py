import scipy.signal
import scipy.interpolate
import scipy.integrate
import matplotlib.pyplot as plt 
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import LambdaCDM
from astropy import units as u
from astropy import constants as const
import matplotlib.pyplot as plt
import numpy as np
#import dynspec
#import pygedm
import astropy.coordinates as c
from astropy.io import fits
from astropy import wcs
import astropy
from astropy.convolution import Gaussian2DKernel

def offSetBeamGains(bPos, imageCoords, beamSigma):
    xOffset = imageCoords[0] - bPos[0]
    yOffset = imageCoords[1] - bPos[1]
    radiusOffset = (xOffset**2+yOffset**2)**0.5
    gains = BFG(radiusOffset, beamSigma)
    return gains

def BFG(x, sigma):
    return np.exp(-1/2*(x**2/(sigma**2)))


def logSpaceIntegrand(logmu, func, funcArgs, base):
    """Useful for evaluating integrals of func in log space"""
    return func(base**logmu, funcArgs)*base**logmu*np.log(base)

def unnormalisedLensFuncAtSubBeam(log10b, dlog10b, OmegaB, bGains, pixRes, magni, muThresh):
    #OmegaB in arcminutes^2, same as pixRes
    print(log10b, dlog10b)
    inBeam = np.abs(np.log10(bGains)-log10b)<np.abs(dlog10b/2)
    if np.sum(inBeam)>0:
        gtrMu = np.zeros(len(muThresh))
        for i in range(len(muThresh)):
            gtrMu[i] = np.sum(magni[inBeam]>=muThresh[i])
    
        modelledArea = np.sum(np.abs(np.log10(bGains)-log10b)<np.abs(dlog10b/2))*(pixRes[0]*pixRes[1])
        numUnmodelledCells = (OmegaB - modelledArea)/(pixRes[0]*pixRes[1])
        print('num in beam', np.sum(inBeam))
        extra1SWhere = muThresh<1
        gtrMu[extra1SWhere] = gtrMu[extra1SWhere]+numUnmodelledCells
        probUN = (-1*np.diff((gtrMu))/np.diff(muThresh)/muThresh[:-1])
        # smoothingKernel = scipy.signal.windows.gaussian(len(muThresh[:-1]),0.05/(np.mean(np.diff(np.log10(muThresh)))))
        # probUNSmooth= np.convolve(probUN,smoothingKernel, mode='same')/np.sum(smoothingKernel)
        interpFunc = scipy.interpolate.interp1d(np.log10(muThresh[:-1]), probUN, bounds_error=False, fill_value=0)
    else: 
        interpFunc = None
    return interpFunc


def mapWidener(magni, wideningFrac):
    smoothKernel = Gaussian2DKernel(int(magni.shape[0]/100))
    smoothMagni = scipy.signal.fftconvolve(np.log10(magni), smoothKernel, mode='valid')
    croppedEachSide = np.floor((np.asarray(magni.shape) - np.asarray(smoothMagni.shape))/2)
    interpFunc = scipy.interpolate.RegularGridInterpolator((np.arange(int(croppedEachSide[0]), (magni.shape[0] - int(croppedEachSide[0])),1), np.arange(int(croppedEachSide[1]), (magni.shape[1] - int(croppedEachSide[1])),1)), (smoothMagni), bounds_error=False, fill_value=None)
    wA = int(magni.shape[0]*wideningFrac/2)
    x = np.meshgrid(np.arange(-wA,magni.shape[0]+wA,1), np.arange(-wA,magni.shape[1]+wA,1))
    expandedMagni = interpFunc((x[0].flatten(), x[1].flatten()))
    finalMagni = (expandedMagni.reshape(np.asarray(magni.shape)+wA*2))
    finalMagni[finalMagni<0] = 0
    finalMagni[:wA,:wA] = np.mean(finalMagni[wA:-wA,:wA])
    finalMagni[-wA:, :wA] = np.mean(finalMagni[-wA:, wA:-wA])
    finalMagni[-wA:,-wA:] = np.mean(finalMagni[wA:-wA, -wA:])
    finalMagni[:wA, -wA:] = np.mean(finalMagni[:wA, wA:-wA])
    completeMagni = finalMagni
    completeMagni[wA:-wA,wA:-wA] = np.log10(magni)
    return 10**completeMagni, x


def normalisedLensFuncsAcrossBeam(D, freq, thresh, nbins, bPos, proj, magni, name, muThresh = 10**(np.arange(-3,6,0.02)+0.05)):
    FWHM = 1.22*(const.c/(freq))/D
    beamSigma=(FWHM/2.)*(2*np.log(2))**-0.5
    dlnb=-np.log(thresh)/nbins
    log10min=np.log10(thresh)
    dlog10b=log10min/nbins
    log10b=(np.arange(nbins)+0.5)*dlog10b
    OmegaB= (2*np.pi*dlnb*(beamSigma*180/np.pi*60)**2).decompose().value
    pixRes = np.abs(np.diag(proj.pixel_scale_matrix*60))
    dataEdge = np.mean(np.concatenate((magni[:,0], magni[:,-1], magni[0,:], magni[-1,:])))
    count = 0
    xOrig = np.meshgrid(np.arange(magni.shape[0]), np.arange(magni.shape[1]))
    x = np.meshgrid(np.arange(magni.shape[0]), np.arange(magni.shape[1]))
    tempMagni = magni.copy()
    while (dataEdge -1) > 0.1:                                        # couldnt find anything so do this downsampling so just do it yourself
        # here we want to form a 2D interpolation of the magni map based on a downsampled version, then extrapolate using that model
        # to higher separations and infill the higher res map with extrapolated values. 
        tempMagni,x = mapWidener(magni, 0.5+0.2*count)
        dataEdge = np.mean(np.concatenate((tempMagni[:,0], tempMagni[:,-1], tempMagni[0,:], tempMagni[-1,:])))
        count = count+1
        print('trapped forever', count)
    magni = tempMagni
    imageCoords = proj.array_index_to_world_values(x[0], x[1])
    bGains = offSetBeamGains(bPos, imageCoords, beamSigma.decompose().value*180/np.pi)
    fig = plt.figure()
    ax = plt.subplot(111, projection=proj)
    sizeDiff = int((len(xOrig[0][:,0])-len(x[0][:,0]))/2)
    tower = np.zeros(bGains.shape)
    for i in range(len(log10b)):
        gainLevel = np.abs(np.log10(bGains)-log10b[i])<np.abs(dlog10b/2)
        tower[gainLevel] = i
    plt.imshow(tower, extent=[-sizeDiff,len(xOrig[0][:,0])+sizeDiff,-sizeDiff,len(xOrig[0][0,:])+sizeDiff], cmap='tab10', vmin=0, vmax=(len(log10b)-1))
    ax.imshow(np.log10(magni).T, aspect='auto', extent=[-sizeDiff,len(xOrig[0][:,0])+sizeDiff,-sizeDiff,len(xOrig[0][0,:])+sizeDiff], alpha=0.7)
    ax.imshow(bGains, alpha=0.5,extent=[-sizeDiff,len(xOrig[0][:,0])+sizeDiff,-sizeDiff,len(xOrig[0][0,:])+sizeDiff], cmap='Greys')
        
    plt.xlabel(r'RA')
    plt.ylabel(r'Dec')
    overlay = ax.get_coords_overlay('icrs')
    overlay.grid(color='white', ls='dotted')
    fig.savefig(str(name))
    plt.close()
    pmus = np.zeros([len(muThresh),len(log10b)])
    probMags = np.log10(muThresh[:-1])
    for i in range(len(log10b)):
        print('beaming like crazy right now', i)
        interpFunc = unnormalisedLensFuncAtSubBeam(log10b[i], dlog10b, OmegaB, bGains, pixRes, magni, muThresh)
        if interpFunc != None:
            muTwo = np.arange(-2,10,0.01)
            pmus[:,i] = ((1/(np.nansum(10**muTwo*interpFunc(muTwo))*np.diff(muTwo)[0]*np.log(10))*interpFunc(np.log10(muThresh))))
        else: 
            pmus[:,i] = np.nan
    return muThresh, pmus





