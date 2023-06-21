""" 
This script generates simulated instances of MC FRBs in z DM space for
a single point in Rparameter space, and evaluates the likelihood over
the rest of that space.

It will first generate a file called "mc.npy", which contains
the *indices* of repeaters in zDM space. The file structure is:
- list over Nbin declination bins
- list over NMC MC calculations (default: 100)
- either empty (no FRBs generated) OR list of 3 indices: iz,iDM,NFRB
- Each entry says "NFRB repeating FRBs were generated at iz,iDM"

The second step is to evaluate the likelihood over the entire space.
It first loops over the Rgamma,Rmax grid, loading in Rmin values from
the input file generated by "fit_repeating_distributions.py".
In each case, it calculates the likelihoods for repeating FRBs only,
by first normalising the repeating zDM grid to unity, and then
storing the sum of the log-likelihoods for each set of FRB
observations. These get summed over all bins, but separate values
are stored for each Nmc calculation. The outputs are kept in
the "savefile".

The third step is to plot the result. It loops over all NMC iterations,
and for each, calculates a Baysian confidence intervals at 1,2,3 sigma
limits. For each grid point, it then records whether or not it lies within
that interval. Finally, it plots contours for all points that lie 50%
of the time in/our of that interval.


"""
import os
from pkg_resources import resource_filename
from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm.craco import loading
from zdm import io
from zdm import repeat_grid as rep

import utilities as ute
import states as st

import pickle
import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

import scipy as sp
from scipy.stats import poisson

import states

import matplotlib
import time
from zdm import beams


beams.beams_path = '/Users/cjames/CRAFT/FRB_library/Git/H0paper/papers/Repeaters/BeamData/'

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

global indir


def main(Nbin=6,FC=1.0):
    global indir
    indir = 'Rfitting39_'+str(FC)+'/'
    infile = indir+'mc_FC39'+str(FC)+'converge_set_0_output.npz'
    savefile = indir+'MCevaluation_set_0_all_lls.npy'
    if not os.path.exists(savefile):
        evaluate_MC(infile,savefile,Nbin=Nbin,wRgamma=-2.2,wRmax=30,Nreps=100)
    
    plot_results(infile,savefile)

def plot_results(infile,MCfile):
    """
    Plots results of evaluating the MC
    
    """
    
    
    # loads relevant arrays from the infile
    Rmins,Rmaxes,Rgammas = load_data(infile)
    
    # for each combination of Rmax, Rgamma, we have a list of
    # lls which is as long as the number of generations
    MC = np.load(MCfile)
    Ng,Nx,Nrep = MC.shape
    
    levels = [0.68,0.95,0.997]
    Nlevel = len(levels)
    count = np.zeros([Nlevel,Ng,Nx])
    
    for i in np.arange(Nrep):
        array = ute.make_bayes(MC[:,:,i])
        # determines which bits are inside various contours
        
        ls=ute.get_contour_level(array,levels) # l is highest value outside contour
        
        #print("Array is ",array)
        #exit()
        for j,l in enumerate(ls):
            OK1,OK2 = np.where(array > l)
            count[j,OK1,OK2] += 1
    
    labels = ['$\\gamma_r$','$R_{\\rm max}$']
    
    for i,l in enumerate(levels):
        op = 'Posteriors/set_0_MC_evaluation_'+str(l*100)+'.pdf'
        conts = [[[count[0],50],[count[1],50],[count[2],50]]]
        plot_2darr(count[i],labels,op,[Rgammas,np.log10(Rmaxes)],
            [Rgammas,Rmaxes], clabel = '$\\%~{\\rm in}~'+str(i+1)+'\\sigma$ C.I.',
            conts=conts)
        # instead, get contours for each and *then* plot them!


def evaluate_MC(infile,savefile,Nbin=6,wRgamma=-2.2,wRmax=30,Nreps=100):
    """ 
    evaluates the MC set, sves the data in the save file
    
    """ 
    # gets the possible states for evaluation
    statelist,names = st.get_states()
    
    sdir = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/Surveys')
    state = statelist[0]
    
    Rmins,Rmaxes,Rgammas = load_data(infile)
    
    # values for simulated truth - returns closest values, and their indices
    
    tRmax, iRmax = select_which(Rmaxes,wRmax)
    tRgamma, iRg = select_which(Rgammas,wRgamma)
    tRmin = Rmins[iRg,iRmax]
    
    # gets survey and grid data - fast, to be re-used
    ss,gs,nss,nrs = initialise_grids(state,Nbin)
    
    # generates MC z,DM values for FRBs. Does this for each declination bin
    # result has dimensions of:
    # bin, trial, list of z,DM, Nfrb triplets (in case we get two FRBs in the same bin)
    global indir
    MCfile = indir+'mc.npy'
    
    
    
    if os.path.exists(MCfile):
        zDMlists = np.load(MCfile,allow_pickle=True)
    else:
        norm = normalise_constant(ss,gs,tRmin,tRmax,tRgamma,Nreps=Nreps,Nbin=Nbin)
        # need to reduce constant by this factor!!!!
        for g in gs:
            print(g.state.FRBdemo.lC)
            g.state.FRBdemo.lC = g.state.FRBdemo.lC -np.log10(norm)
            print(g.state.FRBdemo.lC)
        # need to reduce constant by this factor!!!!
        zDMlists = gen_mc_set(ss,gs,tRmin,tRmax,tRgamma,Nreps=Nreps,Nbin=Nbin)
        np.save(MCfile,zDMlists)
    t0=time.time()
    all_lls = np.zeros([Rgammas.size,Rmaxes.size,Nreps])
    for i,Rgamma in enumerate(Rgammas):
        for j,Rmax in enumerate(Rmaxes):
            Rmin=Rmins[i,j]
            print("Evaluating MC set",Rmin,Rmax,Rgamma," time is ",time.time()-t0)
            llslist = evaluate_mc_set(ss,gs,Rmin,Rmax,Rgamma,zDMlists,Nbin=Nbin)
            llslist = np.array(llslist)
            all_lls[i,j,:] = llslist
    
    np.save(savefile,all_lls)
    
def select_which(array,value):
    """
    Reuturns nearest value and its index
    """
    
    ir = np.argmin((array-value)**2)
    closest = array[ir]
    return closest,ir

def load_data(infile):
    """
    Loads saved file, extracts relevant arrays
    """
    data = np.load(infile)
    lps=data['arr_0']
    lns=data['arr_1']
    ldms=data['arr_2']
    lpNs=data['arr_3']
    Nrs=data['arr_4']
    Rmins=data['arr_5']
    Rmaxes=data['arr_6']
    Rgammas=data['arr_7']
    lskps=data['arr_8']
    lrkps=data['arr_9']
    ltdm_kss=data['arr_10']
    ltdm_krs=data['arr_11']
    lprod_bin_dm_krs=data['arr_12']
    lprod_bin_dm_kss=data['arr_13']
    Rstar = data['arr_14']
    mcrs = data['arr_15']
    MChs = data['arr_16']
    MCrank = data['arr_17']
    
    # we only need these values for this routine
    return Rmins,Rmaxes,Rgammas
    
    

def initialise_grids(state,Nbin=6):
    """
    Initialises basic zdm grids in absence of repeaters
    """
    ############## loads CHIME surveys and grids #############
    
    # old implementation
    # defines list of surveys to consider, together with Tpoint
    sdir = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/Surveys')
    
    bdir='Nbounds'+str(Nbin)+'/'
    beams.beams_path = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/'+bdir)
    bounds = np.load(bdir+'bounds.npy')
    solids = np.load(bdir+'solids.npy')
    
    ss=[]
    gs=[]
    nrs=[]
    nss=[]
    irs=[]
    iss=[]
    NR=0
    NS=0
    # we initialise surveys and grids
    Cnreps = np.array([])
    for ibin in np.arange(Nbin):
        
        name = "CHIME_decbin_"+str(ibin)+"_of_"+str(Nbin)
        
        s,g = survey_and_grid(survey_name=name,NFRB=None,sdir=sdir,init_state=state)
        
        ss.append(s)
        gs.append(g)
        
        ir = np.where(s.frbs['NREP']>1)[0]
        nr=len(ir)
        irs.append(ir)
        nreps = s.frbs['NREP'][ir]
        Cnreps = np.concatenate((Cnreps,nreps))
        i_s = np.where(s.frbs['NREP']==1)[0]
        ns=len(i_s)
        iss.append(i_s)
        
        NR += nr
        NS += ns
        
        nrs.append(nr)
        nss.append(ns)
    
    return ss,gs,nrs,nss


def evaluate_mc_set(ss,gs,Rmin,Rmax,Rgamma,zDMlists,FC=1.,verbose=False,Nbin=6,Rmult=1.):
    """
    Constructs repeat grid for above parameters, and evaluates likelihoods
    of zDMlists accordingly
    """
    
    alllls = []
    for ibin in np.arange(Nbin):
        t0=time.time()
        # calculates repetition info
        g=gs[ibin]
        s=ss[ibin]
        g.state.rep.Rmin = Rmin
        g.state.rep.Rmax = Rmax
        g.state.rep.Rgamma = Rgamma
        
        rg = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,opdir=None,bmethod=2,\
                Exact=True,MC=None)
        norm = np.sum(rg.exact_reps)
        llnorm = np.log10(norm)
        minval = 1e-8 # hard-coded, typical minimum calculable value
        
        # first loops over nsims in this declination bin
        for iset,frbset in enumerate(zDMlists[ibin]):
            if ibin==0:
                alllls.append(0.)
            if len(frbset) == 0: # no FRBs generated in this iteration
                continue
            
            ls = rg.exact_reps[frbset[0],frbset[1]] * frbset[2]
            zeros = np.where(ls ==0)[0]
            if len(zeros)>0:
                ls[zeros] = minval
            lls = np.log10(ls) - llnorm
            # replace literal zeroes with minimum
            # this is because there should be no non-zero values possible
            # except due to numerical limits
            alllls[iset] += np.sum(lls)
        
    return alllls

def normalise_constant(ss,gs,Rmin,Rmax,Rgamma,Nreps=100,FC=1.,verbose=False,Nbin=6,Rmult=1.):
    """
    
    Calcs the normalising constant required to reproduce the number of single CHIME FRBs.
    This then modifies the source density for future MC calculations
    """
    
    
    numbers = np.array([])
    NS=0
    for ibin in np.arange(Nbin):
        
        # calculates repetition info
        g=gs[ibin]
        s=ss[ibin]
        g.state.rep.Rmin = Rmin
        g.state.rep.Rmax = Rmax
        g.state.rep.Rgamma = Rgamma
        
        rg = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,opdir=None,bmethod=2,\
                Exact=True,MC=False)
        
        # gets single rates
        ns = np.sum(rg.exact_singles)
        NS += ns
    # hard-coded number: 474
    norm = NS/ute.NSingles
    print("This parameter set generates ",NS," single FRBs")
    print("We thus renormalise to the observed 474 by a factor of ",NS/474)
    
    return norm

def gen_mc_set(ss,gs,Rmin,Rmax,Rgamma,Nreps=100,verbose=False,Nbin=6,Rmult=1.):
    """
    
    FC is fraction of CHIME singles explained by repeaters. <= 1.0.
    Don't set this to less than 1., we can't reproduce the repeaters we have!
    
    mcrs are the rates for histogram binning of results
    Chist is a histogram of CHIME repetition rates
    Cnreps are the actual repetition rates of CHIME bursts
    """
    
    # create empty array of lists
    zDMlists = []
    for i in np.arange(Nbin):
        zDMlists.append([])
    
    numbers = np.array([])
    t0=time.time()
    for ibin in np.arange(Nbin):
        
        # calculates repetition info
        g=gs[ibin]
        s=ss[ibin]
        g.state.rep.Rmin = Rmin
        g.state.rep.Rmax = Rmax
        g.state.rep.Rgamma = Rgamma
        
        rg = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,opdir=None,bmethod=2,\
                Exact=False,MC=Rmult)
        
        ireps = np.where(rg.MC_reps > 0)
        if len(ireps[0])==0:
            zDMlists[ibin].append([])
        else:
            izs = ireps[0]
            idms = ireps[1]
            nFRB = rg.MC_reps[izs,idms]
            zDMlists[ibin].append([izs,idms,nFRB])
        
        # in theory this loop should be able to go faster, since
        # it's just a bunch of indices. If the FRBs were not
        # related, we could create one giant vector of indices.
        # Potentially quicker to do this, and also create
        # a list of indices over which to sum representing
        # individual observations.
        for j in (np.arange(Nreps-1)+1):
            # this regenerates repeater MC
            rg.calc_Rthresh(Exact=False, MC=Rmult)
            # we have the following available
            ireps = np.where(rg.MC_reps > 0)
            if len(ireps[0])==0:
                zDMlists[ibin].append([])
                continue
            izs = ireps[0]
            idms = ireps[1]
            
            nFRB = rg.MC_reps[izs,idms]
            zDMlists[ibin].append([izs,idms,nFRB])
    t1=time.time()
    print("MC generation took ",t1-t0,"seconds")
    return zDMlists


def int_cdf(x,cdf):
    """
    Returns values of cdf for x
    Relies on x being an integer
    0.001 ensures we are immune to minor negative fluctuations
    """
    ix = (x+0.001-2) # because a repeater with value 2 is in the 0th bin
    ix = ix.astype(int)
    vals = cdf[ix]
    return vals

def survey_and_grid(survey_name:str='CRAFT/CRACO_1_5000',
            init_state=None,
            state_dict=None, iFRB:int=0,
               alpha_method=1, NFRB:int=100, 
               lum_func:int=2,sdir=None):
    """ Load up a survey and grid for a CRACO mock dataset

    Args:
        init_state (State, optional):
            Initial state
        survey_name (str, optional):  Defaults to 'CRAFT/CRACO_1_5000'.
        NFRB (int, optional): Number of FRBs to analyze. Defaults to 100.
        iFRB (int, optional): Starting index for the FRBs.  Defaults to 0
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
    if init_state is None:
        state = loading.set_state(alpha_method=alpha_method)
        # Addiitonal updates
        if state_dict is None:
            state_dict = dict(cosmo=dict(fix_Omega_b_h2=True))
            state.energy.luminosity_function = lum_func
        state.update_param_dict(state_dict)
    else:
        state = init_state
    
    # Cosmology
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    # get the grid of p(DM|z)
    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic',
        datdir=resource_filename('zdm', 'GridData'),
        zlog=False,nz=500)

    ############## Initialise surveys ##############
    if sdir is not None:
        print("Searching for survey in directory ",sdir)
    else:
        sdir = os.path.join(resource_filename('zdm', 'craco'), 'MC_Surveys')
    
    
    isurvey = survey.load_survey(survey_name, state, dmvals,
                                 NFRB=NFRB, sdir=sdir, Nbeams=5,
                                 iFRB=iFRB)
    
    # generates zdm grid
    grids = misc_functions.initialise_grids(
        [isurvey], zDMgrid, zvals, dmvals, state, wdist=True)
    print("Initialised grid")

    # Return Survey and Grid
    return isurvey, grids[0]


def plot_2darr(arr,labels,savename,ranges,rlabels,clabel=None,crange=None,\
    conts=None,Nconts=None,RMlim=None,scatter=None,Allowed=False):
    """
    does 2D plot
    
    array is the 2D array to plot
    labels are the x and y axis labels [ylabel,xlabel]
    Here, savename is the output file
    Ranges are the [xvals,yvals]
    Rlabels are [xtics,ytics]
    
    """
    ratio=np.abs((ranges[0][1]-ranges[0][0])/(ranges[0][2]-ranges[0][1]))
    if ratio > 1.01 or ratio < 0.99:
        log0=True
    else:
        log0=False
    
    ratio=np.abs((ranges[1][1]-ranges[1][0])/(ranges[1][2]-ranges[1][1]))
    if ratio > 1.01 or ratio < 0.99:
        log1=True
    else:
        log1=False
    
    dr1 = ranges[1][1]-ranges[1][0]
    dr0 = ranges[0][1]-ranges[0][0]
    
    aspect = (ranges[0].size/ranges[1].size)
    
    extent = [ranges[1][0]-dr1/2., ranges[1][-1]+dr1/2.,\
            ranges[0][0]-dr0/2.,ranges[0][-1]+dr0/2.]
    
    im = plt.imshow(arr,origin='lower',aspect=aspect,extent=extent)
    ax=plt.gca()
    
    # sets x and y ticks to bin centres
    ticks = rlabels[1].astype('str')
    for i,tic in enumerate(ticks):
        ticks[i]=tic[:5]
    ax.set_xticks(ranges[1][1::2])
    ax.set_xticklabels(ticks[1::2])
    plt.xticks(rotation = 90) 
    
    ticks = rlabels[0].astype('str')
    for i,tic in enumerate(ticks):
        ticks[i]=str(rlabels[0][i])[0:4]
    ax.set_yticks(ranges[0][::4])
    ax.set_yticklabels(ticks[::4])
    
    plt.xlabel(labels[1])
    plt.ylabel(labels[0])
    
    #cax = fig.add_axes([ax.get_position().x1+0.03,ax.get_position().y0,0.02,ax.get_position().height])
    #cbar = plt.colorbar(im, cax=cax) # Similar to fig.colorbar(im, cax = cax)
    cbar = plt.colorbar(shrink=0.55)
    if clabel is not None:
        cbar.set_label(clabel)
    if crange is not None:
        if len(crange) == 2:
            plt.clim(crange[0],crange[1])
        else:
            themax=np.nanmax(arr)
            plt.clim(crange+themax,themax)
    
    
    if conts is not None:
        if len(conts) == 2:
            
            ax = plt.gca()
            cs=ax.contour(conts[0],levels=[conts[1]],origin='lower',colors="black",\
                linestyles=[':'],linewidths=[3],extent=extent)
        else:
            colors=["red","white","black","orange"]
            styles=[':','-.','--','-']
            for k,cont in enumerate(conts[0]):
                print("Doing multiple conts")
                cs=ax.contour(cont[0],levels=[cont[1]],origin='lower',colors=colors[k],\
                    linestyles=styles[k],linewidths=[3],extent=extent)
    if Nconts is not None:
        ax = plt.gca()
        cs=ax.contour(Nconts[0],levels=[Nconts[1]],origin='lower',colors="orange",\
            linestyles=['-.'],linewidths=[3],extent=extent)
    
    if Allowed:
        plt.text(1,-2.5,'Allowed')
    
    if RMlim is not None:
        plt.plot([RMlim,RMlim],[extent[2],extent[3]],linestyle='--',color='white',linewidth=3)
    
    if scatter is not None:
        sx=scatter[0]
        sy=scatter[1]
        sm=scatter[2]
        for i, m in enumerate(sm):
            #ax.plot((i+1)*[i,i+1],marker=m,lw=0)
            plt.plot(sx[i],sy[i],marker=m,color='red',linestyle="")
    
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()


    
main()
