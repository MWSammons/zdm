"""

From output of analyse_repeat_dists.py
make plots for four example cases: a,b,c,d
Does this using 30 declination bins
Prouces output:
- MChistogram
- cumMChistogram (both baed on number of repetitions)



"""
from pkg_resources import resource_filename

import numpy as np
from matplotlib import pyplot as plt
import os
import pickle

import utilities as ute

from zdm.craco import loading
from zdm import cosmology as cos
from zdm import repeat_grid as rep
from zdm import misc_functions
from zdm import survey
from zdm import beams
beams.beams_path = '/Users/cjames/CRAFT/FRB_library/Git/H0paper/papers/Repeaters/BeamData/'

import matplotlib
matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

global opdir
opdir = 'TEST/'
if not os.path.exists(opdir):
    os.mkdir(opdir)

def main(Nbin=6):
    Nsets=5
    global opdir
    dims=['$R_{\\rm min}$','$R_{\\rm max}$','$\\gamma_{r}$']
    pdims=['$p(R_{\\rm min})$','$p(R_{\\rm max})$','$p(\\gamma_{r})$']
    names=['Rmin','Rmax','Rgamma']
    string=['Ptot','Pn','Pdm']
    pstrings=['$P_{\\rm tot}$','$P_{N}$','$P_{\\rm DM}$']
    
    
    tmults = np.logspace(0.5,4,8)
    Nmults = tmults.size+1
    
    setnames = ['$\\epsilon_C: {\\rm min}\\, \\alpha$','$\\epsilon_z:$ Best fit','${\\rm min}\\, E_{\\rm max}$',
        '${\\rm max} \\,\\gamma$','${\\rm min} \\, \\sigma_{\\rm host}$']
    
    # gets the possible states for evaluation
    states,names=get_states()
    rlist=[]
    ndm = 1400
    rlist= np.zeros([Nsets,ndm])
    
    tms = np.zeros([Nsets,Nmults])
    
    NMChist = 100
    mcrs=np.linspace(1.5,0.5+NMChist,NMChist) # from 1.5 to 100.5 inclusive
    
    # set the sets!
    i=0
    
    infile='Rfitting/converge_set_'+str(i)+'__output.npz'
        
    data=np.load(infile)
    
    
    Rmins=data['arr_5']
    Rmaxes=data['arr_6']
    Rgammas=data['arr_7']
    
    xRmax = 0.3
    xRgamma = -2.8
    
    #xRmax = 17
    #xRgamma = -1.8
    irg = np.argmin((Rgammas - xRgamma)**2)
    irm = np.argmin((Rmaxes- xRmax)**2)
    Rmin = Rmins[irg,irm]
    Rmax = Rmaxes[irm]
    Rgamma = Rgammas[irg]
    
    label=str(np.log10(Rmax))[0:5]+"  "+str(Rgamma-0.001)[0:5]
    
    states[i].rep.Rmin = Rmin
    states[i].rep.Rmax = Rmax
    states[i].rep.Rgamma = Rgamma
    print("Parameters: ",Rmin,Rmax,Rgamma)
    
    dms,ss,rs,bs,nss,nrs,nbs,Mh,fitv,copyMh,Cnreps,Cnss,Cnrs = generate_state(states[i],\
        mcrs=mcrs,Rmult=1000.,tmults=None,Nbin=Nbin)
    
    
    bins = np.linspace(0,2000,21)
    ddm = dms[1]-dms[0]
    dbin = bins[1]-bins[0]
    scale = dbin/ddm
    
    nCsdms,nCrdms,nCsdecs,nCrdecs = ute.get_chime_dec_dm_data(DMhalo=50,newdata='only')
    
    Csdms,Crdms,Csdecs,Crdecs = ute.get_chime_dec_dm_data(DMhalo=50,newdata=False)
    
    # plots hist of DM for repeaters over all best-fit options
    plt.figure()
    plt.xlim(0,2000)
    plt.xlabel('${\\rm DM}_{\\rm EG}$')
    plt.ylabel('$N_{\\rm rep}({\\rm DM}_{\\rm EG}) \\, [200\\,{\\rm pc}\\,{\\rm cm}^{-3}]^{-1}$')
    
    plt.hist(Crdms,bins=bins,alpha=1.0,label='CHIME catalog 1 (17)',edgecolor='black')
    
    plt.hist(nCrdms,bins=bins,alpha=0.3,label='Golden sample (25)',edgecolor='black',\
        weights=np.full([25],17./25.))
    
    labels=['$a: [~\\,-10,-0.25,-1.3]$','$b: [-1.32,~~~0.25,~~~-3]$',\
        '$c: [-1.38,~~~~~~~~3,~~~-3]$','$d: [\\,-4.8,~~~~~~~~3,~~~-2]$']
    linestyles=['-','--','-.',':']
    plt.xlim(0,1750)
    plt.plot(dms,rs*scale,label=label,linestyle=linestyles[0],linewidth=3)
        
    plt.legend(title='$~~~~~~~~~~~~[\\log_{10} R_{\\rm min},\\log_{10} R_{\\rm max},R_{\\gamma}]$',title_fontsize=12)
    
    plt.tight_layout()
    plt.savefig(opdir+'TEST_dm.pdf')
    plt.close()
    
    ### makes cumulative DM plot ###
    sx,sy,rx,ry = ute.get_chime_rs_dm_histograms(DMhalo=50,newdata=False)
    nsx,nsy,nrx,nry = ute.get_chime_rs_dm_histograms(DMhalo=50,newdata='only')
    
    plt.figure()
    plt.plot(rx,ry,label='CHIME cat 1')
    
    plt.plot(nrx,nry,label='gold 25')
    rs = np.cumsum(rs)
    rs /= rs[-1]
    plt.plot(dms,rs,label='model')
    plt.xlabel('DM')
    plt.ylabel('p(DM)')
    plt.xlim(0,2000)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+'TEST_cum_dm.pdf')
    plt.close()
    
    ############# produce an MC rate plot ###########
    
    plt.figure()
    
    
    #mids=(mcrs[1:] + mcrs[:-1])/2.
    
    vals = np.arange(2,101)
    plt.hist(vals,bins=mcrs,weights=Cnreps,label='CHIME',alpha=0.5)
    
    rates = np.linspace(2,100,99)
    markers=['+','s','x','o']
    
    styles=['-','--','-.',':']
    plt.plot(rates,Mh,linestyle="",marker=markers[0],label=label)
    plt.plot(rates,fitv,linestyle=styles[0],color=plt.gca().lines[-1].get_color())
    
    #plt.scatter(bcs[tooLow],Mh[tooLow],marker='x',s=10.)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$N_{\\rm reps}$')
    plt.ylabel('$N_{\\rm prog}(N_{\\rm reps})$')
    plt.tight_layout()
    plt.savefig(opdir+'TEST_MChistogram.pdf')
    plt.close()
    
    #### now does cumulative plot ###
    
    plt.figure()
    
    
    #mids=(mcrs[1:] + mcrs[:-1])/2.
    
    vals = np.arange(2,101)
    Ccum = np.cumsum(Cnreps)
    #Ccum = Ccum / Ccum[-1]
    
    plt.plot(rates,Ccum,label='CHIME')
    
    rates = np.linspace(2,100,99)
    markers=['+','s','x','o']
    
    styles=['-','--','-.',':']
    cumMC = np.cumsum(Mh)
    plt.plot(rates,cumMC,linestyle=styles[0],label="case "+label)
    
    plt.legend()
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('$N_{\\rm bursts}$')
    plt.ylabel('$N_{\\rm rep}(N_{\\rm bursts})$')
    
    plt.tight_layout()
    plt.savefig(opdir+'TEST_cumMChistogram.pdf')
    plt.close()
    
    
    ######### produce decbin division plot ########
    
    bdir='Nbounds'+str(Nbin)+'/'
    bounds = np.load(bdir+'bounds.npy')
    
    plt.figure()
    
    plt.ylim(0,1)
    plt.xlabel('${\\delta}$ [deg]')
    plt.ylabel('$N_{\\rm rep}(\\delta)$')
    
    sx,sy,rx,ry = ute.get_chime_rs_dec_histograms(DMhalo=50)
    nsx,nsy,nrx,nry = ute.get_chime_rs_dec_histograms(DMhalo=50,newdata='only')
    
    #plt.plot(Crdecs,yCrdecs,label='CHIME repeaters (cat1)')
    #plt.plot(nCrdecs,ynCrdecs,label='CHIME repeaters (3 yr)')
    
    plt.plot(rx,ry,label='CHIME repeaters (cat1)')
    plt.plot(nrx,nry,label='CHIME repeaters (3 yr)')
    
    cumdec = np.cumsum(nrs)
    cumdec = cumdec/cumdec[-1]
    cumdec = np.concatenate(([0],cumdec))
    plt.plot(bounds,cumdec,linestyle=styles[0],label="case "+label)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+'TEST_cumulative_dec_set0.pdf')
    plt.close()
    
    exit()
    ##### we now produce a tmult plot ######
    
    plt.figure()
    plt.xlabel('time [Catalog 1 = 1]')
    plt.ylabel('Rate of repeater discovery')
    tmults = np.logspace(0,4,9)
    for i in np.arange(Nsets):
        plt.plot(tmults,alltrs[i,:]/tmults,label = setnames[i])
    plt.xscale('log')
    plt.legend()
    plt.ylim(0,60)
    plt.xlim(1,1e4)
    plt.tight_layout()
    plt.savefig(opdir+'all_tmult_repeaters.pdf')
    plt.close()
    
    plt.figure()
    plt.xlabel('time [Catalog 1 = 1]')
    plt.ylabel('Rate of new single bursts')
    tmults = np.logspace(0,4,9)
    for i in np.arange(Nsets):
        plt.plot(tmults,alltss[i,:]/tmults,label = setnames[i])
    plt.xscale('log')
    plt.legend()
    #plt.ylim(0,60)
    plt.xlim(1,1e4)
    plt.tight_layout()
    plt.savefig(opdir+'all_tmult_singles.pdf')
    plt.close()
    
    plt.figure()
    plt.xlabel('time [Catalog 1 = 1]')
    plt.ylabel('Rate of busts from repeaters')
    tmults = np.logspace(0,4,9)
    for i in np.arange(Nsets):
        plt.plot(tmults,alltbs[i,:]/tmults,label = setnames[i])
    plt.xscale('log')
    plt.legend()
    #plt.ylim(0,60)
    plt.xlim(1,1e4)
    plt.tight_layout()
    plt.savefig(opdir+'all_tmult_rep_bursts.pdf')
    plt.close()
 
def generate_state(state,Nbin=6,Rmult=1.,mcrs=None,tmults=None):
    """
    Defined to test some corrections to the repeating FRBs method
    """
    # old implementation
    # defines list of surveys to consider, together with Tpoint
    sdir = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/Surveys')
    
    ndm=1400
    nz=500
    
    # holds grids and surveys
    ss = []
    gs = []
    rgs = []
    Cnrs = []
    Cnss =  []
    irs = []
    
    # holds lists of dm and z distributions for predictions
    zrs=[]
    zss=[]
    dmrs=[]
    dmss=[]
    nrs=[]
    nss=[]
    nbs=[]
    
    # holds sums over dm and z
    tdmr = np.zeros([ndm])
    tdmb = np.zeros([ndm])
    tdms = np.zeros([ndm])
    tzr = np.zeros([nz])
    tzs = np.zeros([nz])
    
    # saves number of repeaters
    if tmults is not None:
        tmultlists = np.zeros([tmults.size+1])
        tmultlistr = np.zeros([tmults.size+1])
        tmultlistb = np.zeros([tmults.size+1])
    
    # total number of repeaters and singles
    CNR=0
    CNS=0
    tnr = 0
    tns = 0
    
    bdir='Nbounds'+str(Nbin)+'/'
    beams.beams_path = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/'+bdir)
    bounds = np.load(bdir+'bounds.npy')
    solids = np.load(bdir+'solids.npy')
    
    # for MC distribution
    numbers = np.array([])
    
    Cnreps=np.array([])
    
    # we initialise surveys and grids
    for ibin in np.arange(Nbin):
        # generate basic grids
        name = "CHIME_decbin_"+str(ibin)+"_of_"+str(Nbin)
        s,g = survey_and_grid(survey_name=name,NFRB=None,sdir=sdir,init_state=state)
        
        #ss.append(s)
        #gs.append(g)
        
        ir = np.where(s.frbs['NREP']>1)[0]
        cnr=len(ir)
        irs.append(ir)
        
        i_s = np.where(s.frbs['NREP']==1)[0]
        cns=len(i_s)
        
        CNR += cnr
        CNS += cns
        
        Cnrs.append(cnr)
        Cnss.append(cns)
        
        ###### repeater part ####
        rg = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,MC=False,opdir=None,bmethod=2)
        #rgs.append(rg)
        
        if ibin == 0:
            tot_exact_reps = rg.exact_reps
            tot_exact_singles = rg.exact_singles
            tot_exact_rbursts = rg.exact_rep_bursts
            tot_exact_zeroes = rg.exact_zeroes
        else:
            tot_exact_reps += rg.exact_reps
            tot_exact_singles += rg.exact_singles
            tot_exact_rbursts += rg.exact_rep_bursts
            tot_exact_zeroes += rg.exact_zeroes
        
        # collapses CHIME dm distribution for repeaters and once-off burts
        zr = np.sum(rg.exact_reps,axis=1)
        zs = np.sum(rg.exact_singles,axis=1)
        dmrb = np.sum(rg.exact_rep_bursts,axis=0)
        dmr = np.sum(rg.exact_reps,axis=0)
        dms = np.sum(rg.exact_singles,axis=0)
        nb = np.sum(dmrb)
        nr = np.sum(dmr)
        ns = np.sum(dms)
        
        
        
        if tmults is not None:
            tmultlists[0] += ns
            tmultlistr[0] += nr
            tmultlistb[0] += nb
            for it,tmult in enumerate(tmults):
                print("For bin ",it," doing tmult ",tmult)
                thisrg = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=tmult,MC=False,opdir=None,bmethod=2)
                thisns = np.sum(thisrg.exact_singles)
                thisnr = np.sum(thisrg.exact_reps)
                thisnb = np.sum(thisrg.exact_rep_bursts)
                tmultlists[it+1] += thisns
                tmultlistr[it+1] += thisnr
                tmultlistb[it+1] += thisnb
        
        # adds to running totals
        tzr += zr
        tzs += zs
        tdmb += dmrb
        tdmr += dmr
        tdms += dms
        tnr += nr
        tns += ns
        
        zrs.append(zr)
        zss.append(zs)
        dmrs.append(dmr)
        dmss.append(dms)
        nbs.append(nb)
        nss.append(ns)
        nrs.append(nr)
        rgs.append(rg)
        
        # extract single and repeat DMs, create total list
        nreps=s.frbs['NREP']
        ireps=np.where(nreps>1)
        isingle=np.where(nreps==1)[0]
        if ibin==0:
            alldmr=s.DMEGs[ireps]
            alldms=s.DMEGs[isingle]
            print(type(alldms))
        else:
            alldmr = np.concatenate((alldmr,s.DMEGs[ireps]))
            alldms = np.concatenate((alldms,s.DMEGs[isingle]))
        
        dmvals=g.dmvals
        zvals=g.zvals
        ddm = g.dmvals[1]-g.dmvals[0]
        
        if Rmult is None:
            del s
            del g
            del rg
            continue
        del rg
        ################  MC ##########
        rg = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,opdir=None,bmethod=2,\
                Exact=False,MC=Rmult)
        numbers = np.append(numbers,rg.MC_numbers)
        
        ir = np.where(s.frbs['NREP']>1)[0]
        nr=len(ir)
        irs.append(ir)
        nreps = s.frbs['NREP'][ir]
        Cnreps = np.concatenate((Cnreps,nreps))
        # there is still a memory leak, what a pain
        del s
        del g
        del rg
    
    Chist,bins = np.histogram(Cnreps,bins=mcrs)
    
    
    ############ standard analysis ############3
     
    print("Calculated ",CNS,CNR)
    rnorm = CNS/tns # this is NOT the plotting norm, it is the physical norm
    #ddm = g.dmvals[1]-g.dmvals[0]
    norm = rnorm*200 / ddm
    
    
    nbs = np.array(nbs)
    nrs = np.array(nrs)
    nss = np.array(nss)
    nbs *= rnorm
    nrs *= rnorm
    nss *= rnorm
    
    which = ["Repeaters","Singles","Bursts","Zeroes"]
    tot_exact_reps *= rnorm
    tot_exact_singles *= rnorm
    tot_exact_rbursts *= rnorm
    
    # performs exact plotting
    for ia,array in enumerate([tot_exact_reps,tot_exact_singles,tot_exact_rbursts,tot_exact_zeroes]):
        name = opdir+'TEST_'+which[ia]+".pdf"
        misc_functions.plot_grid_2(array,zvals,dmvals,
            name=name,norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
            project=False,Aconts=None,zmax=3,
            DMmax=3000)
        #norm=3, log=True
    bins=np.linspace(0,3000,16)
    
    tdms *= rnorm
    tdmr *= rnorm
    tdmb *= rnorm
    
    
    ########## MC analysis #############
    nM = len(numbers)
    nC = len(Cnreps)
    Mh,b=np.histogram(numbers,bins=mcrs)
    bcs = b[:-1]+(b[1]-b[0])/2.
    # find the max set below which there are no MC zeros
    firstzero = np.where(Mh==0)[0]
    if len(firstzero) > 0:
        firstzero = firstzero[0]
        f = np.polyfit(np.log10(bcs[:firstzero]),np.log10(Mh[:firstzero]),1,\
            w=1./np.log10(bcs[:firstzero]**0.5))
        #    error = np.log10(bcs[:firstzero]**0.5))
    else:
        f = np.polyfit(np.log10(bcs),np.log10(Mh),1,\
            w=1./np.log10(bcs**0.5))
    
    fitv = 10**np.polyval(f,np.log10(bcs)) *nC/nM
    
    # normalise h to number of CHIME repeaters
    # Note: for overflow of histogram, the actual sum will be reduced
    Mh = Mh*nC/nM
    Nmissed = nC - np.sum(Mh) # expectation value for missed fraction
    
    # Above certain value, replace with polyval expectation
    tooLow = np.where(Mh < 10)[0]
    copyMh = np.copy(Mh)
    copyMh[tooLow] = fitv[tooLow]
    
    # values to return
    # Mh: MC histogram
    # fitv: fitted values
    # copyMh: Mh with over-written too-low values
    
    
    return dmvals,tdms,tdmr,tdmb,nss,nrs,nbs,Mh,fitv,copyMh,Chist,Cnss,Cnrs #,tmultlists,tmultlistr,tmultlistb # returns singles and total bursts
    
def set_state(pset,chime_response=True):
    """
    Sets the state parameters
    """
    
    state = loading.set_state(alpha_method=1)
    state_dict = dict(cosmo=dict(fix_Omega_b_h2=True))
    state.energy.luminosity_function = 2 # this is Schechter
    state.update_param_dict(state_dict)
    # changes the beam method to be the "exact" one, otherwise sets up for FRBs
    state.beam.Bmethod=3
    
    
    # updates to most recent best-fit values
    state.cosmo.H0 = 67.4
    
    if chime_response:
        state.width.Wmethod=0 #only a single width bin
        state.width.Wbias="CHIME"
    
    state.energy.lEmax = pset['lEmax']
    state.energy.gamma = pset['gamma']
    state.energy.alpha = pset['alpha']
    state.FRBdemo.sfr_n = pset['sfr_n']
    state.host.lsigma = pset['lsigma']
    state.host.lmean = pset['lmean']
    state.FRBdemo.lC = pset['lC']
    
    return state


def shin_fit():
    """
    Returns best-fit parameters from Shin et al.
    https://arxiv.org/pdf/2207.14316.pdf
    
    """
    
    pset={}
    pset["lEmax"] = np.log10(2.38)+41.
    pset["alpha"] = -1.39
    pset["gamma"] = -1.3
    pset["sfr_n"] = 0.96
    pset["lmean"] = 1.93
    pset["lsigma"] = 0.41
    pset["lC"] = np.log10(7.3)+4.
    
    return pset

def james_fit():
    """
    Returns best-fit parameters from James et al 2022 (Hubble paper)
    """
    
    pset={}
    pset["lEmax"] = 41.63
    pset["alpha"] = -1.03
    pset["gamma"] = -0.948
    pset["sfr_n"] = 1.15
    pset["lmean"] = 2.22
    pset["lsigma"] = 0.57
    pset["lC"] = 1.963
    
    return pset



def read_extremes(infile='planck_extremes.dat',H0=67.4):
    """
    reads in extremes of parameters from a get_extremes_from_cube
    """
    f = open(infile)
    
    sets=[]
    
    for pset in np.arange(6):
        # reads the 'getting' line
        line=f.readline()
        
        pdict={}
        # gets parameter values
        for i in np.arange(7):
            line=f.readline()
            words=line.split()
            param=words[0]
            val=float(words[1])
            pdict[param]=val
        pdict["H0"]=H0
        pdict["alpha"] = -pdict["alpha"] # alpha is reversed!
        sets.append(pdict)
        
        pdict={}
        # gets parameter values
        for i in np.arange(7):
            line=f.readline()
            words=line.split()
            param=words[0]
            val=float(words[1])
            pdict[param]=val
        pdict["H0"]=H0
        pdict["alpha"] = -pdict["alpha"] # alpha is reversed!
        sets.append(pdict)
    return sets


def get_states():  
    """
    Gets the states corresponding to plausible fits to single CHIME data
    """
    psets=read_extremes()
    psets.insert(0,shin_fit())
    psets.insert(1,james_fit())
    
    
    # gets list of psets compatible (ish) with CHIME
    chime_psets=[4]
    chime_names = ["CHIME min $\\alpha$"]
    
    # list of psets compatible (ish) with zdm
    zdm_psets = [1,2,7,12]
    zdm_names = ["zDM best fit","zDM min $\\E_{\\rm max}$","zDM max $\\gamma$","zDM min $\sigma_{\\rm host}$"]
    
    names=[]
    # loop over chime-compatible state
    for i,ipset in enumerate(chime_psets):
        
        state=set_state(psets[ipset],chime_response=True)
        if i==0:
            states=[state]
        else:
            states.append(states)
        names.append(chime_names[i])
    
    for i,ipset in enumerate(zdm_psets):
        state=set_state(psets[ipset],chime_response=False)
        states.append(state)
        names.append(zdm_names[i])
    
    return states,names       


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


main()
