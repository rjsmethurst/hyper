""" A fantastic python code to determine the quenched SFH hyper-parameters of a population using emcee (http://dan.iel.fm/emcee/current/). This file contains all the functions needed to determine the mean SFH parameters from posterior functions of single galaxies.
    """

import numpy as N
import pylab as P
import time

def intprior(s):
    t, tau = s[0], s[1]
    ttr = N.logical_and(t >=0.003, t <= 13.807108309208775)
    tautr = N.logical_and(tau >=0.003, tau <= 5.0)
    return ttr+tautr

def intpostone(onea, s):
    """ Interim posterior function for samples given alpha values for one morphology.
        :onea:
        half of the alpha hyper parameters describing one morphology
        :s:
        Set of samples from Ni galaxies - Nj samples drawn from each galaxy, shape of s is (2, Ni, Nj) - one array for t and tau
        RETURNS:
        Probability P(tj, tauj|alpha) which is of shape (Ni, Nj)
        """
    ut, st, utau, stau = onea
    t, tau = s[0], s[1]
    return (1/(st * N.sqrt(2 * N.pi)) * N.exp( - (t - ut)**2 / (2 * st**2) ))*(1/(stau * N.sqrt(2 * N.pi)) * N.exp( - (tau - utau)**2 / (2 * stau**2) ))


def intpost(alpha, s, pd, ps):
    """ Likelihood function for hyper parameters alpha.
        :alpha:
        Hyper parameters fit to a function.
        uts, sts, utaus, staus, utd, std, utaud, staud
        :s:
        Set of samples from Ni galaxies - Nj samples drawn from each galaxy - shape of s is (2, Ni, Nj) - one array for t and tau
        :pd:ps:
        Disc and smooth vote fractions to be used a weights in the likelihood. Shape of both is (Ni, )
        
        RETURNS:
        P(tj, tauj|alpha) which is of shape (Ni, ) having summed over all samples j
        """
    sm = intpostone(N.split(alpha, 2)[0], s)
    dc = intpostone(N.split(alpha, 2)[1], s)
    return N.sum((pd*dc + ps*sm)/intprior(s), axis=1)

def lnlike(alpha, s, pd, ps):
    """ Likelihood function for hyper parameters alpha.
        :alpha:
        Hyper parameters fit to a function.
        uts, sts, utaus, staus, utd, std, utaud, staud
        :s:
        Set of samples from Ni galaxies - Nj samples drawn from each galaxy - shape of s is (2, Ni, Nj) - one array for t and tau
        :pd:ps:
        Disc and smooth vote fractions to be used a weights in the likelihood. Shape of both is (Ni, )
        RETURNS:
        One value of lnlike for given alphas for all the samples having summed over all galaxies in population i.
        """
    wj = intpost(alpha, s, pd, ps)
    return s.shape[0]*N.log(s.shape[1]) + N.sum(N.log(wj), axis=0)

def lnprior(alpha):
    """
        Prior probabilty on the hyper parameters alpha.
        :alpha:
        uts, sts, utaus, staus, utd, std, utaud, staud
        """
    uts, sts, utaus, staus, utd, std, utaud, staud = alpha
    if 0.003 <= uts <= 13.807108309208775 and 0.003 <= utaus <= 5.0 and 0.003 <= utd < 13.807108309208775 and 0.003 <= utaud <= 5.0 and 0<sts<2.0 and 0<staus<2.0 and 0<std<2.0 and 0<staud<2.0:
        return 0.0
    else:
        return -N.inf

def lnprob(alpha, s, pd, ps):
    """
        Posterior function for alpha and samples drawn from each galaxy with vote fractions pd and ps.
        :alpha:
        Hyper parameters fit to a function.
        uts, sts, utaus, staus, utd, std, utaud, staud
        :s:
        Set of samples from Ni galaxies - Nj samples drawn from each galaxy - shape of s is (2, Ni, Nj) - one array for t and tau
        :pd:ps:
        Disc and smooth vote fractions to be used a weights in the likelihood. Shape of both is (Ni, )
        
        """
    lp = lnprior(alpha)
    if not N.isfinite(lp):
        return -N.inf
    return lp + lnlike(alpha, s, pd, ps)

def walker_plot(samples, nwalkers, limit):
    """ Plotting function to visualise the steps of the walkers in each parameter dimension for smooth and disc theta values.
        
        :samples:
        Array of shape (nsteps*nwalkers, 4) produced by the emcee EnsembleSampler in the sample function.
        
        :nwalkers:
        The number of walkers that step around the parameter space used to produce the samples by the sample function. Must be an even integer number larger than ndim.
        
        :limit:
        Integer value less than nsteps to plot the walker steps to.
        
        RETURNS:
        :fig:
        The figure object
        """
    s = samples.reshape(nwalkers, -1, 8)
    s = s[:,:limit, :]
    fig = P.figure(figsize=(8,20))
    ax1 = P.subplot(8,1,1)
    ax2 = P.subplot(8,1,2)
    ax3 = P.subplot(8,1,3)
    ax4 = P.subplot(8,1,4)
    ax5 = P.subplot(8,1,5)
    ax6 = P.subplot(8,1,6)
    ax7 = P.subplot(8,1,7)
    ax8 = P.subplot(8,1,8)
    for n in range(len(s)):
        ax1.plot(s[n,:,0], 'k')
        ax2.plot(s[n,:,1], 'k')
        ax3.plot(s[n,:,2], 'k')
        ax4.plot(s[n,:,3], 'k')
        ax5.plot(s[n,:,4], 'k')
        ax6.plot(s[n,:,5], 'k')
        ax7.plot(s[n,:,6], 'k')
        ax8.plot(s[n,:,7], 'k')
    ax1.tick_params(axis='x', labelbottom='off')
    ax2.tick_params(axis='x', labelbottom='off')
    ax3.tick_params(axis='x', labelbottom='off')
    ax4.tick_params(axis='x', labelbottom='off')
    ax5.tick_params(axis='x', labelbottom='off')
    ax6.tick_params(axis='x', labelbottom='off')
    ax7.tick_params(axis='x', labelbottom='off')
    ax4.set_xlabel(r'step number')
    ax1.set_ylabel(r'$\mu_t^{s}$')
    ax2.set_ylabel(r'$\sigma_t^{s}$')
    ax3.set_ylabel(r'$\mu_{\tau}^{s}$')
    ax4.set_ylabel(r'$\sigma_{\tau}^{s}$')
    ax5.set_ylabel(r'$\mu_t^{d}$')
    ax6.set_ylabel(r'$\sigma_t^{d}$')
    ax7.set_ylabel(r'$\mu_{\tau}^{d}$')
    ax8.set_ylabel(r'$\sigma_{\tau}^{d}$')
    P.subplots_adjust(hspace=0.1)
    save_fig = 'walkers_steps_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.pdf'
    fig.savefig(save_fig)
    return fig


