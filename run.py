from posterior import *
from astropy.cosmology import FlatLambdaCDM
import wget

import numpy as N
import pylab as PY
import pyfits as F
import os
import time

import matplotlib.cm as cm


font = {'family':'serif', 'size':16}
P.rc('font', **font)
P.rc('xtick', labelsize='medium')
P.rc('ytick', labelsize='medium')
P.rc('axes', labelsize='x-large')

#reason = str(raw_input('Why are you running this iteration? : '))
#
#Using PyFits to open up the Galaxy Zoo data
file = '/Users/becky//Projects/Green-Valley-Project/data/GZ2_all_GALEX_match_GZ1_k_correct_green_valley.fits'
dat = F.open(file)
gz2data = dat[1].data
dat.close()

col = N.zeros(16*len(gz2data)).reshape(len(gz2data), 16)
col[:,0] = gz2data.field('MU_MR')
col[:,1] = gz2data.field('NUV_U')
col[:,2] = gz2data.field('t01_smooth_or_features_a01_smooth_debiased')
col[:,3] = gz2data.field('t01_smooth_or_features_a02_features_or_disk_debiased')
col[:,4] = ((gz2data.field('Err_MU_MR'))**2 + 0.05**2)**0.5
col[:,5] = ((gz2data.field('Err_NUV_U'))**2 + 0.1**2)**0.5
col[:,6] = gz2data.field('z_1')
col[:,7] = gz2data.field('zErr_1')
col[:,8] = gz2data.field('GV_first')
col[:,9] = gz2data.field('GV_sec')
col[:,10] = gz2data.field('upper_GV')
col[:,11] = gz2data.field('lower_GV')
col[:,12] = gz2data.field('dr7objid')
col[:,13] = gz2data.field('dr8objid')
col[:,14] = gz2data.field('ra_1')
col[:,15] = gz2data.field('dec_1')

non_nan = N.logical_not(N.isnan(col[:,1])).astype(int)
colours = N.compress(non_nan, col, axis=0)
colours = colours[colours[:,0] < 6.0]
colours = colours[colours[:,1] > -3.0]
gvf = colours[colours[:,8]==1]
gv = gvf[gvf[:,9]==1]

red_s = colours[colours[:,0] > colours[:,10]]
blue_c = colours[colours[:,0] < colours[:,11]]
gv_s = gv[gv[:,2] >= 0.8]
gv_d = gv[gv[:,3] >= 0.8]
gv_clean = N.append(gv_s, gv_d, axis=0)

hund = blue_c[:10,:]

N.save('bc_'+str(len(hund))+'.npy', hund)


cosmo = FlatLambdaCDM(H0 = 71.0, Om0 = 0.26)
age = N.array(cosmo.age(hund[:,6]))

w = [7.5, 1.5, 7.0, 1.5, 4.0, 1.5, 4.0, 1.5]
nwalkers = 100
nsteps= 400
start = [7.5, 1.5]
burnin =400

X = N.linspace(0, 14, 100)
Y = N.linspace(0, 5, 100)

sums=N.zeros((len(X)-1, len(Y)-1))
sumd=N.zeros((len(X)-1, len(Y)-1))

# w is the prior conditions on my theta parameters - the mean and standard deviation of tq and tau for the disc and smooth populations
# w = [mu_tqs, mu_taus, mu_tqd, mu_taud, sig_tqs, sig_taus, sig_tqd, sig_taud]
start_time = time.time()
#The rest calls the emcee code and makes plots....
for n in range(len(hund)):
    if N.isnan(hund[n, 1]) == False:
        url = 'http://casjobs.sdss.org/ImgCutoutDR7/getjpeg.aspx?ra='+str(hund[n,14])+'&dec='+str(hund[n,15])+'&scale=0.099183&width=424&height=424'
        #f = wget.download(url, out=str(int(hund[n,13]))+'.jpeg')
        s, ss = sample(2, nwalkers, nsteps, burnin, start, w, hund[n,0], hund[n,4], hund[n, 1], hund[n, 5], age[n], hund[n,3], hund[n,2], hund[n,13])
        tq_mcmc, tau_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(s, [16,50,84],axis=0)))
        print 'tq_smooth',tq_mcmc
        print 'tau_smooth',tau_mcmc
        #fig_samp = walker_plot(s, nwalkers, nsteps, hund[n,-1])
        #fig = corner_plot(s, labels = [r'$ t_{q}$', r'$ \tau$'], extents=[[N.min(s[:,0]), N.max(s[:,0])],[N.min(s[:,1]),N.max(s[:,1])]], bf=[tq_mcmc, tau_mcmc], dr7=hund[n,13])
        #fig.savefig('triangle_t_tau_red_s_'+str(int(hund[n,13]))+'_'+str(len(s))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.pdf')
        Hs, Xs, Ys = N.histogram2d(s[:,0].flatten(), s[:,1].flatten(), bins=(X, Y), weights=N.ones(len(s))*hund[n,2])
        sums += Hs/N.sum(Hs)
        Hd, Xd, Yd = N.histogram2d(s[:,0].flatten(), s[:,1].flatten(), bins=(X, Y), weights=N.ones(len(s))*hund[n,3])
        sumd += Hd/N.sum(Hd)


elap = (time.time() - start_time)/60
print 'Minutes taken for '+str(len(s)/nwalkers
                               )+' steps and '+str(nwalkers)+' walkers', elap


print N.shape(Xs)
print N.shape(sums)

cmap = cm.get_cmap("gray")
cmap._init()
cmap._lut[:-3, :-1] = 0.
cmap._lut[:-3, -1] = N.linspace(1, 0, cmap.N)

P.figure(figsize=(10,5))
ax1 = P.subplot(121)
ax1.contour(Xs[:-1], Ys[:-1], sums.T, colors='k')
ax1.pcolor(Xs[:-1], Ys[:-1], sums.max() - sums.T, cmap=cmap)
ax1.set_xlabel(r'$t_{smooth}$')
ax1.set_ylabel(r'$\tau_{smooth}$')
ax1.set_ylim(0, 3)
ax2 = P.subplot(122)
ax2.contour(Xd[:-1], Yd[:-1], sumd.T, colors='k')
ax2.pcolor(Xd[:-1], Yd[:-1], sumd.max() - sumd.T, cmap=cmap)
ax2.set_xlabel(r'$t_{disc}$')
ax2.set_ylabel(r'$\tau_{disc}$')
ax2.set_ylim(0, 3)
P.tight_layout()
P.savefig('bamford_plot_'+str(len(hund))+'.pdf')




