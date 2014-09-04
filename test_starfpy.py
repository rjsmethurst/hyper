from posterior import *

import numpy as N
import pylab as P
import pyfits as F
import os

import matplotlib.cm as cm


font = {'family':'serif', 'size':16}
P.rc('font', **font)
P.rc('xtick', labelsize='medium')
P.rc('ytick', labelsize='medium')
P.rc('axes', labelsize='x-large')

#age = 12.8
#
#tq = N.linspace(1, 13, 5)
#tq = N.sort(tq)
#tqp = tq + abs(N.random.normal(0.5, 0.4, 5))
#tqm = abs(tq - abs(N.random.normal(0.5, 0.4, 5)))
#
#tau = N.linspace(0.25, 3.75, 5)
#tau = N.sort(tau)
#taup = tau + abs(N.random.normal(0.5, 0.4, 5))
#taum = abs(tau - abs(N.random.normal(0.5, 0.4, 5)))
#
#k = N.array(list(product(tq, tau)))
#kp = N.array(list(product(tqp, taup)))
#km = N.array(list(product(tqm, taum)))
#
#P.figure()
#P.scatter(k[:,0], k[:,1])
#P.show()
#
#col = N.zeros_like(k)
#colp = N.zeros_like(kp)
#colm = N.zeros_like(km)
#for j in range(len(k)):
#    col[j,0], col[j,1] = predict_c_one([k[j,0], k[j,1]], age)
#    colp[j,0], colp[j,1] = predict_c_one([kp[j,0], kp[j,1]], age)
#    colm[j,0], colm[j,1] = predict_c_one([km[j,0], km[j,1]], age)
#P.figure()
#P.scatter(col[:,1], col[:,0])
#P.show()
#
#error_u_r = 0.124
#error_nuv_u = 0.215
#
#w = [7.5, 1.5, 7.0, 1.5, 4.0, 1.5, 4.0, 1.5]
#nwalkers = 100
#nsteps= 400
#start = [7.5, 1.5]
#burnin =400
#
#save = N.zeros((len(col), 17))
#save[:,1] = k[:,0]
#save[:,2] = kp[:,0]
#save[:,3] = km[:,0]
#save[:,4] = k[:,1]
#save[:,5] = kp[:,1]
#save[:,6] = km[:,1]
#save[:,7] = col[:,0]
#save[:,8] = error_nuv_u
#save[:,9] = col[:,1]
#save[:,10] = error_u_r


#for n in range(len(k)):
#    print 'starting run number: ', n
#    s, ss = sample(2, nwalkers, nsteps, burnin, start, w, col[n,1], error_u_r, col[n,0], error_nuv_u, age, 1, 1, n)
#    tq_mcmc, tau_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(s, [16,50,84],axis=0)))
#    save[n,0] = n
#    save[n,11] = tq_mcmc[0]
#    save[n,12] = tq_mcmc[1]
#    save[n,13] = tq_mcmc[2]
#    save[n,14] = tau_mcmc[0]
#    save[n,15] = tau_mcmc[1]
#    save[n,16] = tau_mcmc[2]
#    N.save('test_starfpy_results.npy', save)
#    #s = N.load(dir+l[n])
#    fig = corner_plot(s, [r'$t_q$', r'$\tau$'], [[0, 13.807108309208775], [0, 4]], [(save[n,11], save[n,12], save[n,13]),(save[n,14], save[n,15], save[n,16])], save[n,0], [save[n,1], save[n,4]])
#    fig.savefig('corner_test_starfpy_%s.png' % n, dpi=200)


def place_image(ax, im):
    ax.imshow(im)
    ax.tick_params(axis='x', labelbottom='off', labeltop='off', bottom='off', top='off')
    ax.tick_params(axis='y', labelleft='off', labelright='off', left='off', right='off')

n=0
a = N.arange(25).reshape(5,5)
fig, axes = P.subplots(nrows=5, ncols=5, figsize=(100,100), edgecolor='None')
for row in axes:
    for ax in row:
        im = mpimg.imread('corner_test_starfpy_'+str(a.T.flatten()[n])+'.png')
        place_image(ax, im)
        n+=1
P.tight_layout()
P.subplots_adjust(wspace=0.0)
P.subplots_adjust(hspace=0.0)
P.savefig('mosaic_test.png')



    
