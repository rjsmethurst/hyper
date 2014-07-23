import numpy as N
import pylab as P
import pyfits as F
import time
import os
import matplotlib.cm as cm


font = {'family':'serif', 'size':16}
P.rc('font', **font)
P.rc('xtick', labelsize='medium')
P.rc('ytick', labelsize='medium')
P.rc('axes', labelsize='x-large')

dir='/Users/becky/Projects/hyper_starfpy/samples/hund/'
l = os.listdir(dir)
l = l[1:]
t = N.zeros((1, len(l), 50))
tau = N.zeros((1, len(l), 50))

#file = '/Users/becky/Projects/Green-Valley-Project/data/GZ2_all_GALEX_match_GZ1_k_correct_green_valley.fits'
#dat = F.open(file)
#gz2data = dat[1].data
#dat.close()

X = N.linspace(0, 14, 100)
Y = N.linspace(0, 5, 100)


sums=N.zeros((len(X)-1, len(Y)-1))
sumd=N.zeros((len(X)-1, len(Y)-1))

h = N.load('/Users/becky/Projects/hyper_starfpy/gv_20.npy')
h = h[1:20,:]
i = N.argsort(h[:,13])
ps = h[:,2][i]
pd = h[:,3][i]


for j in range(len(l)):
    s =N.load(dir+l[j])
    Hs, Xs, Ys = N.histogram2d(s[:,0].flatten(), s[:,1].flatten(), bins=(X, Y), weights=N.ones(len(s))*ps[j])
    sums += Hs
    Hd, Xd, Yd = N.histogram2d(s[:,0].flatten(), s[:,1].flatten(), bins=(X, Y), weights=N.ones(len(s))*pd[j])
    sumd += Hd
    

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
P.savefig('bamford_plot_'+str(len(h))+'.pdf')

    
