import numpy as N
import pylab as P
import emcee
import triangle
import time
from hyper import *
from scipy.stats import norm
import os

font = {'family':'serif', 'size':25}
P.rc('font', **font)
P.rc('xtick', labelsize='medium')
P.rc('ytick', labelsize='medium')
P.rc('axes', labelsize='x-large')

nwalkers = 100
ndim = 8
burnin = 500
nsteps = 500
start = [7, 0.5, 1.5, 0.5, 7.0, 0.5, 1.5, 0.5]
#real_alpha = [3.2, 0.5, 0.64, 0.5, 10.56, 0.5, 2.56, 0.5]

#ts = (real_alpha[1])*N.random.randn(1,50,20) + real_alpha[0]
#taus = (real_alpha[3])*N.random.randn(1,50,20) + real_alpha[2]
#td = (real_alpha[5])*N.random.randn(1,50,20) + real_alpha[4]
#taud = (real_alpha[7])*N.random.randn(1,50,20) + real_alpha[6]
#ts = N.random.normal(loc=real_alpha[0], scale=real_alpha[1], size=(1, 50, 20))
#taus = N.random.normal(loc=real_alpha[2], scale=real_alpha[3], size=(1, 50, 20))
#td = N.random.normal(loc=real_alpha[4], scale=real_alpha[5], size=(1, 50, 20))
#taud = N.random.normal(loc=real_alpha[6], scale=real_alpha[7], size=(1, 50, 20))
#
#
#t = N.append(ts, td, axis=1)
#tau = N.append(taus, taud, axis=1)
#s = N.append(t, tau, axis=0)
#print N.shape(s)
#
#P.figure()
#P.scatter(t, tau)
#P.show()
#
#pss = 0.2*N.random.ranf(50) + 0.8
#pdd = 0.2*N.random.ranf(50) + 0.8
##pss = N.ones(50)
##pdd = N.ones(50)
#ps = N.append(pss, 1-pdd, axis=0).reshape(-1,1)
#pd = N.append(1-pss, pdd, axis=0).reshape(-1,1)
#print pd.shape

h = N.load('/Users/becky/Projects/hyper_starfpy/red_s_10.npy')
#h = h[:,:]
i = N.argsort(h[:,13])
ps = h[:,2][i].reshape(-1,1)
pd = h[:,3][i].reshape(-1,1)

dir='/Users/becky/Projects/hyper_starfpy/samples/red_s/'
l= os.listdir(dir)
#l = l[1:]
t = N.zeros((1, len(l), 50))
tau = N.zeros((1, len(l), 50))

for j in range(len(l)):
    samples = N.load(dir+l[j])
    n = N.random.randint(0, len(samples), 50)
    t[0,j,:] = samples[n,0]
    tau[0,j,:]=samples[n,1]

s = N.append(t, tau, axis=0)
print N.shape(s)

st = time.time()
p0 = [start +1e-4*N.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=2, args=(s, pd, ps))
pos, prob, state = sampler.run_mcmc(p0, burnin)
samples = sampler.chain[:,:,:].reshape((-1,ndim))
samples_save = 'samples_burn_in_test_data_hyper_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy'
N.save(samples_save, samples)
biprob = sampler.flatlnprobability
N.save('samples_burnin_red_s_test_data_lnprob_hyper_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy', biprob)
walker_plot(samples, nwalkers, burnin)
sampler.reset()
print 'RESET', pos
# main sampler run
sampler.run_mcmc(pos, nsteps)
samples = sampler.chain[:,:,:].reshape((-1,ndim))
samples_save = 'samples_test_red_s_data_hyper_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy'
N.save(samples_save, samples)
prob = sampler.flatlnprobability
N.save('samples_lnprob_test_data_hyper_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy', prob)
print 'acceptance fraction', sampler.acceptance_fraction
print 'time taken in minutes...', (time.time() - st)/60.0
emcee_alpha = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(samples, [16,50,84],axis=0)))
print 'emcee found values for alpha...', emcee_alpha
fig = triangle.corner(samples, labels=[r'$\mu_t^{s}$', r'$\sigma_t^{s}$',r'$\mu_{\tau}^{s}$',r'$\sigma_{\tau}^{s}$', r'$\mu_t^{d}$', r'$\sigma_t^{d}$',r'$\mu_{\tau}^{d}$',r'$\sigma_{\tau}^{d}$'])
fig.savefig('triangle_hyper_params_test_red_s_data_marshall.pdf')

mc_uts = emcee_alpha[0][0]
mc_sts = emcee_alpha[1][0]
mc_utaus =emcee_alpha[2][0]
mc_staus = emcee_alpha[3][0]
mc_utd = emcee_alpha[4][0]
mc_std = emcee_alpha[5][0]
mc_utaud = emcee_alpha[6][0]
mc_staud = emcee_alpha[7][0]

ts = N.linspace(0, 14, 500)
taus = N.linspace(0, 5, 500)
td = N.linspace(0, 14, 500)
taud = N.linspace(0, 5, 500)

dist_ts = norm(loc=mc_uts, scale=mc_sts**0.5)
dist_taus = norm(loc=mc_utaus, scale=mc_staus**0.5)
dist_td = norm(loc=mc_utd, scale=mc_std**0.5)
dist_taud = norm(loc=mc_utaud, scale=mc_staud**0.5)

pdf_ts = dist_ts.pdf(ts)
pdf_taus = dist_taus.pdf(taus)
pdf_td = dist_td.pdf(td)
pdf_taud = dist_taud.pdf(taud)

prob1 = N.outer(pdf_ts, pdf_taus)
prob2 = N.outer(pdf_td, pdf_taud)

P.figure()
P.scatter(t, tau, color='k', marker='x')
P.contour(prob1.T, extent=(N.min(ts), N.max(ts), N.min(taus), N.max(taus)), colors='r')
P.contour(prob2.T, extent=(N.min(td), N.max(td), N.min(taud), N.max(taud)), colors='b')
P.xlabel(r'$t_{quench}$')
P.ylabel(r'$\tau$')
P.tight_layout()
P.savefig('best_fit_compare_red_s_data.pdf')

