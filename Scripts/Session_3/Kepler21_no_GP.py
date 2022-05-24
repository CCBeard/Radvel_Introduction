import numpy as np
import pandas as pd
import os
import radvel
from radvel.plot import orbit_plots, mcmc_plots
from scipy import optimize
import matplotlib.pyplot as plt


#read in the datafile
data = pd.read_csv('../../Data/Kepler21_rv.csv',sep=' ')

#cutoff = 2459220
#data = data[data.time < cutoff]

t = np.array(data.time)
vel = np.array(data.mnvel)
errvel = np.array(data.errvel)
tel = np.array(data.tel)
telgrps = data.groupby('tel').groups
instnames = telgrps.keys()




Tc = 2456798.7188

#there is only one known planet in this system
nplanets=1
params = radvel.Parameters(nplanets,basis='per tc secosw sesinw k')

params['per1'] = radvel.Parameter(value=2.78578,vary=True)
params['tc1'] = radvel.Parameter(value=Tc,vary=True)
params['secosw1'] = radvel.Parameter(value=0.,vary=False)
params['sesinw1'] = radvel.Parameter(value=0.,vary=False)
params['k1'] = radvel.Parameter(value=2.0)
params['dvdt'] = radvel.Parameter(value=0.0,vary=False)
params['curv'] = radvel.Parameter(value=0.0,vary=False)



model = radvel.model.RVModel(params)


#dictionary of starting guesses for instrument jitter
jit_guesses = {'hires_post':1.0, 'HARPS_N':2.0}#, 'NEID': 1.0}

likes = []
def initialize(tel_suffix):
# Instantiate a separate likelihood object for each instrument.

# Each likelihood must use the same radvel.RVModel object.
	indices = telgrps[tel_suffix]

	like = radvel.likelihood.RVLikelihood(model, t[indices], vel[indices],
	errvel[indices],  suffix='_'+tel_suffix,
	)
# Add in instrument parameters
	like.params['gamma_'+tel_suffix] = radvel.Parameter(value=0.0)
	like.params['jit_'+tel_suffix] = radvel.Parameter(value=jit_guesses[tel_suffix])
	likes.append(like)
for tel in instnames:
    initialize(tel)


#Combine the likelihoods of all instruments into one
rvlike = radvel.likelihood.CompositeLikelihood(likes)

# Generate a posterior object
post = radvel.posterior.Posterior(rvlike)

sigma = np.mean(errvel) #Defined in Haywood et al. 14, stupidly. Used in a few priors

#post.priors += [radvel.prior.HardBounds('secosw1', -1, 1)] #not technically accurate to the paper, but close
#post.priors += [radvel.prior.HardBounds('sesinw1', -1, 1)]

post.priors += [radvel.prior.Gaussian('tc1', 2456798.7188 ,0.00085)] # constrained from transit

post.priors += [radvel.prior.Gaussian('per1',2.7858212,0.0000032)] #constrained from transit

#Lopez-Morales uses a different, modified, Jeffreys Prior, which isn't in radvel by default.
#It can be custom added, but I have not done so for this tutorial. This prior is close
post.priors += [radvel.prior.Jeffreys('k1', 0.01, 10.0)]


post.priors += [radvel.prior.Jeffreys('jit_hires_post', 0.01, 10.0)]
post.priors += [radvel.prior.Jeffreys('jit_HARPS_N', 0.01, 10.0)]


post.priors += [radvel.prior.HardBounds('gamma_HARPS_N', -100.0, 100.0)]
post.priors += [radvel.prior.HardBounds('gamma_hires_post', -100.0, 100.0)]





print(post)


post = radvel.fitting.maxlike_fitting(post, verbose=True, method='Powell')


print(post)

nwalk=150
chains = radvel.mcmc(post,nrun=10000,ensembles=3,nwalkers=nwalk, serial=True)



Plot = orbit_plots.MultipanelPlot(
post,
telfmts={'hires_post':dict(fmt='o', label='HIRES', color='indianred', markersize=7, markeredgecolor='black', mew=1.),
'HARPS_N':dict(fmt='o', label='HARPS-N', color='cyan', markersize=7, markeredgecolor='black', mew=1.)},
)
Plot.plot_multipanel()
plt.show()

quants = chains.quantile([0.159, 0.5, 0.841]) # median & 1sigma limits of posterior distribution
for par in post.params.keys():
	if post.params[par].vary:
		med = quants[par][0.5]
		high = quants[par][0.841] - med
		low = med - quants[par][0.159]
		err = np.mean([high,low])
		err = radvel.utils.round_sig(err)
		med, err, errhigh = radvel.utils.sigfig(med, err)
		print('{} : {} +/- {}'.format(par, med, err))


#make a corner plot
Corner = mcmc_plots.CornerPlot(post,chains)
Corner.plot()

#can be useful for model comparison
print('Final loglikelihood')
print(post.logprob())

#can be useful for model comparison
print('BIC')
print(round(post.bic(), 4))


telsout = post.likelihood.telvec
for item in instnames:
	dataset = (telsout==item)
	print("RMS of "+ str(item)+ " = %4.2f" % np.std(post.likelihood.residuals()[dataset]))
