import sys
import numpy as np
import pandas as pd
import os
import radvel
from radvel.plot import orbit_plots, mcmc_plots
from scipy import optimize
from scipy import spatial

#%matplotlib inline

#Define quantities
t = [] #Timestamps
v = [] #Radial velocity
e = [] #Uncertainty on RV
ds = [] #Integer array denoting which data set each point is from

data = pd.read_csv(os.path.join('../../Data/HD191939_TKS_IX_data.csv'), sep=',')
t = np.array(data['time'])
v = np.array(data['mnvel'])
e = np.array(data['errvel'])
telgrps = data.groupby('tel').groups

#Set instruments
inst = ['APF','HIRES']

#Set Keplerian parameters
nplanets= #how many planets in this system?
params = radvel.Parameters(nplanets,basis='per tc secosw sesinw k')

#Initial guesses for all parameters.

params['per1'] = radvel.Parameter(value=, vary=False)

##############################################
# You need to add all the parameters here ####
##############################################

#Instantiate Model
model = radvel.model.RVModel(params)

#Set up instrument-specific parameters and objects
jit_guesses = #need to add jitter guesses

likes = []
def initialize(tel_suffix):
# Instantiate a separate likelihood object for each instrument.

	# Each likelihood must use the same radvel.RVModel object.
	indices = telgrps[tel_suffix]
	#print(indices)

	like = radvel.likelihood.RVLikelihood(model, t[indices], v[indices], e[indices], suffix='_'+tel_suffix)
	# Add in instrument parameters
	like.params['gamma_'+tel_suffix] = radvel.Parameter(value=np.mean(v[indices]))
	like.params['jit_'+tel_suffix] = radvel.Parameter(value=jit_guesses[tel_suffix])
	likes.append(like)

for tel in inst:
    initialize(tel)

#Instantiate composite GP likelihood function that accounts for multiple instruments
rvlike = radvel.likelihood.CompositeLikelihood(likes)

#Instantiate posterior
post = radvel.posterior.Posterior(rvlike)


##############################################
# You need to add all the priors here     ####
##############################################
#Planets
#what priors do we need?


# A sanity check. Do all the numbers look right?
print(post)

#optimize the model to make MCMC more robust
res = optimize.minimize(
    post.neglogprob_array, post.get_vary_params(), method='Nelder-Mead',
    options=dict(maxiter=200, maxfev=100000, xatol=1e-8)
)

#print the optimized parameters
print(post)

#This does the MCMC
# nwalkers is the number of independent nwalkers
# nrun is the maximum number of steps per walker before the sampler stops
# ensembles is the number of computer cores to use
chains = radvel.mcmc(post,serial=True,nwalkers=150, nrun=10000,ensembles=3)

#Estimate output posteriors
quants = chains.quantile([0.159, 0.5, 0.841]) # median & 1sigma limits of posterior distribu
out = []
for par in post.params.keys():
	if post.params[par].vary:
		med = quants[par][0.5]
		high = quants[par][0.841] - med
		low = med - quants[par][0.159]
		err = np.mean([high,low])
		err = radvel.utils.round_sig(err)
		med, err, errhigh = radvel.utils.sigfig(med, err)
		print('{} : {} +/- {}'.format(par, med, err))

# Plot the RV solution
Plot = orbit_plots.MultipanelPlot(
    post,
    telfmts={'HIRES':dict(fmt='o', label='HIRES', color='blue', markersize=7, markeredgecolor='blue', mew=1.),
	         'APF':dict(fmt='o', label='APF', color='red', markersize=7, markeredgecolor='red', mew=1.)},)
Plot.plot_multipanel()

#Corner plots can help diagnose problems
Corner = mcmc_plots.CornerPlot(post, chains) # posterior distributions
Corner.plot()
