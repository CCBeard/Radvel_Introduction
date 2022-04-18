import numpy as np
import pandas as pd
import os
import radvel
import radvel.likelihood
from radvel.plot import orbit_plots, mcmc_plots
from scipy import optimize

################################
###         51 Peg           ###
################################


data = pd.read_csv('../Data/Peg_51_rv_hires.csv',sep=' ')

t = np.array(data.time)
vel = np.array(data.mnvel)
errvel = np.array(data.errvel)
tel = np.array(data.tel)

telgrps = data.groupby('tel').groups
instnames = telgrps.keys()


nplanets=1
params = radvel.Parameters(nplanets,basis='per tc secosw sesinw k')

params['per1'] = radvel.Parameter(value=4.23079690) #period in days
params['tc1'] = radvel.Parameter(value=2455705.001) #Time of conjunction BJD
params['sesinw1'] = radvel.Parameter(value=0.,vary=False) # fix eccentricity = 0
params['secosw1'] = radvel.Parameter(value=0.,vary=False)
params['k1'] = radvel.Parameter(value=50.) #initial guess for K amplitude, m/s

params['dvdt'] = radvel.Parameter(value=0.,vary=False) #linear trend term
params['curv'] = radvel.Parameter(value=0.,vary=False) #curvature term


model = radvel.model.RVModel(params) #combine all the parameters into a Keplerian model under the hood

jit_guesses = {'hires_j': 1.0} #ditionary of starting guesses for each instruments jitter

likes = [] #a list of likelihoods for each instrument

def initialize(tel_suffix):

    # Instantiate a separate likelihood object for each instrument.
    # Each likelihood must use the same radvel.RVModel object.

    indices = telgrps[tel_suffix]
    like = radvel.likelihood.RVLikelihood(model, t[indices], vel[indices],
                                          errvel[indices], suffix='_'+tel_suffix,
                                          )
    # Add in instrument parameters
    like.params['gamma_'+tel_suffix] = radvel.Parameter(value=np.mean(vel[indices]), vary=True)
    like.params['jit_'+tel_suffix] = radvel.Parameter(value=jit_guesses[tel_suffix], vary=True)
    likes.append(like)

#loop through each instrument and append the likelihood
for tel in instnames:
    initialize(tel)

#merge all the likelihoods into a composite likelihood object for calculations
like = radvel.likelihood.CompositeLikelihood(likes)

#instantiate a posterior object and start defining priors
post = radvel.posterior.Posterior(like)

# Define your priors here

#orbital priors
post.priors += [radvel.prior.Gaussian('per1', 4.23079690, 1.0)] #let's say we know this kind of well
post.priors += [radvel.prior.Gaussian('tc1', 2455705.001, 0.1)]
post.priors += [radvel.prior.HardBounds('k1', 0.01, 100.)] #Let's say we don't know this at all, so we make it very wide

#instrumental priors
post.priors += [radvel.prior.HardBounds('jit_hires_j', 0.01, 10.)]
post.priors += [radvel.prior.HardBounds('gamma_hires_j', -100.,100.)]

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
    telfmts={'hires_j':dict(fmt='o', label='HIRES', color='indianred', markersize=7, markeredgecolor='black', mew=1.),},)
Plot.plot_multipanel()

#Corner plots can help diagnose problems
Corner = mcmc_plots.CornerPlot(post, chains) # posterior distributions
Corner.plot()
