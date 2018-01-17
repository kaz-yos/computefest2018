############################################################
# Initial setup
############################################################

import pystan
import stan_utility
import matplotlib
import matplotlib.pyplot as plot
import numpy

light="#DCBCBC"
light_highlight="#C79999"
mid="#B97C7C"
mid_highlight="#A25050"
dark="#8F2727"
dark_highlight="#7C0000"
green="#00FF00"

############################################################
# Create data
############################################################

model = stan_utility.compile_model('generate_data.stan')
fit = model.sampling(seed=194838, algorithm='Fixed_param', iter=1, chains=1)

data = dict(N = 25, M = 3,
            X = fit.extract()['X'][0,:], y = fit.extract()['y'][0,:])

pystan.stan_rdump(data, 'linear_regression.data.R')

############################################################
# Fit initial Stan program
############################################################

data = pystan.read_rdump('linear_regression.data.R')

model = stan_utility.compile_model('linear_regression1.stan')
fit = model.sampling(data=data, seed=4938483)

# Check diagnostics one by one
stan_utility.check_n_eff(fit)
stan_utility.check_rhat(fit)
stan_utility.check_div(fit)
stan_utility.check_treedepth(fit)
stan_utility.check_energy(fit)

# Or all at once
stan_utility.check_all_diagnostics(fit)

# Default visual summaries
fit.plot()
plot.show()

params = fit.extract()

# Plot marginal posteriors
f, axarr = plot.subplots(2, 3)
for a in axarr[0,:]:
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('none')
for a in axarr[1,:]:
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('none')

axarr[0, 0].set_title("beta_1")
axarr[0, 0].hist(params['beta'][:,0], bins = 25, color = dark, ec = dark_highlight)
axarr[0, 0].axvline(x=5, linewidth=2, color=light)

axarr[0, 1].set_title("beta_2")
axarr[0, 1].hist(params['beta'][:,1], bins = 25, color = dark, ec = dark_highlight)
axarr[0, 1].axvline(x=-3, linewidth=2, color=light)

axarr[0, 2].set_title("beta_3")
axarr[0, 2].hist(params['beta'][:,2], bins = 25, color = dark, ec = dark_highlight)
axarr[0, 2].axvline(x=2, linewidth=2, color=light)

axarr[1, 0].set_title("alpha")
axarr[1, 0].hist(params['alpha'], bins = 25, color = dark, ec = dark_highlight)
axarr[1, 0].axvline(x=10, linewidth=2, color=light)

axarr[1, 1].set_title("sigma")
axarr[1, 1].hist(params['sigma'], bins = 25, color = dark, ec = dark_highlight)
axarr[1, 1].axvline(x=1, linewidth=2, color=light)

plot.show()

# Perform a posterior predictive check by plotting
# posterior predictive distributions against data
f, axarr = plot.subplots(2, 2)
for a in axarr[0,:]:
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('none')
for a in axarr[1,:]:
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('none')

axarr[0, 0].set_title("y_1")
axarr[0, 0].hist(params['y_ppc'][:,0], bins = 25, color = dark, ec = dark_highlight)
axarr[0, 0].axvline(x=data['y'][0], linewidth=2, color=light)

axarr[0, 1].set_title("y_5")
axarr[0, 1].hist(params['y_ppc'][:,4], bins = 25, color = dark, ec = dark_highlight)
axarr[0, 1].axvline(x=data['y'][4], linewidth=2, color=light)

axarr[1, 0].set_title("y_10")
axarr[1, 0].hist(params['y_ppc'][:,9], bins = 25, color = dark, ec = dark_highlight)
axarr[1, 0].axvline(x=data['y'][9], linewidth=2, color=light)

axarr[1, 1].set_title("y_15")
axarr[1, 1].hist(params['y_ppc'][:,14], bins = 25, color = dark, ec = dark_highlight)
axarr[1, 1].axvline(x=data['y'][14], linewidth=2, color=light)

plot.show()

############################################################
# Fit with vectorized Stan program
############################################################

model = stan_utility.compile_model('linear_regression2.stan')
fit = model.sampling(data=data, seed=4938483)

# Check diagnostics
stan_utility.check_all_diagnostics(fit)

# Default visual summaries
params = fit.extract()

# Plot marginal posteriors
f, axarr = plot.subplots(2, 3)
for a in axarr[0,:]:
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('none')
for a in axarr[1,:]:
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('none')

axarr[0, 0].set_title("beta_1")
axarr[0, 0].hist(params['beta'][:,0], bins = 25, color = dark, ec = dark_highlight)
axarr[0, 0].axvline(x=5, linewidth=2, color=light)

axarr[0, 1].set_title("beta_2")
axarr[0, 1].hist(params['beta'][:,1], bins = 25, color = dark, ec = dark_highlight)
axarr[0, 1].axvline(x=-3, linewidth=2, color=light)

axarr[0, 2].set_title("beta_3")
axarr[0, 2].hist(params['beta'][:,2], bins = 25, color = dark, ec = dark_highlight)
axarr[0, 2].axvline(x=2, linewidth=2, color=light)

axarr[1, 0].set_title("alpha")
axarr[1, 0].hist(params['alpha'], bins = 25, color = dark, ec = dark_highlight)
axarr[1, 0].axvline(x=10, linewidth=2, color=light)

axarr[1, 1].set_title("sigma")
axarr[1, 1].hist(params['sigma'], bins = 25, color = dark, ec = dark_highlight)
axarr[1, 1].axvline(x=1, linewidth=2, color=light)

plot.show()

# Perform a posterior predictive check by plotting
# posterior predictive distributions against data
f, axarr = plot.subplots(2, 2)
for a in axarr[0,:]:
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('none')
for a in axarr[1,:]:
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('none')

axarr[0, 0].set_title("y_1")
axarr[0, 0].hist(params['y_ppc'][:,0], bins = 25, color = dark, ec = dark_highlight)
axarr[0, 0].axvline(x=data['y'][0], linewidth=2, color=light)

axarr[0, 1].set_title("y_5")
axarr[0, 1].hist(params['y_ppc'][:,4], bins = 25, color = dark, ec = dark_highlight)
axarr[0, 1].axvline(x=data['y'][4], linewidth=2, color=light)

axarr[1, 0].set_title("y_10")
axarr[1, 0].hist(params['y_ppc'][:,9], bins = 25, color = dark, ec = dark_highlight)
axarr[1, 0].axvline(x=data['y'][9], linewidth=2, color=light)

axarr[1, 1].set_title("y_15")
axarr[1, 1].hist(params['y_ppc'][:,14], bins = 25, color = dark, ec = dark_highlight)
axarr[1, 1].axvline(x=data['y'][14], linewidth=2, color=light)

plot.show()
