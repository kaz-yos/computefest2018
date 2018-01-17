############################################################
# Initial setup
############################################################

import pystan
import stan_utility
import matplotlib
import matplotlib.pyplot as plot
import numpy

light = "#DCBCBC"
light_highlight = "#C79999"
mid = "#B97C7C"
mid_highlight = "#A25050"
dark = "#8F2727"
dark_highlight = "#7C0000"
green = "#00FF00"

############################################################
# Fit Poisson model
############################################################

data = pystan.read_rdump('poisson_progression.data.R')

model = stan_utility.compile_model('poisson_progression1.stan')
fit = model.sampling(data=data, seed=4938483)

# Check diagnostics one by one
stan_utility.check_all_diagnostics(fit)

params = fit.extract()

# Plot marginal posteriors
plot.set_title("lambda")
plot.hist(params['lambda'], bins = 25, color = dark, ec = dark_highlight)
plot.show()

# Perform a posterior predictive check by plotting
# posterior predictive distributions against data
plot.hist(data['y'], bins=range(15), normed=True, alpha=0.75, color=dark, ec=dark_highlight)
plot.hist(params['y_ppc'].flatten(), bins=range(15), normed=True,
          alpha=0.75, color=mid, ec=mid_highlight)
plot.show()
