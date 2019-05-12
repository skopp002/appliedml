import numpy as np
import matplotlib.pyplot as plt
import pdb

accs = np.array([16,24,16,12,16,11,14,15,9,14,7])

def Gaussian_prob(x, _mu=0, _sigma=1):
    """Gaussian disturbution: probability of getting x given mu=_mu, sigma=_sigma"""
    prob_ofX_given_mu_sigma = (1/np.sqrt(2*np.pi*_sigma**2))*np.exp(-(x-_mu)**2/(2*_sigma**2))
    #print ("For ", x ," probability is" ,prob_ofX_given_mu_sigma)
    return prob_ofX_given_mu_sigma


x_span = np.linspace(0,30, 30)
mu, sigma = accs.mean(), np.std(accs)
# pdb.set_trace()
prob_np_array = Gaussian_prob(x_span, mu, sigma)
plt.plot(x_span, prob_np_array)

plt.title("Gaussian Distribution for $\mu=$ {} , and $\sigma=$ {}".format(mu, sigma))
plt.xlabel(r"No Of Accidents")
plt.ylabel(r"$P(x|\mu, \sigma)$")

plt.show()