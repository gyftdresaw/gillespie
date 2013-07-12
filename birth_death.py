
from Gillespie import Gillespie
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# quick test
def birth(d):
    x = d[:] # make a copy because weird stuff is happening
    x[0] += 1
    return x

def birth_rate(d,kb):
    return kb

def death(d):
    x = d[:]
    if x[0] == 0:
        return x
    else:
        x[0] -= 1
        return x

def death_rate(d,kd):
    return kd * d[0]

G = Gillespie([0])
KB = 2.0
KD = 0.5
T = 1000
G.add_reaction(birth,lambda x: birth_rate(x,KB))
G.add_reaction(death,lambda x: death_rate(x,KD))
G.simulate(10,T)

# a few plots
# distribution at a few times
times = np.linspace(0.1,3,7)
for_hist = np.empty((T,len(times)))
for i in xrange(len(times)):
    results = G.get_data(times[i])
    for_hist[:,i] = results[:,0]

plt.figure()
n,bins,patches = plt.hist(for_hist,bins=range(10),align='left',normed=True)
# add expected poisson distribution lines
# disregard time 0, distribution at that time doesn't make sense
poiss_lines = np.empty((len(bins),len(times)))
for i in xrange(len(times)):
    mu_expected = (KB/KD) * (1.0 - np.exp(-KD * times[i]))
    poiss_lines[:,i] = poisson.pmf(bins,mu_expected)
plt.plot(bins,poiss_lines)
plt.legend(times)
plt.title('distribution in time')
plt.show()

# plot mean in time
t = np.linspace(0,10,100)
mean_data = G.get_moment(t,1)
plt.figure()
plt.plot(t,mean_data)
mean_exp = (KB/KD) * (1.0 - np.exp(-KD * t))
plt.plot(t,mean_exp,'--')
plt.title('mean in time')
plt.show()

# plot variance in time
var_data = G.get_moment(t,2) - (mean_data * mean_data)
plt.figure()
plt.plot(t,var_data)
plt.plot(t,mean_exp,'--')
plt.title('var in time')
plt.show()

