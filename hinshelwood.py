
from Gillespie import Gillespie
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean

# n hinshelwood model implementation

# define birth process for the ith species
def birth(d,i):
    # watchout!! numpy array slice copying is really weird
    x = np.copy(d) # make a copy because weird stuff is happening
    x[i] += 1
    return x

# birth_rate for the ith species
# kb is a rate parameter array
def birth_rate(d,kb,i):
    return kb[i] * d[i-1]

N = 2
KB = np.array([1.0,0.5])
T = 1000
sim_time = 3.0
G = Gillespie(np.ones(N))

# really weird, i is not getting copied when passed to the lambdas
# issue has to do with lambda scope (lambda looks up variable in scope each time
# going to create separate scope for it
def birth_factory(KB,i):
    return (lambda x: birth(x,i),lambda x: birth_rate(x,KB,i))
for i in xrange(N):
    b,brate = birth_factory(KB,i)
    G.add_reaction(b,brate)

G.simulate(sim_time,T)

# a few plots
# distribution at a few times
times = np.linspace(0.1,3.0,7)
for_hist = np.empty((T,len(times)))
for i in xrange(len(times)):
    results = G.get_data(times[i])
    for_hist[:,i] = results[:,0]

plt.figure()
n,bins,patches = plt.hist(for_hist,bins=10,align='left',normed=True)
plt.show()

# plot mean in time
t = np.linspace(0,sim_time,100)
mean_data = G.get_moment(t,1)
plt.figure()
plt.plot(t,mean_data)
plt.legend(range(N))
mean_exp = np.empty((len(t),2))
A = (1.0/2.0)*(1.0 + (KB[0]/gmean(KB)))
B = 1.0 - A
mean_exp[:,0] = A * np.exp(gmean(KB)*t) + B * np.exp(-gmean(KB)*t)
mean_exp[:,1] = gmean(KB) * (A * np.exp(gmean(KB)*t) - B * np.exp(-gmean(KB)*t))
plt.plot(t,mean_exp,'--')
plt.title('mean in time')
plt.show()


'''
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
'''
