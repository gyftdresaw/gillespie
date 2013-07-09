

# first stab at Gillespie
# definitely needs to be revised/extensibilized

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

class Gillespie:
    def __init__(self,IC):
        self.reactions = [] # list of functions governing species
        self.rates = [] # list of functions governing rates
        self.IC = IC # initial conditions list
        

    def add_reaction(self,f_reaction,f_rate):
        self.reactions.append(f_reaction)
        self.rates.append(f_rate)

    def get_rates(self,curr):
        return np.array([f(curr) for f in self.rates])

    # desired time t, number of trials trials
    def simulate(self,t,trials):
        self.trials = trials
        self.data_points = [[self.IC] for i in xrange(trials)]
        self.time_points = [[0.0] for i in xrange(trials)]
        # perform simulation trials times
        for i in xrange(trials):
            time = 0.0
            data = self.IC
            while time < t:
                rates = self.get_rates(data)
                total_rate = reduce(lambda x,y: x+y,rates)
                # input to exponential is scale = 1/rate
                time_step = np.random.exponential(1.0/total_rate)
                # print (time,data,rates,total_rate,time_step)
                
                time += time_step
                self.time_points[i].append(time) # record time
                norm_rates = rates/sum(rates) # rates need to be floats
                # pretty hackish
                choice = np.nonzero(np.random.multinomial(1,norm_rates))[0][0]
                data = self.reactions[choice](data)
                # print (time_step,choice,norm_rates,data,time)
                self.data_points[i].append(data)
                # print self.data_points
        
    # get all data from simulations at time t
    def get_data(self,t):
        data = np.empty((self.trials,len(self.IC)))
        for i in xrange(self.trials):
            di = 0
            while self.time_points[i][di] <= t:
                di += 1
            # add data right before
            data[i,:] = self.data_points[i][di-1]
        # data is (trials x num_species)
        return data

    # for each time in times 
    # calculate moment m for each species
    def get_moment(self,times,m):
        mvals = np.empty((len(times),len(self.IC)))
        for i in xrange(len(times)):
            d = self.get_data(times[i])
            # calculate moment down the columns
            mvals[i,:] = np.mean(d ** m, axis = 0)
        # mvals is (times x num_species)
        return mvals

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




