

# first stab at Gillespie
# definitely needs to be revised/extensibilized

import numpy as np
import matplotlib.pyplot as plt

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
                time_step = np.random.exponential(total_rate)
                
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
            while self.time_points[i][di] < t:
                di += 1
            # add data right before
            data[i,:] = self.data_points[i][di-1]
        
        return data


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
        return d
    else:
        x[0] -= 1
        return x

def death_rate(d,kd):
    return kd * d[0]

G = Gillespie([0])
KB = 1.2
KD = 1.0
G.add_reaction(birth,lambda x: birth_rate(x,KB))
G.add_reaction(death,lambda x: death_rate(x,KD))
G.simulate(100.0,1000)
results = G.get_data(100.0)

hist, bins = np.histogram(results,bins = 20)
width = 0.7*(bins[1]-bins[0])
center = (bins[:-1]+bins[1:])/2
plt.bar(center, hist, align = 'center', width = width)
plt.show()
