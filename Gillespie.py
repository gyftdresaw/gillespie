

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

    # checks if data satisfies fpt_thresh
    # if item in fpt_thresh is None, it is treated
    # as if there's no condition on that species
    # otherwise we check for equality
    def fpt_satisfied(self,fpt_thresh,data):
        if fpt_thresh is None:
            return True
        checks = [fpt_thresh[i] == data[i] for i in xrange(len(data)) if not fpt_thresh[i] is None]
        return reduce(lambda x,y: x and y, checks)

    # desired time t, number of trials trials
    # for calculating first passage times:
    #   fpt_thresh is a list of conditions on each species
    #   that need to reached before each run terminates
    def simulate(self,t,trials,fpt_thresh=None):
        self.trials = trials
        self.data_points = [[self.IC] for i in xrange(trials)]
        self.time_points = [[0.0] for i in xrange(trials)]
        # perform simulation trials times
        for i in xrange(trials):
            time = 0.0
            data = self.IC
            fpt_sat = self.fpt_satisfied(fpt_thresh,data)
            while time < t or not fpt_sat:
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
                
                # if our fpt conditions have not yet been satisfied
                # check again with updated data
                if not fpt_sat:
                    fpt_sat = self.fpt_satisfied(fpt_thresh,data)
        
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

    # for each trial return the first time
    # when fpt_thresh conditions are met
    def get_fpt(self,fpt_thresh):
        fptimes = np.empty(self.trials)
        for i in xrange(self.trials):
            j = 0
            while not self.fpt_satisfied(fpt_thresh,self.data_points[i][j]):
                j += 1
            fptimes[i] = self.time_points[i][j]
        
        return fptimes

