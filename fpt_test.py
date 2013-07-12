
from Gillespie import Gillespie
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# test first passage time for simple birth death
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


T = 1000
KB = 1.0
KD = 0.0
fpt_param = 1

G = Gillespie([0])
G.add_reaction(birth,lambda x: birth_rate(x,KB))
G.add_reaction(death,lambda x: death_rate(x,KD))

fpt = [fpt_param]
G.simulate(10.0,T,fpt_thresh = fpt)
fptimes = G.get_fpt(fpt)

n,bins,patches = plt.hist(fptimes,bins=20,normed=True)
xmin,xmax,ymin,ymax = plt.axis()
to_plot = np.linspace(xmin,xmax,1000)[1:]
# plot gamma on top of this
gpdf = gamma.pdf(to_plot,fpt_param,scale=1.0/KB)

plt.plot(to_plot,gpdf,'--')
plt.show()
