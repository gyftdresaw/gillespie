
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

N = 3
KB = np.array([2.0,0.5,1.0])
T = 1000
sim_time = 4.0
IC = np.ones(N)
G = Gillespie(IC)

# really weird, i is not getting copied when passed to the lambdas
# issue has to do with lambda scope (lambda looks up variable in scope each time
# going to create separate scope for it
def birth_factory(KB,i):
    return (lambda x: birth(x,i),lambda x: birth_rate(x,KB,i))
for i in xrange(N):
    b,brate = birth_factory(KB,i)
    G.add_reaction(b,brate)

print 'starting simulation'
G.simulate(sim_time,T)
print 'simulation done'

# a few plots
# distribution at a few times
times = np.linspace(0.1,3.0,7)
for_hist = np.empty((T,len(times)))
for i in xrange(len(times)):
    results = G.get_data(times[i])
    for_hist[:,i] = results[:,0]

plt.figure()
n,bins,patches = plt.hist(for_hist,bins=10,align='left',normed=True)
plt.legend(['%.1f' % t for t in times])
plt.show()

# plot mean in time
'''
t = np.linspace(0,sim_time,100)
mean_data = G.get_moment(t,1)
plt.figure()
plt.plot(t,mean_data)
plt.legend(range(N))
mean_exp = np.empty((len(t),2))
A = (1.0/2.0)*(1.0 + (KB[0]/gmean(KB)))
B = 1.0 - A
mean_exp[:,0] = A * np.exp(gmean(KB)*t) + B * np.exp(-gmean(KB)*t)
mean_exp[:,1] = (gmean(KB)/KB[0]) * (A * np.exp(gmean(KB)*t) - B * np.exp(-gmean(KB)*t))
plt.plot(t,mean_exp,'--')
plt.title('mean in time')
plt.show()
'''
# checking asymptotics

## first the mean ##

# make some useful matrices and things
kap = gmean(KB)
# eigenvalues, trying to keep order consistent with notes
lams = np.array([kap*np.exp((2*np.pi*m / N)*1j) for m in xrange(1,N+1)])

# holds corresponding eigenvectors
U = np.zeros((N,N)) + 1j * np.zeros((N,N))
for n in xrange(N):
    for m in xrange(N):
        U[n,m] = np.prod(KB[:(n+1)])/ (lams[m] ** (n+1))

# not really kosher but we'll just get the inverse for convenience
Uinv = np.linalg.inv(U)

# mean prediction at various times
# complete mean, not just asymptotic
t = np.linspace(0,sim_time,100)
mean_predict = np.empty((len(t),N))
for i in xrange(len(t)):
    D = np.diag(np.exp(t[i] * lams))
    mean_predict[i,:] = np.real(np.dot(U,np.dot(D,np.dot(Uinv,IC))))

mean_data = G.get_moment(t,1)
# plot it
plt.figure()
plt.plot(t,mean_data)
plt.legend(range(N))
plt.plot(t,mean_predict,'--')
plt.title('mean in time')
plt.show()

V = np.transpose(Uinv)

## now the covariance ##
# fill in covariance entry by entry
M_exact = np.zeros((N,N,len(t))) + 1j * np.zeros((N,N,len(t)))
for a in xrange(N):
    for b in xrange(N):
        # sum over three indices
        for k in xrange(N):
            for j in xrange(N):
                for r in xrange(N):
                    pre = V[k,a] * U[k,j] * V[r,j] * V[k,b] * IC[r] * lams[j]
                    post_num =  (np.exp((lams[a] + lams[b])*t) - np.exp(lams[j]*t))
                    post_denom = lams[a] + lams[b] - lams[j]
                    M_exact[a,b,:] += pre * post_num / post_denom

# asymptotic only NN entry of M matters
M_asymptotic = np.zeros((N,N,len(t))) + 1j * np.zeros((N,N,len(t)))
M_asymptotic[N-1,N-1,:] = M_exact[N-1,N-1,:]

# get S from M
S_exact = np.empty((N,N,len(t)))
for i in xrange(len(t)):
    S_exact[:,:,i] = np.real(np.dot(U,np.dot(M_exact[:,:,i],np.transpose(U))))
    
S_asymptotic = np.empty((N,N,len(t)))
for i in xrange(len(t)):
    S_asymptotic[:,:,i] = np.real(np.dot(U,np.dot(M_asymptotic[:,:,i],np.transpose(U))))

# check this against the real data
cov_data = np.zeros((N,N,len(t)))
for i in xrange(len(t)):
    d = G.get_data(t[i])
    cov_data[:,:,i] = np.cov(d,rowvar=0)

# plot the covariance in time
# get some colormap going on
import matplotlib.cm as cm
colors_primary = cm.Dark2(np.linspace(0,1,N*N))
colors_accent = cm.Accent(np.linspace(0,1,N*N))
plt.figure()
for i in xrange(N):
    for j in xrange(N):
        plt.plot(t,cov_data[i,j,:],color=colors_primary[i*N + j])
plt.legend([(i,j) for i in xrange(N) for j in xrange(N)],loc='upper left')
for i in xrange(N):
    for j in xrange(N):
        plt.plot(t,S_exact[i,j,:],'--',color=colors_primary[i*N + j])
plt.title('cov in time')
plt.show()

# plot scaled
plt.figure()
for i in xrange(N):
    for j in xrange(N):
        plt.plot(t,cov_data[i,j,:]/(mean_data[:,i]*mean_data[:,j]),color=colors_primary[i*N + j])
plt.legend([(i,j) for i in xrange(N) for j in xrange(N)],loc='lower right')
for i in xrange(N):
    for j in xrange(N):
        plt.plot(t,S_exact[i,j,:]/(mean_predict[:,i]*mean_predict[:,j]),'--',color=colors_primary[i*N + j])
plt.title('cov scaled in time')
plt.show()


# check third moment scaling
third_moment = G.get_moment(t,3)
plt.figure()
plt.plot(t,np.log(third_moment))
plt.title('third moment log scale')
plt.show()

# plot scaled distribution

# a few plots
# distribution at a few times
times = np.linspace(0.1,sim_time,7)
means = G.get_moment(times,1)
for_hist = np.empty((T,len(times)))
for i in xrange(len(times)):
    results = G.get_data(times[i])
    for_hist[:,i] = results[:,0]/means[i,0]

plt.figure()
n,bins,patches = plt.hist(for_hist,bins=10,align='left',normed=True)
plt.legend(['%.1f' % t for t in times])
plt.title('scaled distribution species 0')
plt.show()
