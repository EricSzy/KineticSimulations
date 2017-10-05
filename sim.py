import sys
from numpy import *
from openopt import *
import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
np.random.seed(1)

# Number of MonteCarlo iterations
MC_num = 500

# Simulation Parameters
TimePtsCorrect = [.001, .005, .01, .05, .1, .2, .3, .5, 1]
TimePtsMismatch = [1, 2, 3, 4, 5, 6, 7, 10, 15, 30, 60]
NTPConcCorrect = [0.625, 1.25, 2.5, 5, 10, 20, 40, 80, 200]
NTPConcMismatch = [50, 100, 200, 300, 500, 750, 1000, 1500]

# Initial NMR/kinetic parameters and errors

# Tautomeric ES1
kf1, kf1_err = 1.72, .09
kr1, kr1_err = 2509., 116.
# Anioinc ES2
kf2, kf2_err = 1.13, 0.14
kr2, kr2_err = 1479., 149.
# ES1 > ES2
kf3, kf3_err = 951., 84.
# ES2 > E1
kf4, kf4_err = 856., 77.

#Set Depending on Model and if ES2 is occuring
k_2i = 100


#=====================
# Fitting Equations
#==
def expfun(p, X):
    a,R = p
    return a*(1-exp(-R*X))
#==
def expfit(X, Y, p0):
    #-
    def chi2(p):
        YS = expfun(p,X)
        return sum((Y-YS)**2)
    #-
    nlp = NLP(chi2, p0)
    result = nlp.solve('ralg', iprint= -1)
    return result.xf
#==
def polfun(p, X):
    k,r = p
    return ((r*X)/(k+X))
#==
def polfit (X,Y, p0):
    #-
    def chi2(p):
        YS = polfun(p, X)
        return sum((Y-YS)**2)
    #-
    nlp = NLP(chi2, p0)
    result = nlp.solve('ralg', iprint = -1)
    return result.xf
#=====================
def Fitting(SchemeDict, TimeList, NTPlist, ProdAmblitudeFitGuess, kObsGuess, kPolGuess, KdGuess):
#Fitting for kobs and kpol. Inputs are the approprate Scheme Dict of [dNTP] and Time point resutls, as well as
#the approprate Correct/Incorrect Time and [dNTP] conditions 
    x = 0
    ListOfkObs = []
    for key in SchemeDict.keys():
    # Fetch n or n+x list from Dict of 'product populations'
        temp_list = list(SchemeDict.values()[x])
    # Flatten list of list to single list
        ProdValues = [val for sublist in temp_list for val in sublist]
    # Add one to access next key in next cycle
        x = x+1 
    # Data formatting
        data1 = column_stack(TimeList)
        data2 = column_stack(ProdValues)
    # Fitting Programs for kobs
        a,R = expfit(data1, data2, [ProdAmblitudeFitGuess, kObsGuess])
        FF = expfun([a,R], data1)
        #Add kobs value to list
        ListOfkObs.append(R)
    
## Fitting for kpol from kobs values ##
# Data formatting
    data1 = column_stack(NTPlist)
    data2 = column_stack(ListOfkObs)
# Fitting
    k,r = polfit(data1, data2, [kPolGuess, KdGuess])
    F = polfun([k,r],data1)
    #print 'kpol', r
    #print 'Kd', k
    return r, k

#===================
# Kinetic Simulations
# Correct and Incorrect Simulations share the same set of rate constants, save for the 
# inclusion of tautomerization/ionization for incorrect incorporation.

# Shared rate constants are declared here; dNTP on and off rate are declared in each scheme

k2 = 268 #forward conformational change rate constant
k2t = k2 
k2i = k2
k_2 = 100 #reverse conformational change rate constant
k_2t = k_2
k3 = 9000 #forward chemisty rate constant
k_3 = .004 #reverse chemisty rate constant


#===================
#Run Kinetic Sheme #1 
def SimulateSchemeOne():
    SchemeOneProduct = []
    for Conc in NTPConcCorrect:
        # Defining additioanl rate constants and starting populations
        C0 = array([1.0, 0.0, 0.0, 0.0]) #Simulation starts with 100% population as E-DNA. 
        k1 = Conc * 100  # dNTP on rate
        k_1 = 1900 # dNTP off rate i.e. 1900 s-1 is equal to a 19uM Kd

        # Rate Matrix
        K = zeros((4,4))
        K[0, 0] = -k1
        K[0, 1] = k_1
        K[1, 0] = k1
        K[1, 1] = -k_1-k2
        K[1, 2] = k_2
        K[2, 1] = k2
        K[2, 2] = -k_2-k3
        K[2, 3] = k_3
        K[3, 2] = k3
        K[3, 3] = -k_3
        
        w,M = linalg.eig(K)
        M_1 = linalg.inv(M)

        # Simulate for each timepoint
        for num in TimePtsCorrect:
            T = linspace (0, float(num), 2)
            B = zeros(T.shape)
            C = zeros(T.shape)
            D = zeros(T.shape)
            E = zeros(T.shape)
            

            for i,t in enumerate(T):
                A = dot(dot(M,diag(exp(w*t))), M_1)
                B[i] = dot(A[0,:], C0)
                C[i] = dot(A[1,:], C0)
                D[i] = dot(A[2,:], C0)
                E[i] = dot(A[3,:], C0)
                
                
            SchemeOneProduct.append(E[-1])
# Data Handling
    SchemeOneDct  = collections.OrderedDict()
    x = 0
    for Number in NTPConcCorrect:
        SchemeOneDct['%s' % Number] = [SchemeOneProduct[x:x+len(TimePtsCorrect)]] 
        x = x + len(TimePtsCorrect)
    
    kpolOne, kdOne = Fitting(SchemeOneDct, TimePtsCorrect, NTPConcCorrect, .99, 5, 268, 19)
    return kpolOne, kdOne

def SimulateSchemeTwo(kt, k_t, ki, k_i, kti, kit):
    SchemeTwoProduct = []
    for Conc in NTPConcMismatch:
        C0 = array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        k1 = Conc * 100 # dNTP on rate
        k_1 = 70000 # dNTP off rate

        K = zeros((6, 6))
        K[0, 0] = -k1
        K[0, 1] = k_1
        K[1, 0] = k1
        K[1, 1] = -k_1-kt-ki
        K[1, 2] = k_t
        K[1, 3] = k_i
        K[2, 1] = kt
        K[2, 2] = -k_t-k2t-kti
        K[2, 3] = kit
        K[2, 4] = k_2t
        K[3, 1] = ki
        K[3, 2] = kti
        K[3, 3] = -k_i-k2i-kit
        K[3, 4] = k_2i
        K[4, 2] = k2t
        K[4, 3] = k2i
        K[4, 4] = -k_2t-k_2i-k3
        K[4, 5] = k_3
        K[5, 4] = k3
        K[5, 5] = -k_3

        w,M = linalg.eig(K)
        M_1 = linalg.inv(M)

        for num in TimePtsMismatch:
            T = linspace (0, float(num), 2)
            B = zeros(T.shape)
            C = zeros(T.shape)
            D = zeros(T.shape)
            E = zeros(T.shape)
            F = zeros(T.shape)
            G = zeros(T.shape)
            

            for i,t in enumerate(T):
                A = dot(dot(M,diag(exp(w*t))), M_1)
                B[i] = dot(A[0,:], C0)
                C[i] = dot(A[1,:], C0)
                D[i] = dot(A[2,:], C0)
                E[i] = dot(A[3,:], C0)
                F[i] = dot(A[4,:], C0)
                G[i] = dot(A[5,:], C0)
                
                
            SchemeTwoProduct.append(G[-1])

    SchemeTwoDct = collections.OrderedDict()
    
    x = 0
    for Number in NTPConcMismatch:
        SchemeTwoDct['%s' % Number] = [SchemeTwoProduct[x:x+len(TimePtsMismatch)]]
        x = x + len(TimePtsMismatch)

    kpolTwo, kdTwo = Fitting(SchemeTwoDct, TimePtsMismatch, NTPConcMismatch, .99, .5, .5, 700)
    return kpolTwo, kdTwo

def simulation_routine(params):

    # Unpack params/errors
    kf1, kr1, kf2, kr2, kf3, kf4 = params

    # Run the Simulation
    kpol, kd = SimulateSchemeTwo(kf1, kr1, kf1, kr2, kf3, kf4)
    fobs = (kpol / kd) / (kpol_correct / kd_correct)
    return fobs
    

# Store all the fobs here
fobs = []

######################################
# Propagating error by drawing
# parameters from normal distribution
# given by associated parameter error
######################################

# Loop over number of MC iterations
kpol_correct, kd_correct = SimulateSchemeOne()
for iteration in range(MC_num):
    ############################
    # Randomly draw from
    # normal distribution
    # param = mu, error = sigma
    ############################

    # Draw population value within normal distribution
    # given by RD fitted error
    # Eg. if original value is 0.002, random draw
    # from normal distribution where mu=0.002
    # and sigma = 0.00015 could be something like
    # 0.0021567 as that's within the range of the error
    # for that parameter
    #new_es1_pop = np.random.normal(loc=es1_pop, scale=es1_err)

    # Repeat draw for other parameters
    new_kf1 = np.random.normal(loc=kf1, scale=kf1_err)
    new_kr1 = np.random.normal(loc=kr1, scale=kr1_err)
    new_kf2 = np.random.normal(loc=kf2, scale=kf2_err)
    new_kr2 = np.random.normal(loc=kr2, scale=kr2_err)
    new_kf3 = np.random.normal(loc=kf3, scale=kf3_err)
    new_kf4 = np.random.normal(loc=kf4, scale=kf4_err)

    # Now feed these randomly drawn permutations of the parameters
    # to your target function (i.e. your kinetic sim) and get the
    # distribution of fobs (or whatever) values. From this you
    # can calculate the SD/SEM/etc of fobs using the error in 
    # all the dependent parameters
    fobs.append(simulation_routine(params=[new_kf1, new_kr1, new_kf2, new_kf2, new_kf3, new_kf4]))
    print "MC Error Iteration:", iteration
######################################
# Calcluate error and mean of fobs
# Assumes normally distributed error
# in inputs and outputs
######################################
fobs = np.asarray(fobs)
mu_fobs, sigma_fobs = fobs.mean(), fobs.std()
print("Mean of Fobs:", mu_fobs)
print("Std.dev of Fobs:", sigma_fobs)

# Plot distribution of calculated fobs
# - used this code: https://stackoverflow.com/questions/7805552/fitting-a-histogram-with-python
fig, ax = plt.subplots(dpi=120)
n, bins, patches = plt.hist(fobs, 60, normed=1, facecolor='skyblue', alpha=0.75)
y = mlab.normpdf(bins, mu_fobs, sigma_fobs)
l = ax.plot(bins, y, 'r-', linewidth=2)

# Set labels
ax.set_xlabel(r'$F_{obs}$', fontsize=16)
ax.set_ylabel("Normalized Counts", fontsize=16)
ax.set_title(r"$F_{obs}\,|\,\mu=%0.6f\,|\,\sigma=%0.6f$" % (mu_fobs, sigma_fobs), fontsize=14)
plt.tight_layout()
plt.show()
