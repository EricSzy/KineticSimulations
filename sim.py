# Import Dependences
import sys
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

# Random Seed For Error Selection
np.random.seed(1)

#==============================
# Generate input .csv file with column headings
# >python sim.py -setup

setup_head = ('index', 'kt (s-1)', 'kt error',
				'k_t (s-1)', 'k_t error', 'ki (s-1)',
				'ki error', 'k_i (s-1)', 'k_i error',
				'kta (s-1)', 'kta error', 'kat (s-1)',
				'kat error')

if str(sys.argv[1]) == "-setup":
	with open('batch_input.csv', 'wb') as s:
		writer = csv.writer(s)
		writer.writerow(setup_head)
		exit()

# Open PDF for output plots
pp = PdfPages('Fit_Plots.pdf')

# Number of MonteCarlo iterations
MC_num = int(sys.argv[2])

# Create empty lists for holding final output results
fobs_mu, fobs_sigma = [], []
kpol_mu, kpol_sigma = [], []
kd_mu, kd_sigma = [], []

# Scheme 1: Correct incorporation of dCTP-dG base pair.
# Scheme 2: Incorrect incorporation of dTTP-dG base pair.

# Simulation time points (s).
TimePtsCorrect = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
TimePtsMismatch = [1, 2, 3, 4, 5, 6, 7, 10, 15, 30, 60]
# Simulation dNTP concentrations (uM).
NTPConcCorrect = [0.625, 1.25, 2.5, 5, 10, 20, 40, 80, 200]
NTPConcMismatch = [50, 100, 200, 300, 500, 750, 1000, 1500]

# Fitting Equations
def ExpFit(X, a, R):
	# Exponential fit for kobs from [product]
	return a *(1-np.exp(-R*X))

def PolFit(X, k, r):
	# Fit for kpol and Kd from kobs and [dNTP]
	return ((r*X)/(k+X))

# Fitting Functions
def FittingCorrect(idf, TimeList, NTPList, p0):
	# Fitting and plotting for correct incorporation
	a0, kobs0, kpol0, kd0 = p0
	fit_kobs = []
	idf['TIMEPTS'] = idf.index
	fig, ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize = (16, 8))
	timepts = np.linspace(0, max(TimeList), 100)
	ntppts = np.linspace(0, max(NTPList), 100)
	for i in NTPList:
		x = idf['TIMEPTS'].values.tolist()
		y = idf["%s" % i].values.tolist()
		popt, pcov = curve_fit(ExpFit, x, y, p0 = [a0, kobs0], maxfev = 10000)
		a,R = popt[0], popt[1]
		fit_kobs.append(R)
		ax[0].plot(x, y, 'ko')
		ax[0].plot(timepts, [ExpFit(x, popt[0], popt[1]) for x in timepts], color = 'C0')
	ax[0].set_xlabel('Time (s)', fontsize = 18)
	ax[0].set_ylabel('Fractional Product Formation', fontsize = 18)
	ax[0].set_title('Correct Incorporation', fontsize = 18)

	popt, pcov = curve_fit(PolFit, NTPList, fit_kobs, p0 = [kpol0, kd0], maxfev = 10000)
	k, r = popt[0], popt[1]
	ax[1].plot(NTPList, fit_kobs, 'ko')
	ax[1].plot(ntppts, [PolFit(x, k, r) for x in ntppts], color = 'C0')
	ax[1].set_xlabel('dNTP Concentration (uM)', fontsize = 18)
	ax[1].set_ylabel('k$_{obs}$ s$^{-1}$', fontsize = 18)
	plt.tight_layout()
	plt.savefig(pp, format = 'pdf')
	return r, k

def Fitting(idf, TimeList, NTPList, p0):
	# Fitting for mismatch incorporation
	a0, kobs0, kpol0, kd0 = p0
	fit_kobs = []
	idf['TIMEPTS'] = idf.index
	for i in NTPList:
		x = idf['TIMEPTS'].values.tolist()
		y = idf["%s" % i].values.tolist()
		popt, pcov = curve_fit(ExpFit, x, y, p0 = [a0, kobs0], maxfev = 10000)
		a,R = popt[0], popt[1]
		fit_kobs.append(R)
	if TimeList == TimePtsMismatch:
		kobs_mc_err.append(fit_kobs)
	popt, pcov = curve_fit(PolFit, NTPList, fit_kobs, p0 = [kpol0, kd0], maxfev = 10000)
	k, r = popt[0], popt[1]
	return r, k

def MCErrPlots(RawPtsList, kobsList, NTPList, TimeList, 
				kpolList, kdList, FobsList, index):
	#MC Error analysis and plotting for mismatch incorporation
	RawPts_df = pd.concat(RawPtsList)
	RawAvg = RawPts_df.reset_index(drop=True).groupby('TIMEPTS').mean()
	RawMin = RawPts_df.reset_index(drop=True).groupby('TIMEPTS').apply(min)
	RawMax = RawPts_df.reset_index(drop=True).groupby('TIMEPTS').apply(max)

	kobs_df = pd.DataFrame(kobsList)
	kobsAvg = kobs_df.mean()
	kobsMin = kobs_df.apply(min)
	kobsMax = kobs_df.apply(max)

	timepts = np.linspace(0, max(TimeList), 100)
	ntppts = np.linspace(0, max(NTPList), 100)

	fig, ax = plt.subplots(2, 2, sharex=False, sharey=False, 
							figsize = (16, 16))
	# Set axis labels and titles
	ax[0, 0].set_xlabel('Time (s)', fontsize = 18)
	ax[0, 0].set_ylabel('Fractional Product Formation', fontsize = 18)
	ax[0, 0].set_title('Index (%s)' % index, fontsize = 18)
	
	ax[0, 1].set_xlabel('dNTP Concentration (uM)', fontsize = 18)
	ax[0, 1].set_ylabel('k$_{obs}$ s$^{-1}$', fontsize = 18)
	ax[0, 1].set_title('', fontsize = 18)
	
	ax[1, 0].set_xlabel('F$_{pol}$', fontsize=18)
	ax[1, 0].set_ylabel("Normalized Counts", fontsize=18)
	
	ax[1, 1].set_xlabel('k$_{pol}$ (s$^{-1}$)', fontsize=18)
	ax[1, 1].set_ylabel("Normalized Counts", fontsize=18)
	
	# Plot [0, 0] Product vs. Time
	for i in NTPList:
		x = TimeList
		y = RawAvg['%s' % i].values.tolist()
		popt, pcov = curve_fit(ExpFit, x, y, p0 = [1, .5], maxfev = 10000)
		ax[0, 0].plot(x, y, 'ko')
		ax[0, 0].errorbar(x, y, yerr = [(j-k)/2 for (j, k) in zip(RawMax["%s" % i].values.tolist(), RawMin["%s" % i].values.tolist())],
					color = 'k', linesytle = 'none', fmt = 'none')
		fY = [ExpFit(x, popt[0], popt[1]) for x in timepts]
		ax[0, 0].plot(timepts, fY, alpha = 0.5, color = 'C0', zorder = 0)
	
	# Plot [0, 1] kobs vs. [dNTP]
	ax[0, 1].scatter(NTPList, kobsAvg, color = 'k', zorder = 10)
	ax[0, 1].errorbar(NTPList, kobsAvg, yerr = [(x - y)/2 for (x, y) in zip(kobsMax, kobsMin)], 
				color = 'k', linestyle = 'None', fmt = 'none')
	for (r, k) in zip(kpolList, kdList):
		fY = [PolFit(x, k, r) for x in ntppts]
		ax[0, 1].plot(ntppts, fY, alpha = 0.5, color = 'C0', zorder = 0)
	
	# Plot [1, 0] Histgram of Fpol values
	fpolResults = np.asarray(FobsList)
	del FobsList[:]
	mu, sigma = Outlier(fpolResults)
	fobs_mu.append(mu)
	fobs_sigma.append(sigma)
	n, bins, patches = ax[1, 0].hist(fpolResults, 60, normed=1, facecolor='skyblue', alpha=0.75)
	x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
	ax[1, 0].plot(x, mlab.normpdf(x, mu, sigma))

	# Plot [1, 1] Histgram of kpol values
	kpolResults = np.asarray(kpolList)
	del kpolList[:]
	mu, sigma = Outlier(kpolResults)
	kpol_mu.append(mu)
	kpol_sigma.append(sigma)
	n, bins, patches = ax[1, 1].hist(kpolResults, 60, normed=1, facecolor='skyblue', alpha=0.75)
	x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
	ax[1, 1].plot(x, mlab.normpdf(x, mu, sigma))
	# Format and Save
	plt.tight_layout()
	plt.savefig(pp, format = 'pdf')
	
	# Kd - Not Plotted - Written to output .csv file
	kdResults = np.asarray(kdList)
	del kdList[:]
	mu, sigma = Outlier(kdResults)
	kd_mu.append(mu)
	kd_sigma.append(sigma)

def Outlier(InputList):
	# Removes outliers based on modified z-score
	# for the sigma/mu for histogram fit.
	# All raw data is still plotted in histogram.
	MedianValue = np.median(InputList)
	AdjustedResults = [np.math.fabs(x - MedianValue) for x in InputList]
	AdjustedMedian = np.median(AdjustedResults)
	z_score = [(0.6745 * x) / (AdjustedMedian) for x in AdjustedResults]
	trim_outliers = [x for (x,y) in zip(InputList, z_score) if y < 3.75]
	TrimmedResults = np.asarray(trim_outliers)
	mu_adj, sigma_adj = TrimmedResults.mean(), TrimmedResults.std()
	return mu_adj, sigma_adj

# Kinetic Schemes
# Correct and Incorrect Simulations share the same set of rate constants, 
# except inclusion of tautomerization/ionization rate constants.

# DataFrame for polymerase microscopic rate constants
polymerase_df = pd.DataFrame(
	{"k_1c" : [1900, 1000, 1000],
	 "k_1i" : [70000, 65000, 70000],
	 "k2": [268, 1365, 660],
	 "k_2": [100, 11.9, 1.6],
	 "k3": [9000, 6.4, 360],
	 "k_3": [.004, .001, .001],
	 "fitc_guess" : [268, 6, 200]},
	 index = ['E', 'B', 'T7'])
polymerase_df = polymerase_df[['k_1c', 'k_1i', 'k2', 'k_2', 'k3', 'k_3', 'fitc_guess']]

k_1c, k_1i, k2, k_2, k3, k_3, fitc_guess = polymerase_df.T["%s" % sys.argv[3]].values.tolist()

k_2t = k_2
k2t = k2 
k2i = k2

if str(sys.argv[4]) == 'ES1':
	k2i = 0
	k_2i = 0
elif str(sys.argv[4]) == 'ES2':
	k2t = 0
	k_2t = 0

#===================
# Mathematics for kinetic scheme one (Correct Incorporation)
def SchemeOne(time, conc):
	# Simulation starts with 100% population as E-DNA. 
	C0 = np.array([1.0, 0.0, 0.0, 0.0]) 
	k1 = conc * 100  # dNTP on rate

	# Rate Matrix
	K = np.zeros((4,4))
	K[0, 0] = -k1
	K[0, 1] = k_1c
	K[1, 0] = k1
	K[1, 1] = -k_1c-k2
	K[1, 2] = k_2
	K[2, 1] = k2
	K[2, 2] = -k_2-k3
	K[2, 3] = k_3
	K[3, 2] = k3
	K[3, 3] = -k_3
		
	w,M = np.linalg.eig(K)
	M_1 = np.linalg.inv(M)

	T = np.linspace(0, float(time), 2)
	B = np.zeros(T.shape)
	C = np.zeros(T.shape)
	D = np.zeros(T.shape)
	E = np.zeros(T.shape)

	for i,t in enumerate(T):
		A = np.dot(np.dot(M,np.diag(np.exp(w*t))), M_1)
		B[i] = np.dot(A[0,:], C0)
		C[i] = np.dot(A[1,:], C0)
		D[i] = np.dot(A[2,:], C0)
		E[i] = np.dot(A[3,:], C0)
	return E[-1]

# Mathematics for kinetic scheme two (Incorrect Incorporation)
def SchemeTwo(time, conc, rates):
	kt, k_t, ki, k_i, kti, kit, k_2i = rates
	
	C0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	k1 = conc * 100 # dNTP on rate	

	K = np.zeros((6, 6))
	K[0, 0] = -k1
	K[0, 1] = k_1i
	K[1, 0] = k1
	K[1, 1] = -k_1i-kt-ki
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

	w,M = np.linalg.eig(K)
	M_1 = np.linalg.inv(M)

	T = np.linspace (0, float(time), 2)
	B = np.zeros(T.shape)
	C = np.zeros(T.shape)
	D = np.zeros(T.shape)
	E = np.zeros(T.shape)
	F = np.zeros(T.shape)
	G = np.zeros(T.shape)

	for i,t in enumerate(T):
		A = np.dot(np.dot(M,np.diag(np.exp(w*t))), M_1)
		B[i] = np.dot(A[0,:], C0)
		C[i] = np.dot(A[1,:], C0)
		D[i] = np.dot(A[2,:], C0)
		E[i] = np.dot(A[3,:], C0)
		F[i] = np.dot(A[4,:], C0)
		G[i] = np.dot(A[5,:], C0)
	return G[-1]		

def RunSchemeOne():
	
	df = pd.DataFrame({'TIMEPTS':TimePtsCorrect})
	for i in NTPConcCorrect:
		df["%s" % i] = df['TIMEPTS'].apply(SchemeOne, args = (i,))
	df = df.set_index('TIMEPTS')
	kpolOne, kdOne = FittingCorrect(df, TimePtsCorrect, NTPConcCorrect, p0 = [.99, 5, fitc_guess, k_1c / 100])
	return kpolOne, kdOne

def RunSchemeTwo(rates):
	
	df2 = pd.DataFrame({'TIMEPTS':TimePtsMismatch})
	for i in NTPConcMismatch:
		df2["%s" % i] = df2['TIMEPTS'].apply(SchemeTwo, args = (i, rates,))
	df2 = df2.set_index('TIMEPTS')
	kpolTwo, kdTwo = Fitting(df2, TimePtsMismatch, NTPConcMismatch, p0 = [.99, .5, .5, k_1i / 100])
	return kpolTwo, kdTwo, df2

def simulation_routine(params):

    kpol, kd, raw_pts = RunSchemeTwo(params)
    fobs = (kpol / kd) / (kpol_correct / kd_correct)
    kpol_list.append(kpol)
    kd_list.append(kd)
    fobs_list.append(fobs)
    RawPtsMCErr.append(raw_pts)
    

# Run Simulations with propagating error by drawing parameters from normal distribution

# Run Simulations for Correct Incoporation 
kpol_correct, kd_correct = RunSchemeOne()

print "kpol:", format(kpol_correct, '.2f'), "Kd:", format(kd_correct, '.2f')

# Read in rate constants
RateConstants = pd.read_csv(str(sys.argv[1]))
RateConstants.columns = ['index', 'kt', 'kt_err', 'k_t', 'k_t_err', 'ki', 'ki_err', 
							'k_i', 'k_i_err', 'kta', 'kta_err', 'kat', 'kat_err']

# Counter for how many sets of rate constants are being run
sim_num = len(list(enumerate(RateConstants.index, 1)))
# Counter initalized at 1
sim_count = 1

# Set values and error for set of input rate constants
for value in RateConstants.index:
	print "Simulation: %s / %s" % (sim_count, sim_num)
	kt, kt_err = RateConstants.kt[value], RateConstants.kt_err[value]
	k_t, k_t_err = RateConstants.k_t[value], RateConstants.k_t_err[value]
	ki, ki_err = RateConstants.ki[value], RateConstants.ki_err[value]
	k_i, k_i_err = RateConstants.k_i[value], RateConstants.k_i_err[value]
	kat, kat_err = RateConstants.kat[value], RateConstants.kat_err[value]
	kta, kta_err = RateConstants.kta[value], RateConstants.kta_err[value]
	
	# k_2i and k_2t are assumed equal.
	# This statment sets k_2i to 0 if ES2 in not formed.
	# This prevents backflow to ES2 via product.
	if ki == 0 and kta == 0:
		k_2i = 0
	else:
		k_2i = k_2

	# Empty lists hold results from MC error iterations for one set of rate constants.
	# Lists are input into the ErrorAnalysis function and are cleared before running 
	# the next set of rate constants. 
	kobs_mc_err = []
	fobs_list = []
	kpol_list = []
	kd_list = []
	RawPtsMCErr = []
	# New set of rate constants determined by drawing from a nomral distribution of value and error. 
	
	for iteration in range(MC_num):
		new_kt = np.random.normal(loc=kt, scale=kt_err)
		new_k_t = np.random.normal(loc=k_t, scale=k_t_err)
		new_ki = np.random.normal(loc=ki, scale=ki_err)
		new_k_i = np.random.normal(loc=k_i, scale=k_i_err)
		new_kat = np.random.normal(loc=kat, scale=kat_err)
		new_kta = np.random.normal(loc=kta, scale=kta_err)


    	# Now feed these randomly drawn permutations of the parameters to simulations
		simulation_routine(params=
			[new_kt, new_k_t, new_ki, new_k_i, new_kat, new_kta, k_2i])
		sys.stdout.write("MC Error: %s / %s           \r" % (iteration+1, MC_num))
		sys.stdout.flush()

	MCErrPlots(RawPtsMCErr, kobs_mc_err, NTPConcMismatch, TimePtsMismatch, kpol_list, kd_list, fobs_list, sim_count)
	sim_count += 1

# Write Out Final Results to 'output.csv'
Master = zip(fobs_mu, fobs_sigma, kpol_mu, kpol_sigma, kd_mu, kd_sigma)
error_info = ('Number of MC iterations', '%s' % MC_num, 'Polymerase',
				 '%s' % sys.argv[3], 'Model', '%s' % sys.argv[4])
heading = ('Fpol (mean)', 'Fpol (Std. Dev.)', 'kpol (mean)',
			 'kpol (Std.Dev)', 'Kd (mean)', 'Kd (Std. Dev)')

with open('Fit_Output.csv', 'wb') as f:
	writer = csv.writer(f)
	writer.writerow(error_info)
	writer.writerow(heading)
	writer.writerows(Master)

# Close PDF files with plots
pp.close()
