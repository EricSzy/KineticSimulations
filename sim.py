#!/usr/bin/python

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

## PDFs for output plots
pp = PdfPages('kobs_plots.pdf')
pf = PdfPages('kpol_plots.pdf')
pg = PdfPages('Fpol_Histogram.pdf')
ph = PdfPages('kpol_Histogram.pdf')
pi = PdfPages('Kd_Histogram.pdf')
pj = PdfPages('kobs_Histogram.pdf')

# Empty lists for holding output results
fobs_mu = []
fobs_sigma = []
kpol_mu = []
kpol_sigma = []
kd_mu = []
kd_sigma = []
kobs_mu = []
kobs_sigma = []

# Number of MonteCarlo iterations
MC_num = int(sys.argv[2])

# Scheme 1: Correct incorporation of dCTP-dG base pair.
# Scheme 2: Incorrect incorporation of dTTP-dG base pair.

# Simulation time points and dNTP concentrations.
TimePtsCorrect = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
TimePtsMismatch = [1, 2, 3, 4, 5, 6, 7, 10, 15, 30, 60]

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
def Fitting(idf, TimeList, NTPlist, index, iteration, p0):
	aGuess, kobsGuess, kpolGuess, kdGuess = p0
	fit_kobs = []
	idf['TIMEPTS'] = idf.index
	if iteration == 0: # Generates kobs and kpol plots for the 1st MC error iteration. 
		plt.clf()
		for number in NTPlist:
			data1 = idf['TIMEPTS'].values.tolist()
			data2 = idf["%s" % number].values.tolist()
			popt, pcov = curve_fit(ExpFit, data1, data2, p0 = [aGuess, kobsGuess], maxfev = 10000)
			a,R = popt[0], popt[1]
			fit_kobs.append(R) 
			plt.plot(data1, data2, 'ko')
			if TimeList == TimePtsCorrect:
				fit_time = np.linspace(0,1,1000)
				fit_result = [(a*(1-np.exp(value*-R))) for value in fit_time]
				plt.plot(fit_time, fit_result, color = 'C0')
				plt.title("Correct incorporation")
			else:
				fit_time = np.linspace(0,60,1000)
				fit_result = [(a*(1-np.exp(value*-R))) for value in fit_time]
				plt.plot(fit_time, fit_result, color = 'C0')
				plt.title("Incorrect Incorporation - Index %s" % index)
		# Final plot for all [dNTP]
		plt.ylabel('Product', fontsize = 14)
		plt.xlabel('time (s)', fontsize = 14)
		plt.ylim(0, 1.1)
		plt.tight_layout()
		plt.savefig(pp, format = 'pdf')
		plt.clf()

		# Fitting for kpol and Kd from kobs values
		data3 = NTPlist 
		data4 = fit_kobs		
		popt, pcov = curve_fit(PolFit, data3, data4, p0 = [kpolGuess, kdGuess], maxfev = 10000)
		k, r = popt[0], popt[1]	
		plt.plot(data3, data4, 'ko')
		if TimeList == TimePtsCorrect:
			fit_ntp = np.linspace(0,200,1000)
			fit_result = [(r*x)/(k+x) for x in fit_ntp]
			plt.title("Correct incorporation")
		else:
			fit_ntp = np.linspace(0,1500,1000)
			fit_result = [(r*x)/(k+x) for x in fit_ntp]
			plt.title("Incorrect Incorporation - Index %s" % index)
		plt.plot(fit_ntp, fit_result)
		plt.xlabel('dNTP concentration (uM)', fontsize=14)
		plt.ylabel('kobs (s$^{-1}$)', fontsize = 14)
		plt.tight_layout()
		plt.savefig(pf, format = 'pdf')
		plt.clf()
		return r, k
	
	else: 
		for number in NTPlist:
			data1 = idf['TIMEPTS'].values.tolist()
			data2 = idf["%s" % number].values.tolist()
			popt, pcov = curve_fit(ExpFit, data1, data2, p0 = [aGuess, kobsGuess], maxfev = 10000)
			a,R = popt[0], popt[1]
			fit_kobs.append(R)

		data3 = NTPlist 
		data4 = fit_kobs		
		popt, pcov = curve_fit(PolFit, data3, data4, p0 = [kpolGuess, kdGuess], maxfev = 10000)
		k, r = popt[0], popt[1]
		return r, k

# Calculates sigma and mu for given input parameter
def ErrorAnalysis(parameter, input_list, fileoutput, listoutput1, listoutput2, index):
	raw_results = np.asarray(input_list)
	del input_list[:]
	#Outlier Detection Based on Modified Z-score
	results_median = np.median(raw_results)
	adjusted = [np.math.fabs(x - results_median) for x in raw_results]
	median_adjusted = np.median(adjusted)
	z_score = [(0.6745 * x) / (median_adjusted) for x in adjusted]
	trim_outliers = [x for (x,y) in zip(raw_results, z_score) if y < 3.75]
	trimmed = np.asarray(trim_outliers)
	mu_adj, sigma_adj = trimmed.mean(), trimmed.std()
	print("Mean of %s" % str(parameter), mu_adj)
	print("Std Dev of %s" % str(parameter), sigma_adj)
	
	fig, ax = plt.subplots(dpi=120)
	n, bins, patches = plt.hist(raw_results, 60, normed=1, facecolor='skyblue', alpha=0.75)
	x = np.linspace(mu_adj - 4 * sigma_adj, mu_adj + 4 * sigma_adj, 100)
	plt.plot(x, mlab.normpdf(x, mu_adj, sigma_adj))
	
	ax.set_xlabel(str(parameter), fontsize=16)
	ax.set_ylabel("Normalized Counts", fontsize=16)
	ax.set_title(r"$%s\,|\,\mu=%0.6f\,|\,\sigma=%0.6f$ | index = %s" % (parameter, mu_adj, sigma_adj, index), fontsize=14)
	plt.tight_layout()
	plt.savefig(fileoutput, format = 'pdf')
	listoutput1.append(mu_adj)
	listoutput2.append(sigma_adj)

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
	for value in NTPConcCorrect:
		df["%s" % value] = df['TIMEPTS'].apply(SchemeOne, args = (value,))
	df = df.set_index('TIMEPTS')
	kpolOne, kdOne = Fitting(df, TimePtsCorrect, NTPConcCorrect, 0, 0,  p0 = [.99, 5, fitc_guess, k_1c / 100])
	return kpolOne, kdOne

def RunSchemeTwo(index, iteration, rates):
	
	df2 = pd.DataFrame({'TIMEPTS':TimePtsMismatch})
	for value in NTPConcMismatch:
		df2["%s" % value] = df2['TIMEPTS'].apply(SchemeTwo, args = (value, rates,))
	df2 = df2.set_index('TIMEPTS')
	kpolTwo, kdTwo = Fitting(df2, TimePtsMismatch, NTPConcMismatch, index, iteration,  p0 = [.99, .5, .5, k_1i / 100])
	return kpolTwo, kdTwo

def simulation_routine(index, iteration, params):

    kpol, kd = RunSchemeTwo(index, iteration, params)
    fobs = (kpol / kd) / (kpol_correct / kd_correct)
    kobs = (kpol * 100) / (kd + 100)
    kpol_list.append(kpol)
    kd_list.append(kd)
    kobs_list.append(kobs)
    print "kpol:", format(kpol, '.3f'), "kobs[100 uM]:", format(kobs, '.3f'), "Kd:", format(kd, '.0f'), "Fpol:", fobs
    return fobs


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
	fobs_list = []
	kpol_list = []
	kd_list = []
	kobs_list = []

	# New set of rate constants determined by drawing from a nomral distribution of value and error. 
	for iteration in range(MC_num):
		new_kt = np.random.normal(loc=kt, scale=kt_err)
		new_k_t = np.random.normal(loc=k_t, scale=k_t_err)
		new_ki = np.random.normal(loc=ki, scale=ki_err)
		new_k_i = np.random.normal(loc=k_i, scale=k_i_err)
		new_kat = np.random.normal(loc=kat, scale=kat_err)
		new_kta = np.random.normal(loc=kta, scale=kta_err)

    	# Now feed these randomly drawn permutations of the parameters to simulations
		fobs_list.append(simulation_routine(sim_count, iteration, params=[new_kt, new_k_t, new_ki, new_k_i, new_kat, new_kta, k_2i]))
		print "MC Error Iteration: %s / %s" % (iteration+1, MC_num)
	
	# Calculates sigma and mu from MC error interations for each parameter
	ErrorAnalysis("Fobs", fobs_list, pg, fobs_mu, fobs_sigma, sim_count)
	ErrorAnalysis("kpol", kpol_list, ph, kpol_mu, kpol_sigma, sim_count)
	ErrorAnalysis("Kd", kd_list, pi, kd_mu, kd_sigma, sim_count)
	ErrorAnalysis("kobs", kobs_list, pj, kobs_mu, kobs_sigma, sim_count)
	sim_count += 1

# Write Out Final Results to 'output.csv'
Master = zip(fobs_mu, fobs_sigma, kpol_mu, kpol_sigma, kd_mu, kd_sigma, kobs_mu, kobs_sigma)
error_info = ('Number of MC iteration', '%s' % MC_num)
heading = ('Fobs (mean)', 'Fobs (Std. Dev.)', 'kpol (mean)',
			 'kpol (Std.Dev)', 'Kd (mean)', 'Kd (Std. Dev', 
			 'kobs @ 100uM dNTP', 'kobs_err')

with open('output.csv', 'wb') as f:
	writer = csv.writer(f)
	writer.writerow(error_info)
	writer.writerow(heading)
	writer.writerows(Master)

# Close PDF files with plots
pp.close()
pf.close()
pg.close()
ph.close()
pi.close()
pj.close()
