## Import Package Dependences ##
import sys
from numpy import *
from openopt import *
import matplotlib.pyplot as plt
import csv
import collections
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import spline
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.mlab as mlab
random.seed(1)


#==============================
## Generate input csv file with column headings
# > python sim.py -setup

setup_head = ('index', 'kt (s-1)', 'kt error', 'k_t (s-1)', 'k_t error', 'ki (s-1)', 'ki error', 'k_i (s-1)', 'k_i error', 'kta (s-1)', 'kta error', 'kat (s-1)', 'kat error')

if str(sys.argv[1]) == "-setup":
	with open('batch_input.csv', 'wb') as s:
		writer = csv.writer(s)
		writer.writerow(setup_head)
		exit()

## PDF Plot Outputs ##
pp = PdfPages('kobs_plots.pdf')
pf = PdfPages('kpol_plots.pdf')
pg = PdfPages('Fpol_Histogram.pdf')
ph = PdfPages('kpol_Histogram.pdf')
pi = PdfPages('Kd_Histogram.pdf')
pj = PdfPages('kobs_Histogram.pdf')

## Lists for writing out batch output results ##
fobs_out = []
fobs_out_err = []
kpol_out = []
kpol_out_err =[]
kd_out = []
kd_out_err = []
kobs_out = []
kobs_out_err = []

# Number of MonteCarlo iterations
MC_num = int(sys.argv[2])

# Scheme 1: Correct incorporation of dCTP-dG base pair
# Scheme 2: Incorrect incorporation of dTTP-dG base pair

#Simulation time points and dNTP concentrations
TimePtsCorrect = [.001, .005, .01, .05, .1, .2, .3, .5, 1]
TimePtsMismatch = [1, 2, 3, 4, 5, 6, 7, 10, 15, 30, 60]

NTPConcCorrect = [0.625, 1.25, 2.5, 5, 10, 20, 40, 80, 200]
NTPConcMismatch = [50, 100, 200, 300, 500, 750, 1000, 1500]

##=====================##
# Fitting Equations
##====================##
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
def Fitting(SchemeDict, TimeList, NTPlist, ProdAmblitudeFitGuess, kObsGuess, kPolGuess, KdGuess, index, iteration):
	if iteration == 0:
		x = 0
		ListOfkObs = []
		for key in SchemeDict.keys():

			temp_list = list(SchemeDict.values()[x]) # Fetch n or n+x list from Dict of 'product populations'
			ProdValues = [val for sublist in temp_list for val in sublist] # Flatten list of list to single list
			
			x = x+1 # Add one to access next key in next cycle
			data1 = column_stack(TimeList) # Data formatting
			data2 = column_stack(ProdValues) # Data formatting
			a,R = expfit(data1, data2, [ProdAmblitudeFitGuess, kObsGuess]) # Fitting for kobs
			ListOfkObs.append(R) #Append kobs to list of kobs for kpol fitting

			plt.plot(data1, data2, 'ko')
			if TimeList == TimePtsCorrect:
				test_time = linspace(0,1,1000)
				test_result1 = [(a*(1-exp(value*-R))) for value in test_time]
				plt.plot(test_time, test_result1, color = 'C0')
				plt.title("Correct incorporation")
			else:
				test_time = linspace(0,60,1000)
				test_result1 = [(a*(1-exp(value*-R))) for value in test_time]
				plt.plot(test_time, test_result1, color = 'C0')
				plt.title("Incorrect Incorporation - Index %s" % index)
		# Final plot for all [dNTP]
		plt.ylabel('Product', fontsize = 14)
		plt.xlabel('time (s)', fontsize = 14)
		plt.tight_layout()
		plt.savefig(pp, format = 'pdf')
		plt.clf()

		# Fitting for kpol from kobs values
		
		data1 = column_stack(NTPlist) # Data Handling
		data2 = column_stack(ListOfkObs) 
		# Fitting for kpol (k = Kd; r = kpol)
		k,r = polfit(data1, data2, [kPolGuess, KdGuess])
		# Plotting	
		plt.plot(data1, data2, 'ko')
		if TimeList == TimePtsCorrect:
			test_ntp = linspace(0,200,1000)
			test_result = [(r*x)/(k+x) for x in test_ntp]
			plt.title("Correct incorporation")
		else:
			test_ntp = linspace(0,1500,1000)
			test_result = [(r*x)/(k+x) for x in test_ntp]
			plt.title("Incorrect Incorporation - Index %s" % index)
		plt.plot(test_ntp, test_result)
		plt.xlabel('dNTP Conc. (uM)', fontsize=14)
		plt.ylabel('kobs (s-$^1)$', fontsize = 14)
		plt.tight_layout()
		plt.savefig(pf, format = 'pdf')
		plt.clf()
		return r, k
	
	else: 
		x = 0
		ListOfkObs = []
		for key in SchemeDict.keys():
		
			temp_list = list(SchemeDict.values()[x]) # Fetch n or n+x list from Dict of 'product populations'
			ProdValues = [val for sublist in temp_list for val in sublist] # Flatten list of list to single list

			x = x+1 # Add one to access next key
			data1 = column_stack(TimeList)
			data2 = column_stack(ProdValues)
			a,R = expfit(data1, data2, [ProdAmblitudeFitGuess, kObsGuess]) #Fit for kobs
			ListOfkObs.append(R) #Append R (kobs) to list of kobs for use in kpol fit

		## Fitting for kpol from kobs values ##
		data1 = column_stack(NTPlist)
		data2 = column_stack(ListOfkObs)
		k,r = polfit(data1, data2, [kPolGuess, KdGuess])
		return r, k

def ErrorAnalysis(parameter, input_list, fileoutput, listoutput1, listoutput2):
	name = asarray(input_list)
	del input_list[:]
	mu, sigma = name.mean(), name.std()
	print("Mean of %s" % str(parameter), mu)
	print("Std Dev of %s" % str(parameter), sigma)
	fig, ax = plt.subplots(dpi=120)
	n, bins, patches = plt.hist(name, 60, normed=1, facecolor='skyblue', alpha=0.75)
	y = mlab.normpdf(bins, mu, sigma)
	l = ax.plot(bins, y, 'r-', linewidth=2)
	ax.set_xlabel(str(parameter), fontsize=16)
	ax.set_ylabel("Normalized Counts", fontsize=16)
	ax.set_title(r"$%s\,|\,\mu=%0.6f\,|\,\sigma=%0.6f$" % (parameter, mu, sigma), fontsize=14)
	plt.tight_layout()
	plt.savefig(fileoutput, format = 'pdf')
	plt.clf()
	listoutput1.append(mu)
	listoutput2.append(sigma)
	return mu, sigma
#===================
# Kinetic Simulations
# Correct and Incorrect Simulations share the same set of rate constants, save for the 
# inclusion of tautomerization/ionization for incorrect incorporations.

# Shared rate constants are declared here; dNTP on and off rate are declared in each scheme
if sys.argv[3] == 'E':
	k_1c = 1900
	k_1i = 70000
	k2 = 268 #forward conformational change rate constant
	k_2 = 100 #reverse conformational change rate constant
	k3 = 9000 #forward chemisty rate constant
	k_3 = .004 #reverse chemisty rate constant
	fitc_guess = 268
elif sys.argv[3] == 'B':
	k_1c = 1000
	k_1i = 65000
	k2 = 1365
	k_2 = 11.9
	k3 = 6.4
	k_3 = .001
	fitc_guess = 6
elif sys.argv[3] == 'T7':
	k_1c = 1000
	k_1i = 70000
	k2 = 660
	k_2 = 1.6
	k3 = 360
	k_3 = .001
	fitc_guess = 200
else:
	print 'Define Polymerase Model'
	exit()

k_2t = k_2
k2t = k2 
k2i = k2

#===================
#Run Kinetic Sheme #1 
def SimulateSchemeOne():
	SchemeOneProduct = []
	for Conc in NTPConcCorrect:
		# Defining additioanl rate constants and starting populations
		C0 = array([1.0, 0.0, 0.0, 0.0]) #Simulation starts with 100% population as E-DNA. 
		k1 = Conc * 100  # dNTP on rate

		# Rate Matrix
		K = zeros((4,4))
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
	
	kpolOne, kdOne = Fitting(SchemeOneDct, TimePtsCorrect, NTPConcCorrect, .99, 5, fitc_guess, k_1c / 100, 0, 0)
	return kpolOne, kdOne
	
def SimulateSchemeTwo(kt, k_t, ki, k_i, kti, kit, k_2i, index, iteration):
	SchemeTwoProduct = []
	for Conc in NTPConcMismatch:
		C0 = array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
		k1 = Conc * 100 # dNTP on rate	

		K = zeros((6, 6))
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

	kpolTwo, kdTwo = Fitting(SchemeTwoDct, TimePtsMismatch, NTPConcMismatch, .9, .5, .5, k_1i / 100, index, iteration)
	return kpolTwo, kdTwo

def simulation_routine(params):

    # Unpack params/errors
    kt, k_t, ki, k_i, kat, kta, k_2i, index, iteration = params

    # Run the Simulation
    kpol, kd = SimulateSchemeTwo(kt, k_t, ki, k_i, kat, kta, k_2i, index, iteration)
    fobs = (kpol / kd) / (kpol_correct / kd_correct)
    kobs = (kpol * 100) / (kd + 100)
    kpol_list.append(kpol)
    kd_list.append(kd)
    kobs_list.append(kobs)
    print "kpol:", format(kpol, '.3f'), "kobs[100 uM]:", format(kobs, '.3f'), "Kd:", format(kd, '.0f'), "Fpol:", fobs
    return fobs

######################################
# Propagating error by drawing
# parameters from normal distribution
# given by associated parameter error
######################################
kpol_correct, kd_correct = SimulateSchemeOne()
print "kpol:", format(kpol_correct, '.2f'), "Kd:", format(kd_correct, '.2f')

RateConstants = pd.read_csv(str(sys.argv[1]))
RateConstants.columns = ['index', 'kt', 'kt_err', 'k_t', 'k_t_err', 'ki', 'ki_err', 
							'k_i', 'k_i_err', 'kta', 'kta_err', 'kat', 'kat_err']

sim_num = len(list(enumerate(RateConstants.index, 1)))
sim_count = 1
for value in RateConstants.index:
	print "Simulation: %s / %s" % (sim_count, sim_num)
	kt, kt_err = RateConstants.kt[value], RateConstants.kt_err[value]
	k_t, k_t_err = RateConstants.k_t[value], RateConstants.k_t_err[value]
	ki, ki_err = RateConstants.ki[value], RateConstants.ki_err[value]
	k_i, k_i_err = RateConstants.k_i[value], RateConstants.k_i_err[value]
	kat, kat_err = RateConstants.kat[value], RateConstants.kat_err[value]
	kta, kta_err = RateConstants.kta[value], RateConstants.kta_err[value]
	if ki == 0:
		k_2i = 0
	else:
		k_2i = k_2

# Loop over number of MC iterations
	fobs_list = []
	kpol_list = []
	kd_list = []
	kobs_list = []

	for iteration in range(MC_num):
		new_kt = random.normal(loc=kt, scale=kt_err)
		new_k_t = random.normal(loc=k_t, scale=k_t_err)
		new_ki = random.normal(loc=ki, scale=ki_err)
		new_k_i = random.normal(loc=k_i, scale=k_i_err)
		new_kat = random.normal(loc=kat, scale=kat_err)
		new_kta = random.normal(loc=kta, scale=kta_err)

    	# Now feed these randomly drawn permutations of the parameters to target function
		fobs_list.append(simulation_routine(params=[new_kt, new_k_t, new_ki, new_k_i, new_kat, new_kta, k_2i, sim_count, iteration]))
		print "MC Error Iteration: %s / %s" % (iteration+1, MC_num)
	sim_count += 1

	ErrorAnalysis("Fobs", fobs_list, pg, fobs_out, fobs_out_err)
	ErrorAnalysis("kpol", kpol_list, ph, kpol_out, kpol_out_err)
	ErrorAnalysis("Kd", kd_list, pi, kd_out, kd_out_err)
	ErrorAnalysis("kobs", kobs_list, pj, kobs_out, kobs_out_err)
			
## Write Out Final Results ##
Master = zip(fobs_out, fobs_out_err, kpol_out, kpol_out_err, kd_out, kd_out_err, kobs_out, kobs_out_err)
heading = ('Fobs (mean)', 'Fobs (Std. Dev.)', 'kpol (mean)', 'kpol (Std.Dev)', 'Kd (mean)', 'Kd (Std. Dev', 'kobs @ 100uM dNTP', 'kobs_err')
error_info = ('Number of MC iteration', '%s' % MC_num)
with open('output.csv', 'wb') as f:
	writer = csv.writer(f)
	writer.writerow(error_info)
	writer.writerow(heading)
	writer.writerows(Master)

#=========================

pp.close()
pf.close()
pg.close()
ph.close()
pi.close()
pj.close()
