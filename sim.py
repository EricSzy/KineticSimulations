# Dependences
import sys
import csv
from numpy import *
from openopt import *
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.mlab as mlab
from matplotlib.backends.backend_pdf import PdfPages

#Random Seed For Error Selection
random.seed(1)

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

## Output PDF plots
pp = PdfPages('kobs_plots.pdf')
pf = PdfPages('kpol_plots.pdf')
pg = PdfPages('Fpol_Histogram.pdf')
ph = PdfPages('kpol_Histogram.pdf')
pi = PdfPages('Kd_Histogram.pdf')
pj = PdfPages('kobs_Histogram.pdf')

## Empty List for holding batch output results
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
## Fitting Equations
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
##=====================##
## Fit Function
##====================##
#Takes in the [product] from the kinetic schemes and calculates
#kpol and Kd as is done with a Pre-Steady-State Kinetic Experiment.
def Fitting(idf, TimeList, NTPlist, ProdAmblitudeFitGuess, 
			kObsGuess, kPolGuess, KdGuess, index, iteration):
	ListOfkObs = []
	if iteration == 0:
		for number in NTPlist:
			data1 = column_stack(TimeList)
			data2 = column_stack(idf["%s" % number].values.tolist())
			a,R = expfit(data1, data2, [ProdAmblitudeFitGuess, kObsGuess])

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
		for number in NTPlist:
			data1 = column_stack(TimeList)
			data2 = column_stack(idf["%s" % number].values.tolist())
			a,R = expfit(data1, data2, [ProdAmblitudeFitGuess, kObsGuess])

			ListOfkObs.append(R) #Append kobs to list of kobs for kpol fitting
		
		data1 = column_stack(NTPlist) # Data Handling
		data2 = column_stack(ListOfkObs) 
		# Fitting for kpol (k = Kd; r = kpol)
		k,r = polfit(data1, data2, [kPolGuess, KdGuess])
		# Plotting	
		plt.plot(data1, data2, 'ko')
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
#Run Kinetic Sheme #1 
def SimulateSchemeOne():
	SchemeOneProduct = []
	df = pd.DataFrame({'TIMEPTS':TimePtsCorrect}).set_index('TIMEPTS')
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
		df['%s' % Conc] = SchemeOneProduct
		del SchemeOneProduct[:]
	
	kpolOne, kdOne = Fitting(df, TimePtsCorrect, NTPConcCorrect, .99, 5, fitc_guess, k_1c / 100, 0, 0)
	return kpolOne, kdOne
	
def SimulateSchemeTwo(kt, k_t, ki, k_i, kti, kit, k_2i, index, iteration):
	SchemeTwoProduct = []
	df2 = pd.DataFrame({'TIMEPTS':TimePtsMismatch}).set_index('TIMEPTS')
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
		df2['%s' % Conc] = SchemeTwoProduct
		del SchemeTwoProduct[:]

	kpolTwo, kdTwo = Fitting(df2, TimePtsMismatch, NTPConcMismatch, .9, .5, .5, k_1i / 100, index, iteration)
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
