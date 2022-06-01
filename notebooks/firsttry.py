import numpy as np
import pandas as pd
import math


import random
import scipy

from desdeo_tools.utilities.pmod import get_pmod
from desdeo_tools.utilities.pmda import get_pmda
import ehmetric as eh
from desdeo_tools.utilities.rmetric import RMetric



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from desdeo_problem import variable_builder, MOProblem, VectorObjective
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from sklearn.preprocessing import Normalizer
from desdeo_emo.EAs import NSGAIII, RVEA, IOPIS_RVEA, IOPIS_NSGAIII, IBEA

from desdeo_emo.utilities import animate_init_, animate_next_

refpoints_dtlz2_l = [[0.10, 0.10, 0.94],[0.10, 0.94, 0.10],  [0.94, 0.10, 0.10], [0.7,0.6,0.55] ]
refpoints_dtlz2_d = [[0.20, 0.20, 0.85],[0.15, 0.25, 0.83],  [0.18, 0.22, 0.80], [0.23,0.25,0.75] ]

refpoints_dtlz7 = [[0.11, 0.10, 5.4 ],[0.70, 0.14, 4.50],  [0.76, 0.76, 3.5], [0.14, 0.70, 4.5] ]

refpoints = refpoints_dtlz2_d
minimization = [True]*3 # for eh-metric
rvea_solutions = []
nsga_solutions = []
RP = []
n_interactions = 4
n_obj = 3
n_var = 8
name = "DTLZ2"
problem = test_problem_builder(
    name=name, n_of_objectives=n_obj, n_of_variables=n_var
    )
evolver_rvea = RVEA(problem, n_gen_per_iter=40, n_iterations=1, interact=True, total_function_evaluations= 60000 )#total_function_evaluations= 9000000
evolver_nsga = NSGAIII(problem, n_gen_per_iter=40, n_iterations=1, interact=True, total_function_evaluations= 60000) #total_function_evaluations= 600000
#evolver_nsga.translation_param = 0.3

#evolver_nsga = IOPIS_RVEA(problem, n_gen_per_iter=500, n_iterations=1)
problem.ideal = np.asarray([0] * n_obj)
problem.ideal_fitness = problem.ideal
problem.nadir = abs(np.random.normal(size=n_obj, scale=0.15)) + 1

non_dominated = evolver_rvea.population.non_dominated_fitness()

for i in range(n_interactions):
    pref_rvea, plot_rvea = evolver_rvea.requests()
    pref_rvea[2].response = pd.DataFrame(
    [refpoints[i]], columns=pref_rvea[2].content["dimensions_data"].columns
    )
    pref_nsga, plot_nsga = evolver_nsga.requests()
    pref_nsga[2].response = pd.DataFrame(
    [refpoints[i]], columns=pref_nsga[2].content["dimensions_data"].columns
    )
    print(f"Running iteration {evolver_rvea._iteration_counter+1}")
    evolver_rvea.iterate(pref_rvea[2])
    evolver_nsga.iterate(pref_nsga[2]) #

    non_dominated_rvea = evolver_rvea.population.non_dominated_fitness()
    rvea_results = evolver_rvea.population.objectives[non_dominated_rvea]
    rvea_solutions.append(rvea_results)

    non_dominated_nsga = evolver_nsga.population.non_dominated_fitness()
    nsga_results = evolver_nsga.population.objectives[non_dominated_nsga]
    nsga_solutions.append(nsga_results)
    
    