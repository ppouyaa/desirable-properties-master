import numpy as np
import pandas as pd
import csv

import baseADM
from baseADM import *
import generatePreference as gp

from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors

from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII

from desdeo_tools.utilities.pmod import get_pmod
from desdeo_tools.utilities.pmda import get_pmda
from desdeo_tools.utilities.upcf import getUPCF

import ehmetric as eh
from desdeo_tools.utilities.rmetric import RMetric

from my_nds import fast_non_dominated_sort_indices as nds

#Configuration.show_compile_hint = False

def ccf(*fronts):

    _fronts = np.vstack(fronts)
    cf_index = nds(_fronts)[0][0]
    cf_cff = (cf_index < len(fronts[0])).sum()
    rvea_ccf = (cf_index < len(fronts[0])+ len(fronts[1])).sum() - cf_cff
    nsga_ccf = (cf_index >= len(fronts[0])+ len(fronts[1])).sum()

    return rvea_ccf, nsga_ccf


#problem_names = [ "DTLZ2",  "DTLZ4","DTLZ1","DTLZ3"]
problem_names = ["DTLZ1", "DTLZ2", "DTLZ3","DTLZ4", "DTLZ5", "DTLZ6"]
#problem_names = ["DTLZ5","DTLZ1", "DTLZ2", "DTLZ3","DTLZ4","DTLZ6","DTLZ7",]
#n_objs = np.asarray([3, 4, 5, 6, 7, 8, 9])  # number of objectives
n_objs = np.asarray([3, 5])
#n_objs = np.asarray([3,4,7,9])
K = 9
n_vars = K + n_objs - 1  # number of variables

num_gen_per_iter = [100]  # number of generations per iteration

algorithms = ["iRVEA", "iNSGAIII"]  # algorithms to be compared

solutions_columns = ["problem","num_obj","iteration", "iRVEA_solutions", "iNSGA_solutions"]
# the followings are for formatting results
column_names = (
    ["problem", "num_obj", "iteration", "reference_point",]
    + [algorithm + "_EH-metric" for algorithm in algorithms]
    + [algorithm + "_R_HV" for algorithm in algorithms]
    + [algorithm + "_PMOD" for algorithm in algorithms]
    + [algorithm + "_PMDA" for algorithm in algorithms]
    +[algorithm + "_CCF" for algorithm in algorithms]
    +[algorithm + "_FE" for algorithm in algorithms]
    +[algorithm + "UPCF" for algorithm in algorithms]
)
excess_columns = [
    "_EH-metric",
    "_R_HV",
    "_PMOD",
    "_PMDA",
    "_CCF",
    "_FE",
    "_UPCF"
]
solutions_data = pd.DataFrame(columns=solutions_columns)
solutions_data_row = pd.DataFrame(columns=solutions_columns, index=[1])
data = pd.DataFrame(columns=column_names)
data_row = pd.DataFrame(columns=column_names, index=[1])

# ADM parameters
L = 4  # number of iterations for the learning phase
D = 4  # number of iterations for the decision phase
lattice_resolution = 7  # density variable for creating reference vectors
with open('refpoints.csv', newline='') as f:
    reader = csv.reader(f)
    refpoints = list(reader)
refpoints = [[i for i in item if i != ''] for item in refpoints]
refpoints = [[float(y) for y in refpoint] for refpoint in refpoints]
refpoints =refpoints[0:106]

# p = rp_hypervolume.target_hypervolume()


counter = 1
total_count = len(num_gen_per_iter) * len(n_objs) * len(problem_names)
for j in range(0,10):
    k = 0
    for gen in num_gen_per_iter:
        for n_obj, n_var in zip(n_objs, n_vars):
            
            for problem_name in problem_names:
                
            
                minimization = [True] * n_obj # for EH metric
                name = problem_name+str(n_obj)
                
                print(f"Loop {counter} of {total_count}")
                counter += 1
                problem = test_problem_builder(
                    name=problem_name, n_of_objectives=n_obj, n_of_variables=n_var
                )

                

                true_nadir = np.asarray([1] * n_obj)

                # interactive
                int_rvea = RVEA(problem, interact=True, n_gen_per_iter=gen, n_iterations= 1, total_function_evaluations= 50000 )
                int_nsga = NSGAIII(problem, interact=True, n_gen_per_iter=gen, n_iterations= 1, total_function_evaluations= 50000)

                # initial reference point is specified randomly
                response = np.random.rand(n_obj)
                problem.ideal = np.asarray([0] * n_obj)
                problem.ideal_fitness = np.asarray([0] * n_obj)
                problem.nadir = abs(np.random.normal(size=n_obj, scale=0.15)) + 1
                # run algorithms once with the randomly generated reference point
                pref_int_rvea = int_rvea.requests()[0]
                pref_int_nsga = int_nsga.requests()[0]
                pref_int_rvea[2].response = pd.DataFrame(
                    [response], columns=pref_int_rvea[2].content["dimensions_data"].columns
                )
                pref_int_nsga[2].response = pd.DataFrame(
                    [response], columns=pref_int_nsga[2].content["dimensions_data"].columns
                )
                ref_point = response.reshape(1, n_obj)
                int_rvea.ref_point = ref_point
                int_nsga.ref_point = ref_point
                int_rvea.name_alg = 'RV'
                int_nsga.name_alg = 'NS'
                int_rvea.name_prob = problem_name
                int_nsga.name_prob = problem_name
                pref_int_rvea = int_rvea.iterate(pref_int_rvea[2])[0]
                pref_int_nsga = int_nsga.iterate(pref_int_nsga[2])[0]

                # build initial composite front
                
                rvea_ccf_count = 0
                nsga_ccf_count = 0
                cf = generate_composite_front(
                    int_rvea.population.objectives, int_nsga.population.objectives
                )
                # creates uniformly distributed reference vectors
                for i in range(8):

                    data_row[["problem", "num_obj", "iteration"]] = [
                        problem_name,
                        n_obj,
                        i + 1,
                    ]
                    solutions_data_row[["problem", "num_obj", "iteration"]] = [
                        problem_name,
                        n_obj,
                        i + 1,
                    ]

                    # generates the next reference point for the next iteration in the learning phase
                    response = refpoints[k]
                    k +=1
                    data_row["reference_point"] = [response]
                    problem.ideal = np.asarray([0] * n_obj)
                    problem.ideal_fitness = np.asarray([0] * n_obj)

                    # run algorithms with the new reference point
                    pref_int_rvea, plot_rvea = int_rvea.requests()
                    pref_int_nsga, plot_rvea = int_nsga.requests()

                    pref_int_rvea[2].response = pd.DataFrame(
                        [response], columns=pref_int_rvea[2].content["dimensions_data"].columns
                    )
                    pref_int_nsga[2].response = pd.DataFrame(
                        [response], columns=pref_int_nsga[2].content["dimensions_data"].columns
                    )
                    ref_point = np.asarray(response)
                    int_rvea.ref_point = ref_point.reshape(-1,1)
                    int_nsga.ref_point = ref_point.reshape(-1,1)
                    int_rvea.name_alg = 'RV'
                    int_nsga.name_alg = 'NS'
                    int_rvea.name_prob = problem_name
                    int_nsga.name_prob = problem_name
                    
                    _, pref_int_rvea = int_rvea.iterate(pref_int_rvea[2])
                    _, pref_int_nsga = int_nsga.iterate(pref_int_nsga[2])

                    #getting the solutions:
                    rvea_solutions = int_rvea.end()[1]
                    nsga_solutions = int_nsga.end()[1]

                    # extend composite front with newly obtained solutions
                    rvea_ccf , nsga_ccf = ccf(cf, int_rvea.population.objectives, int_nsga.population.objectives)
                    rvea_ccf_count = rvea_ccf
                    nsga_ccf_count = nsga_ccf
                    cf = generate_composite_front(
                        cf, int_rvea.population.objectives, int_nsga.population.objectives
                    )
                    
                    # R-metric calculation
                    # normalize solutions before sending r-metric
                    #normalization
                    _max = np.max(np.vstack((rvea_solutions,nsga_solutions, ref_point)) , axis= 0)
                    _min = np.min(np.vstack((rvea_solutions,nsga_solutions, ref_point)) , axis= 0)
                    norm_rvea = (rvea_solutions - _min)/ (_max - _min)
                    norm_nsga = (nsga_solutions - _min)/ (_max - _min)
                    norm_RP = (ref_point - _min)/ (_max - _min)
                    areas, steps = eh.RunEHMetric([pd.DataFrame(norm_rvea),\
                         pd.DataFrame(norm_nsga)], norm_RP, minimization)
                    upcf = getUPCF([pd.DataFrame(norm_rvea),\
                         pd.DataFrame(norm_nsga)], norm_RP, minimization, 0.1)
                    #print("this is [RVEA, NSGA] EH-metric:", areas)
                    rmetric_rvea = RMetric(problem, norm_RP, delta = 0.2).calc(F = norm_rvea, others = norm_nsga)
                    rmetric_nsga = RMetric(problem, norm_RP, delta = 0.2).calc(F = norm_nsga, others = norm_rvea)
                    if areas[0] == 0 :
                        rmetric_rvea = [0, 0]
                    rvea_pmod = get_pmod(ref_point, rvea_solutions,0.1,2)
                        
                    nsga_pmod = get_pmod(ref_point, nsga_solutions, 0.1, 2)
                    rvea_pmda = get_pmda(ref_point, rvea_solutions, n_obj, 0.5, 0.01)
                    nsga_pmda = get_pmda(ref_point, nsga_solutions, n_obj, 0.5, 0.01)

                    data_row[["iRVEA" + excess_col for excess_col in excess_columns]] = [
                        areas[0],
                        rmetric_rvea[1],
                        rvea_pmod,
                        rvea_pmda,
                        rvea_ccf_count,
                        int_rvea._function_evaluation_count,
                        upcf[0]
                    ]
                    data_row[["iNSGAIII" + excess_col for excess_col in excess_columns]] = [
                        areas[1],
                        rmetric_nsga[1],
                        nsga_pmod,
                        nsga_pmda,
                        nsga_ccf_count,
                        int_nsga._function_evaluation_count,
                        upcf[1]
                    ]

                    solutions_data_row[["iRVEA_solutions", "iNSGA_solutions"]] = [
                        str(rvea_solutions),
                        str(nsga_solutions),
                    ]

                    data = data.append(data_row, ignore_index=1)
                    solutions_data = solutions_data.append(solutions_data_row, ignore_index=1)

                
                data.to_csv("new_results2_it_"+str(j)+".csv", index=False)
    data = pd.DataFrame(columns=column_names)
    solutions_data.to_csv("solutions2_it_"+str(j)+".csv")
