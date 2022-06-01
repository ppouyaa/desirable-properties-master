import numpy as np
import pandas as pd

import baseADM
from baseADM import *
import generatePreference as gp

from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors

from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII

from pymoo.factory import get_problem, get_reference_directions
import rmetric as rm
from sklearn.preprocessing import Normalizer
#from pymoo.configuration import Configuration

import rp_hypervolume
import xlsxwriter

#Configuration.show_compile_hint = False

#problem_names = [ "DTLZ2",  "DTLZ4","DTLZ1","DTLZ3"]
problem_names = ["DTLZ1", "DTLZ2", "DTLZ3","DTLZ4"]
#problem_names = ["DTLZ5","DTLZ1", "DTLZ2", "DTLZ3","DTLZ4","DTLZ6","DTLZ7",]
#n_objs = np.asarray([3, 4, 5, 6, 7, 8, 9])  # number of objectives
n_objs = np.asarray([3, 4, 7])
#n_objs = np.asarray([3,4,7,9])
K = 9
n_vars = K + n_objs - 1  # number of variables

num_gen_per_iter = [50]  # number of generations per iteration

algorithms = ["iRVEA", "iNSGAIII"]  # algorithms to be compared

# the followings are for formatting results
column_names = (
    ["problem", "num_obj", "iteration", "num_gens", "reference_point", "Nadir"]
    + [algorithm + "EH-metric" for algorithm in algorithms]
    + [algorithm + "_R_HV" for algorithm in algorithms]
    + [algorithm + "PMOD" for algorithm in algorithms]
    + [algorithm + "PMDA" for algorithm in algorithms]
    +[algorithm + "CCF" for algorithm in algorithms]
    +[algorithm + "HV-U" for algorithm in algorithms]
)
excess_columns = [
    "_EH-metric",
    "_R_HV",
    "_PMOD",
    "_PMDA",
    "_CCF",
    "_HV-U"
]
data = pd.DataFrame(columns=column_names)
data_row = pd.DataFrame(columns=column_names, index=[1])

# ADM parameters
L = 4  # number of iterations for the learning phase
D = 4  # number of iterations for the decision phase
lattice_resolution = 7  # density variable for creating reference vectors


p = rp_hypervolume.target_hypervolume()


counter = 1
total_count = len(num_gen_per_iter) * len(n_objs) * len(problem_names)
for gen in num_gen_per_iter:
    for n_obj, n_var in zip(n_objs, n_vars):
        for problem_name in problem_names:
        

            name = problem_name+str(n_obj)
            # writerRVEA = pd.ExcelWriter("RVEA"+name+".xlsx", engine='xlsxwriter')
            # writerRVEA.save()
            # writerRVEA = pd.ExcelWriter("NSGA"+name+".xlsx", engine='xlsxwriter')
            # writerRVEA.save()
            
            print(f"Loop {counter} of {total_count}")
            counter += 1
            problem = test_problem_builder(
                name=problem_name, n_of_objectives=n_obj, n_of_variables=n_var
            )

            problem.ideal = np.asarray([0] * n_obj)
            problem.nadir = abs(np.random.normal(size=n_obj, scale=0.15)) + 1

            true_nadir = np.asarray([1] * n_obj)

            # interactive
            int_rvea = RVEA(problem, interact=True, n_gen_per_iter=gen, n_iterations= 1,  )
            int_nsga = NSGAIII(problem, interact=True, n_gen_per_iter=gen, n_iterations= 1,)

            # initial reference point is specified randomly
            response = np.random.rand(n_obj)

            # run algorithms once with the randomly generated reference point
            a, pref_int_rvea = int_rvea.requests()
            b, pref_int_nsga = int_nsga.requests()
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
            _, pref_int_rvea = int_rvea.iterate(pref_int_rvea[2])
            _, pref_int_nsga = int_nsga.iterate(pref_int_nsga[2])

            # build initial composite front
            cf = generate_composite_front(
                int_rvea.population.objectives, int_nsga.population.objectives
            )

            # the following two lines for getting pareto front by using pymoo framework
            problemR = get_problem(problem_name.lower(), n_var, n_obj)
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=20)
            pareto_front = problemR.pareto_front(ref_dirs)

            # creates uniformly distributed reference vectors
            reference_vectors = ReferenceVectors(lattice_resolution, n_obj)

            # learning phase
            for i in range(L):
                data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
                    problem_name,
                    n_obj,
                    i + 1,
                    gen,
                ]

                # After this class call, solutions inside the composite front are assigned to reference vectors
                base = baseADM(cf, reference_vectors)
                # generates the next reference point for the next iteration in the learning phase
                response = gp.generateRP4learning(base)
                print(response)
                data_row["reference_point"] = [response]

                # run algorithms with the new reference point
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
                _, pref_int_rvea = int_rvea.iterate(pref_int_rvea[2])
                _, pref_int_nsga = int_nsga.iterate(pref_int_nsga[2])


                # extend composite front with newly obtained solutions
                cf = generate_composite_front(
                    cf, int_rvea.population.objectives, int_nsga.population.objectives
                )

                # R-metric calculation
                ref_point = response.reshape(1, n_obj)

                # normalize reference point
                rp_transformer = Normalizer().fit(ref_point)
                norm_rp = rp_transformer.transform(ref_point)

                rmetric = rm.RMetric(problemR, norm_rp, pf=pareto_front)

                # normalize solutions before sending r-metric
                rvea_transformer = Normalizer().fit(int_rvea.population.objectives)
                norm_rvea = rvea_transformer.transform(int_rvea.population.objectives)

                nsga_transformer = Normalizer().fit(int_nsga.population.objectives)
                norm_nsga = nsga_transformer.transform(int_nsga.population.objectives)

                # R-metric calls for R_IGD and R_HV
                rigd_irvea, rhv_irvea = rmetric.calc(norm_rvea, others=norm_nsga)
                rigd_insga, rhv_insga = rmetric.calc(norm_nsga, others=norm_rvea)
                results = p.calculate_hv_rp([norm_rvea,norm_nsga], response, [2]*n_obj)
                np.savetxt(int_rvea.name_alg+problem_name+str(n_obj)+"it"+str(i+1)+".csv", results[6][0], delimiter = ",")
                np.savetxt(int_nsga.name_alg+problem_name+str(n_obj)+"it"+str(i+1)+".csv", results[6][1], delimiter = ",")
                data_row["Nadir"] = [results[7]]
                #results = p.calculate_hv_rp([norm_rvea,norm_nsga], response, 1)
                # positive_hv = results[2][0]
                # negative_hv = results[3][0]
                # total_hv = results[0][0]
                # original_hv = results[4][0]
                # roi_hv = results[5]


                data_row[["iRVEA" + excess_col for excess_col in excess_columns]] = [
                    rigd_irvea,
                    rhv_irvea,
                    results[4][0],
                    results[2][0],
                    results[3][0],
                    results[0][0],
                    results[5],
                ]
                data_row[["iNSGAIII" + excess_col for excess_col in excess_columns]] = [
                    rigd_insga,
                    rhv_insga,
                    results[4][1],
                    results[2][1],
                    results[3][1],
                    results[0][1],
                    results[5],
                ]

                data = data.append(data_row, ignore_index=1)
                
            # Decision phase
            # After the learning phase the reference vector which has the maximum number of assigned solutions forms ROI
            max_assigned_vector = gp.get_max_assigned_vector(base.assigned_vectors)

            for i in range(D):
                data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
                    problem_name,
                    n_obj,
                    L + i + 1,
                    gen,
                ]

                # since composite front grows after each iteration this call should be done for each iteration
                base = baseADM(cf, reference_vectors)

                # generates the next reference point for the decision phase
                response = gp.generatePerturbatedRP4decision(base, max_assigned_vector[0])
                print(response)

                data_row["reference_point"] =[response]

                # run algorithms with the new reference point
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
                _, pref_int_rvea = int_rvea.iterate(pref_int_rvea[2])
                _, pref_int_nsga = int_nsga.iterate(pref_int_nsga[2])


                # extend composite front with newly obtained solutions
                cf = generate_composite_front(
                    cf, int_rvea.population.objectives, int_nsga.population.objectives
                )

                # R-metric calculation
                ref_point = response.reshape(1, n_obj)

                rp_transformer = Normalizer().fit(ref_point)
                norm_rp = rp_transformer.transform(ref_point)

                # for decision phase, delta is specified as 0.2
                rmetric = rm.RMetric(problemR, norm_rp, delta=0.2, pf=pareto_front)

                # normalize solutions before sending r-metric
                rvea_transformer = Normalizer().fit(int_rvea.population.objectives)
                norm_rvea = rvea_transformer.transform(int_rvea.population.objectives)

                nsga_transformer = Normalizer().fit(int_nsga.population.objectives)
                norm_nsga = nsga_transformer.transform(int_nsga.population.objectives)

                rigd_irvea, rhv_irvea = rmetric.calc(norm_rvea, others=norm_nsga)
                rigd_insga, rhv_insga = rmetric.calc(norm_nsga, others=norm_rvea)
                results = p.calculate_hv_rp([norm_rvea,norm_nsga], response, [2]*n_obj)
                np.savetxt(int_nsga.name_alg+problem_name+str(n_obj)+"it"+str(L+i+1)+".csv", results[6][0], delimiter = ",")
                np.savetxt(int_rvea.name_alg+problem_name+str(n_obj)+"it"+str(L+i+1)+".csv", results[6][1], delimiter = ",")
                data_row["Nadir"] = [results[7]]

                data_row[["iRVEA" + excess_col for excess_col in excess_columns]] = [
                    rigd_irvea,
                    rhv_irvea,
                    results[4][0],
                    results[2][0],
                    results[3][0],
                    results[0][0],
                    results[5],
                ]
                data_row[["iNSGAIII" + excess_col for excess_col in excess_columns]] = [
                    rigd_insga,
                    rhv_insga,
                    results[4][1],
                    results[2][1],
                    results[3][1],
                    results[0][1],
                    results[5],
                ]


                data = data.append(data_row, ignore_index=1)

            data.to_csv("Final_try2.csv", index=False)
