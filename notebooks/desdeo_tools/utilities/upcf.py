import pandas as pd
import numpy as np
from pymoo.factory import get_performance_indicator


def getUPCF(solutionsets, refPoint, minimization, r):
    sets = PreScreening(solutionsets, minimization)

    solutions_in_preferred_region = SolutionsInPrefferedRegion(sets, refPoint, r)
    hvs = GetHV(solutions_in_preferred_region)

    return hvs


def PreScreening(solutionsets, minimization):
    allSolutions = []
    for set in solutionsets:
        for index, solution in set.iterrows():
            allSolutions.append(solution.values)


    keep = [True]*len(allSolutions)

    for p in range(len(allSolutions)):
        if not keep[p]:
            continue

        for q in range(p+1, len(allSolutions)):
            if not keep[q]: 
                continue
            
            dom = NDS_Domination(allSolutions[p], allSolutions[q], minimization)
            if dom == -1:
                keep[q] = False
                continue
            if dom == 1:
                keep[p] = False
                break

    retSolutionSets = []
    i = 0
    for set in solutionsets:
        newSet = pd.DataFrame()
        for index, solution in set.iterrows():
            if keep[i]:
                newSet = newSet.append(solution)
            i = i+1
        retSolutionSets.append(newSet.values)

    return retSolutionSets

def SolutionsInPrefferedRegion(solutionsets, refPoint, r):
  
    d_to_midpoint=[]
    lenght_sets = []
    distances = []
    
    counter = 0
    for set in solutionsets:
        lenght_sets.append(len(set))
        for s in set:
            distances.append(np.linalg.norm(s - refPoint.reshape(-1,1)))
    mid_point = np.argmin(distances)
    for set in solutionsets:
        for s in set:
            #d_to_midpoint.append(np.linalg.norm(s- mid_point))
            if np.linalg.norm(s- mid_point) < r:
                pass
                
            else:
                s = None
    counter = 0
    res = []
    new_sets = [0] * len(solutionsets)
    for set in solutionsets:

        if set.ndim == 1:
            res.append([i for i in set if i is not None])
        else: 
            for s in set:
                res.append([i for i in s if i is not None])
        new_sets[counter] = res
        res = []
        counter += 1

    return new_sets

def GetHV(solutionsets):
    
    hv = []
    for s in solutionsets:
        if s == []:
            pass
        else:
            all_s = np.vstack(s)
    nadir_point = np.max(all_s, axis=0)
    for set in solutionsets:
        if set == []:
            hv.append(0)
        else:
            pymo_hv = get_performance_indicator("hv", ref_point=nadir_point)
            hv.append(pymo_hv.do(np.asarray(set)))

    return hv


# returns:
# -1 if p > q
# 1 if q > p
# 0 otherwise
def NDS_Domination(a, b, minimization):
    t = int(0)
    for i in range(len(minimization)):
        if minimization[i]:
            if a[i] < b[i]:
                t = t|1
            elif a[i] > b[i]:
                t = t|2
        else:
            if a[i] > b[i]:
                t = t|1
            elif a[i] < b[i]:
                t = t|2
        if t == 3:
            break

    if t == 1:
        return -1
    if t == 2:
        return 1

    return 0


