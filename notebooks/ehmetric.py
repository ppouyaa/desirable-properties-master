import pandas as pd

def RunEHMetric(solutionsets, refPoint, minimization):
    sets = PreScreening(solutionsets, minimization)
    areas, steps = GetAreas(sets, refPoint)

    return areas, steps


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
        retSolutionSets.append(newSet)

    return retSolutionSets

def GetAreas(solutionsets, refPoint):
    areasPerSet = []
    stepsPerSet = []

    for set in solutionsets:
        i = 0   # number of points
        j = 0.0 # number of steps
        area = 0.0
        if len(set) != 0.0:
            iStep = 1.0/len(set)
            stepsByPoints = []
            for index, point in set.iterrows():
                side = GetSideOfContainingCube(point, refPoint)
                stepsByPoints.append((side, point))
            sortedSteps = sorted(stepsByPoints, key=lambda x: x[0])
            for point in sortedSteps:
                i = i +1
                area = area + (i*iStep) * (point[0]-j)
                j = point[0]

        areasPerSet.append(area)
        stepsPerSet.append(j)
    
    maxSteps = max(stepsPerSet)
    for i in range(len(areasPerSet)):
        if areasPerSet[i] != 0.0:
            delta = maxSteps - stepsPerSet[i]
            if delta > 0.0:
                areasPerSet[i] = areasPerSet[i] + delta

    return (areasPerSet, maxSteps)

def GetSideOfContainingCube(point, refPoint):
    biggest = 0.0
    for i in range(len(point)):
        d = abs(point[i] - refPoint[i])
        biggest = max(biggest, d)

    return biggest


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


