import ehmetric as eh
import pandas as pd

# load data
df1 = pd.read_csv('dtlz2_3_rnsga2.csv', delimiter=';')
df2 = pd.read_csv('dtlz2_3_pbea.csv', delimiter=';')

df1 = df1[df1['nds_rank'] == 0][['y0','y1','y2']]
df2 = df2[df2['nds_rank'] == 0][['y0','y1','y2']]

# normalize data
_max = pd.concat([df1,df2]).max()
_min = pd.concat([df1,df2]).min()

df1 = (df1-_min)/(_max-_min)
df2 = (df2-_min)/(_max-_min)

dfs = [df1, df2]

# normalize reference point
refPoint = [0.217, 0.485, 0.752]
refPoint = (refPoint-_min)/(_max-_min)

minimization = [True]*3

# run eh-metric
areas, steps = eh.RunEHMetric(dfs, refPoint, minimization)

print(areas)

