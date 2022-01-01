# coding: utf-8

# In[1]:


import argparse
import sys

parser = argparse.ArgumentParser(
    prog="Deal optimiser",
    description="Optimization of deals using probability of wining threshold value",
)
parser.add_argument(
    "-t",
    "--threshold",
    default=0.5,
    metavar="theresh_hold",
    type=float,
    help="Threshold value to evaluate Third objective function",
    nargs=1,
)
# parser.add_argument("-f", "--file_name", metavar = "file_name", default="input.xlsx", type= str, help = "File Input",  nargs = '?')
parser.add_argument(
    "-p",
    "--pdgw",
    metavar="pdgw",
    default=10,
    type=int,
    help="Probability of deal goes well",
    nargs="?",
)
parser.add_argument(
    "-n",
    "--deal_size",
    metavar="deal_size",
    default=10,
    type=int,
    help="Deal size",
    nargs="?",
)
parser.add_argument(
    "-c",
    "--cost_coef",
    metavar="cost_coef",
    default=5.0,
    type=float,
    help="Cost coefficient ",
    nargs="?",
)
args = None
THRESHOLD = None
FILE_NAME = None
COST_COEF = None
PDGW_SIZE = None
DEAL_SIZE = None
try:
    args = parser.parse_args()
except Exception as ex:
    # argparse.ArgumentParser.error(message = str(ex))
    print("Exception : {}".format(str(ex)))
    sys.exit(0)
print("Arguments: {}".format(args))
THRESHOLD = args.threshold  # [0]
# FILE_NAME = args.file_name
PDGW_SIZE = (
    args.pdgw if args.pdgw is not None or args.pdgw == 5 or args.pdgw == 10 else 10
)
DEAL_SIZE = (
    args.deal_size
    if args.deal_size is not None or (args.deal_size >= 1 and args.deal_size <= 10)
    else 10
)
COST_COEF = (
    args.cost_coef
    if args.cost_coef is not None or args.cost_coef == 2.5 or args.cost_coef == 5
    else 5
)

if THRESHOLD == None or THRESHOLD > 1 or THRESHOLD < 0:
    print("Invalid Threshold value")
    sys.exit(0)
print(
    "Threshold: {} \nPDGW: {} \nDeal Size: {}\nCost Coefficient: {} ".format(
        THRESHOLD, PDGW_SIZE, DEAL_SIZE, COST_COEF
    )
)


# In[2]:


# THRESHOLD = 0.7
group_size = 1
PDGW = [(i + 1) / (PDGW_SIZE * 1.0) for i in range(PDGW_SIZE)]
# PDGW =  [i/10 for i in range(1,11)]
# PDGW =  [i/10 for i in range(1,3)]


# #### Importing modules

# In[3]:


import numpy as np
import pandas as pd
import random
import scipy
import traceback
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# from matplotlib import colors as mcolors
from collections import OrderedDict

random.seed(1000)
np.random.seed(100)
from pathlib import Path
import os
import math
import matplotlib as mpl
from os import listdir
from os.path import isfile, join
from model import pred

if os.environ.get("DISPLAY", "") == "":
    print("no display found. Using non-interactive Agg backend")
    mpl.use("Agg")
from matplotlib import pyplot as plt

plt.style.use("grayscale")


# In[4]:


X = Y = [math.pow(10, i) for i in range(-4, 6, 2)]
# X = Y = [ 1e4, 1e5]


# In[5]:


selected_deals_size = DEAL_SIZE
candidates_size = DEAL_SIZE
TOL = 1e-15
ATOL = 0


# In[6]:


FOLDER_LABEL = "Threshold- " + str(THRESHOLD)
try:
    os.mkdir(FOLDER_LABEL)
    os.mkdir(FOLDER_LABEL + "/Report")
    os.mkdir(FOLDER_LABEL + "/Report/ConstDec")
    os.mkdir(FOLDER_LABEL + "/Report/ExpDec")
    os.mkdir(FOLDER_LABEL + "/Result figures")
    os.mkdir(FOLDER_LABEL + "/Result figures/First obj single")
    os.mkdir(FOLDER_LABEL + "/Result figures/Second obj single")
    os.mkdir(FOLDER_LABEL + "/Result figures/First obj vs Second obj single")
except Exception as err:
    print("Error has occured: ", str(err))


#  ### Loading Training data

# In[7]:


dataset_file_name = "input.xlsx"
# dataset_file_name = FILE_NAME

fileName = file_name = os.path.abspath(dataset_file_name)
df = pd.read_excel(fileName)

# CLF_MODEL = getClassifier(FILE_NAME)
# CLF_MODEL = model

#  # Objective Functions
#
#  ### First Objective (cost-price)*probability

# In[8]:
def get_preferred_countries(market_penetration=10):
    unique_countries = list(df["Country"].unique())
    # print("Unique countries: {}".format(unique_countries))
    preferred_countries = np.random.choice(
        unique_countries,
        int(np.ceil(market_penetration / 100.0 * len(unique_countries))),
        False,
    )
    return preferred_countries


# print("Preferred countries: {}".format(get_preferred_countries()))


def firstObjective(df, prices, costs, probFun):
    try:
        probs = probFun(df, prices)
        return np.sum(
            [(price - cost) * prob for price, cost, prob in zip(prices, costs, probs)]
        )
    except Exception as ex:
        raise Exception(str(ex))


#  ### Second Objective Probability of winning the bid at a given price


def secondObjective(df, prices, probFun):
    try:
        return np.sum(probFun(df, prices))
    except Exception as ex:
        raise Exception(str(ex))


# ### Third objective Sum of price*Probability_of_wining*Yd


def thirdObjectiveExpDec(df, prices, probFun, PDGW, threshold, x, y, is_group=False):
    try:
        probs = probFun(df, prices)
        res = [
            (max(0, prob - threshold)) * (x * math.exp(-y * pdgw))
            for prob, pdgw in zip(probs, PDGW)
        ]
        if is_group:
            return res
        else:
            return np.sum(res)
    except Exception as ex:
        raise Exception(str(ex))


def thirdObjectiveConstDec(df, prices, probFun, PDGW, threshold, x, y, is_group=False):
    try:
        probs = probFun(df, prices)
        res = [
            (max(0, prob - threshold)) * (x - y * pdgw)
            for prob, pdgw in zip(probs, PDGW)
        ]
        if is_group:
            return res
        else:
            return np.sum(res)
    except Exception as ex:
        raise Exception(str(ex))


# ### Three objectives problem definition


def all_three_objectivesConstDec(df, prices, costs, probFun, PDGW, threshold, x, y):
    try:
        return [
            firstObjective(df, prices, costs, probFun),
            secondObjective(df, prices, probFun),
            thirdObjectiveConstDec(df, prices, probFun, PDGW, threshold, x, y),
        ]
    except Exception as ex:
        raise Exception(str(ex))


def all_three_objectivesExpDec(df, prices, costs, probFun, PDGW, threshold, x, y):
    try:
        return [
            firstObjective(df, prices, costs, probFun),
            secondObjective(df, prices, probFun),
            thirdObjectiveExpDec(df, prices, probFun, PDGW, threshold, x, y),
        ]
    except Exception as ex:
        raise Exception(str(ex))


# ### Generating radom costs for profit margins between 1.15 to 1.25 percents


def randomCosts(df, deal_indexes):
    costs = [
        float(
            "%.2f"
            % np.round(
                df.iloc[deal_indexes[i]]["Price"] / (np.random.random() / 10 + 1.15), 2
            )
        )
        for i in range(len(deal_indexes))
    ]
    return costs


# ### Get candidates original price


def getCandidatesOriginalPrices(df, deal_indexes):
    prices = [
        float(df.iloc[deal_indexes[i]]["Price"]) for i in range(len(deal_indexes))
    ]
    return prices


# ### Method to extract candidate bids based on given parameters


def candidateBids(df, size):
    df_copy = df.copy()
    df_copy = df_copy[df_copy["Outcome"] == 1]
    candidate_list = []
    candidate_df = df_copy
    available_size = min(len(candidate_df), size)
    candidate_indexes = random.sample(list(candidate_df.index), available_size)
    candidate_df = candidate_df.loc[candidate_indexes]
    candidate_df.drop("Outcome", axis=1, inplace=True)
    for idx in candidate_indexes:
        candidate_list.append(df_copy.loc[idx].drop("Outcome"))
    return candidate_list, candidate_df, candidate_indexes


# ### Selecting candidates group

all_candidate_lists, candid_df, candid_indexes = candidateBids(df, size=candidates_size)


# ### Predicting probability of winning based on bid features


def WinPrediction(testdataframe):
    try:
        probability = []
        output = pred(testdataframe)
        for i in output:
            probability.append(i[1])
        return probability
    except Exception as ex:
        raise Exception(str(ex))


# ### Method to find probability of wining by setting price values


def probability_function(df, values, column_name="Price"):
    try:
        df_copy = df.copy()
        values = np.round(values, 4)
        if not np.isfinite(values).all():
            raise Exception("Value is too big")
        if len(values) == 1:
            df_copy[column_name] = values[0]
            df_copy = df_copy.values.reshape(1, -1)
        else:
            df_copy[column_name] = values
        return WinPrediction(df_copy)
    except Exception as ex:
        raise Exception(str(ex))


# ### Figures


testing_prices = []
avg_prob_winnings = dict()
max_prob_winning = dict()

first_objectives = []
second_objectives = []
third_objectives = []

first_objs = []
second_objs = []
third_objs = []

costs = randomCosts(df, candid_indexes)
prices = getCandidatesOriginalPrices(df, candid_indexes)

threshold = [0.5 + 0.3 * random.random() for i in range(candidates_size)]
# PDGW =  [0.2,0.4,0.6,0.8,1]

all_costs = costs
all_prices = prices
testing_prices_dict = dict()
all_costs_dict = dict()

for i, c in enumerate(all_costs):
    testing_prices.append(list(np.linspace(c, 5 * c, 150)))
    testing_prices_dict[candid_indexes[i]] = list(np.linspace(c, 5 * c, 150))
    all_costs_dict[candid_indexes[i]] = c

group_cand_ind = 0

for i, candidate in enumerate(all_candidate_lists):
    f = [
        firstObjective(candidate, [test_price], [all_costs[i]], probability_function)
        for test_price in testing_prices[i]
    ]
    s = [
        secondObjective(candidate, [test_price], probability_function)
        for test_price in testing_prices[i]
    ]
    # t =[thirdObjectiveConstDec(candidate,[test_price],probability_function,[PDGW[i%5]],all_clfs[0],threshold[i]) for test_price in testing_prices[i]]
    avg_prob_winnings[candid_indexes[i]] = np.average(s)
    max_prob_winning[candid_indexes[i]] = np.max(s)
    first_objs.append(f)
    second_objs.append(s)
    # third_objs.append(t)
    group_cand_ind += 1

    if group_cand_ind == group_size:
        group_cand_ind = 0
        first_objectives.append(first_objs)
        second_objectives.append(second_objs)
        # third_objectives.append(third_objs)
        first_objs = []
        second_objs = []
        # third_objs=[]

ord_avg_win = OrderedDict(
    sorted(avg_prob_winnings.items(), key=lambda t: t[1], reverse=True)
)


# ### Normalizing first objective

# In[30]:


first_objectives_norm = []
f_obj_temp = []
for temp_f_objs in first_objectives:
    for temp_f in temp_f_objs:
        min_f, max_f = min(temp_f), max(temp_f)
        new_f = [(val - min_f) / (max_f - min_f) for val in temp_f]
        f_obj_temp.append(new_f)
    first_objectives_norm.append(f_obj_temp)
    f_obj_temp = []


# ### Three objectives multiple deals

# In[31]:


f_aggs, s_aggs, t_aggs = [], [], []
for f_objs, s_objs, t_objs in zip(
    first_objectives, second_objectives, third_objectives
):
    f_agg_temp, s_agg_temp, t_agg_temp = [], [], []
    for i in range(len(f_objs[0])):
        sum_f, sum_s, sum_t = 0, 0, 0
        for k in range(len(f_objs)):
            sum_f += f_objs[k][i]
            sum_s += s_objs[k][i]
            sum_t += t_objs[k][i]
        f_agg_temp.append(sum_f)
        s_agg_temp.append(sum_s)
        t_agg_temp.append(sum_t)
    f_aggs.append(f_agg_temp)
    s_aggs.append(s_agg_temp)
    t_aggs.append(t_agg_temp)


# ### Normalized aggregated three objectives

# In[32]:


first_objectives_agg_norm = []
second_objectives_agg_norm = []
# third_objectives_agg_norm=[]


for temp_fs, temp_ss in zip(f_aggs, s_aggs):
    min_f, max_f = min(temp_fs), max(temp_fs)
    min_s, max_s = min(temp_ss), max(temp_ss)
    # min_t,max_t = min(temp_ts),max(temp_ts)
    new_f = [(val - min_f) / (max_f - min_f) for val in temp_fs]
    new_s = [(val - min_s) / (max_s - min_s) for val in temp_ss]
    # new_t = [(val-min_t)/(max_t-min_t) if (max_t-min_t)!=0 else 0  for val in temp_ts]
    first_objectives_agg_norm.append(new_f)
    second_objectives_agg_norm.append(new_s)
    # third_objectives_agg_norm.append(new_t)
    new_f, new_s = [], []


# ### Visualization of profit

# #####  First objective single deals

# In[33]:


colors = sorted(list(mcolors.cnames), reverse=True)
labels = "1st obj "
plt.style.use("fivethirtyeight")
fig_number = 1
for index, f_objs in enumerate(first_objectives):
    for i, f in enumerate(f_objs):
        fg, axes = plt.subplots(1, 1, figsize=(12, 8))
        axes.scatter(
            testing_prices[(index) * group_size + i],
            f,
            c=colors[i % 5],
            label=labels + " deal " + str((index) * group_size + i + 1),
        )
        axes.legend()
        axes.set_title("Price vs Expected Profit")
        axes.set_xlabel("Prices")
        axes.set_ylabel("Expected Profit")
        fg.tight_layout()
        fg.savefig(
            str(FOLDER_LABEL)
            + "/Result figures/First obj single/Price vs Expected Profit Figure# "
            + str(fig_number)
        )
        fig_number += 1

plt.close("all")


# ### Visualization of winning probability single deals
#

# In[34]:


labels = "1st obj "
plt.style.use("fivethirtyeight")
fig_number = 1
for index, s_objs in enumerate(second_objectives):
    for i, s in enumerate(s_objs):
        fg, axes = plt.subplots(1, 1, figsize=(12, 8))
        axes.scatter(
            testing_prices[(index) * group_size + i],
            s,
            c=colors[i % 5],
            label=labels + " deal " + str((index) * group_size + i + 1),
        )
        axes.legend()
        axes.set_title("Price vs Probability of winning")
        axes.set_xlabel("Price")
        axes.set_ylabel("Probability of winning")
        fg.tight_layout()
        fg.savefig(
            str(FOLDER_LABEL)
            + "/Result figures/Second obj single/Price vs Probability of winning Figure# "
            + str(fig_number)
        )
        fig_number += 1


plt.close("all")


# ### Visualization of winning probability single deals

# In[35]:


labels = "1st obj "
plt.style.use("fivethirtyeight")
fig_number = 1
for index, s_objs in enumerate(second_objectives):
    for i, s in enumerate(s_objs):
        fg, axes = plt.subplots(1, 1, figsize=(12, 8))
        axes.scatter(
            testing_prices[(index) * group_size + i],
            s,
            c=colors[i % 5],
            label=labels + " deal " + str((index) * group_size + i + 1),
        )
        axes.legend()
        axes.set_title("Price vs Probability of winning")
        axes.set_xlabel("Price")
        axes.set_ylabel("Probability of winning")
        fg.tight_layout()
        fg.savefig(
            str(FOLDER_LABEL)
            + "/Result figures/Second obj single/Price vs Probability of winning Figure# "
            + str(fig_number)
        )
        fig_number += 1


plt.close("all")


# #### Visualization of Profit vs probability of winning single deals

# In[36]:


plt.style.use("fivethirtyeight")
fig_number = 1
for index, (f_objs, s_objs) in enumerate(zip(first_objectives, second_objectives)):
    for i, (f, s) in enumerate(zip(f_objs, s_objs)):
        fg, axes = plt.subplots(1, 1, figsize=(12, 8))
        axes.scatter(f, s, c=colors[i], label="deal " + str(index * group_size + i + 1))
        axes.legend()
        axes.set_title("Expected Profit vs probability")
        axes.set_xlabel("Expected profit")
        axes.set_ylabel("Probability")
        fg.savefig(
            str(FOLDER_LABEL)
            + "/Result figures/First obj vs Second obj single/Expected Profit vs Probability of winning single deal Figure# "
            + str(fig_number)
        )
        fig_number += 1
    fg.tight_layout()
plt.close("all")


#  #### All Prices and costs


# In[38]:


for i, (cost, price) in enumerate(zip(all_costs, all_prices)):
    if i % 5 == 0:
        print("\nCandidate deals group : {}".format((i / 5) + 1))
    print("Deal: {} Cost: {} Price: {}".format(i + 1, round(cost, 2), round(price, 2)))
# print("Prices: {}\nCosts: {}".format(all_prices,all_costs))


# Next three funcs: Ideal and nadir points for the three objectives


def getFirstIdealDiffEvo(fun, df, costs, probFun, bound, ideal=True):
    try:
        coef = 1 if ideal == True else -1
        f = lambda x: coef * fun(df, [x], costs, probFun)
        x_opt = scipy.optimize.differential_evolution(f, bound, tol=TOL, atol=ATOL)
        return x_opt
    except Exception as ex:
        raise Exception(str(ex))


def getSecondIdealDiffEvo(fun, df, probFun, bound, ideal=True):
    try:
        coef = 1 if ideal == True else -1
        f = lambda x: coef * fun(df, [x], probFun)
        x_opt = scipy.optimize.differential_evolution(f, bound, tol=TOL, atol=ATOL)
        return x_opt
    except Exception as ex:
        raise Exception(str(ex))


def getThirdIdealDiffEvo(
    fun, df, probFun, PDGW, threshold, bound, x=1, y=1, ideal=True
):
    try:
        coef = 1 if ideal == True else -1
        f = lambda x: coef * fun(df, [x], probFun, PDGW, threshold, x, y)
        x_opt = scipy.optimize.differential_evolution(f, bound, tol=TOL, atol=ATOL)
        return x_opt
    except Exception as ex:
        raise Exception(str(ex))


def OptimizeASF_three_objectivesModified(
    df, function, probFun, costs, PDGW, bound, reference, Ideal, Nadir, threshold, x, y
):
    try:
        f = lambda z: -min(
            [
                (
                    function(df, z, costs, probFun, PDGW, threshold, x, y)[i]
                    - reference[i]
                )
                / (Nadir[i] - Ideal[i])
                for i in range(len(reference))
            ]
        ) - 0.000001 * sum(
            [
                (function(df, z, costs, probFun, PDGW, threshold, x, y)[i])
                / (Nadir[i] - Ideal[i])
                for i in range(len(reference))
            ]
        )
        x_opt = scipy.optimize.differential_evolution(f, bound, tol=TOL, atol=ATOL)
        return x_opt
    except Exception as ex:
        raise Exception("Exception is raised: " + str(ex))


# #### Parameter setting


# group_bounds = [[(c,float("%.2f"%round(5*c,2))) for c in all_costs[i*group_size:(i+1)*group_size]] for i in range(int(len(all_costs)/group_size))]
single_bounds = [(c, float("%.2f" % round(COST_COEF * c, 2))) for c in all_costs]


# ### Ideal and Nadir


def get_reference_points(IDEAL_VECTOR, NADIR_VECTOR, fav_objs=[]):
    nadir_coef, ideal_coef, ref = [], [], []
    for i in range(1, 4):
        if i in fav_objs:
            ideal_coef.append(2 / 3)
            nadir_coef.append(1 / 3)
        else:
            ideal_coef.append(1 / 3)
            nadir_coef.append(2 / 3)

    ref = [
        ideal * i_coef + nadir * n_coef
        for ideal, nadir, i_coef, n_coef in zip(
            IDEAL_VECTOR, NADIR_VECTOR, ideal_coef, nadir_coef
        )
    ]
    return ref


def getIdealAndNadir(i, deal, third_obj_funcs, pdgw, x, y):
    try:
        bound = [(all_costs[i], 5 * all_costs[i])]
        # print(bound)
        first_ideal = getFirstIdealDiffEvo(
            firstObjective, deal, [all_costs[i]], probability_function, bound
        )  # ideal

        first_nadir = getFirstIdealDiffEvo(
            firstObjective, deal, [all_costs[i]], probability_function, bound, False
        )

        second_ideal = getSecondIdealDiffEvo(
            secondObjective, deal, probability_function, bound
        )

        second_nadir = getSecondIdealDiffEvo(
            secondObjective, deal, probability_function, bound, False
        )

        third_ideal = getThirdIdealDiffEvo(
            third_obj_funcs, deal, probability_function, [pdgw], THRESHOLD, bound, x, y
        )

        third_nadir = getThirdIdealDiffEvo(
            third_obj_funcs,
            deal,
            probability_function,
            [pdgw],
            THRESHOLD,
            bound,
            x,
            y,
            False,
        )

        # print("Round: {} \nFirst: {}\nSecond: {}\nThird: {}".format(i+1,first_ideal.fun,second_ideal.fun,third_ideal.fun))
        NADIR_VECTOR = [
            -1 * first_nadir.fun,
            -1 * second_nadir.fun,
            -1 * third_nadir.fun,
        ]
        IDEAL_VECTOR = [first_ideal.fun, second_ideal.fun, third_ideal.fun]
        return IDEAL_VECTOR, NADIR_VECTOR
    except Exception as ex:
        raise Exception(str(ex))


# ### Differential evolution Three objectives optimization single deals
#

# In[ ]:


maxRun = (len(X) * len(Y) - len(X)) * selected_deals_size * len(PDGW)
price_vals, fun_vals, s_fun_vals, t_fun_vals = {}, {}, {}, {}
counter = 0
# IDEAL_VECTOR, NADIR_VECTOR = getIdealAndNadir(i,deal,thirdObjectiveConstDec,pdgw,X[0],Y[0])
PREFERRED_COUNTRIES = get_preferred_countries()
for x in X:
    for y in Y:
        if x != y:
            for tempIndex in range(0, 10):
                (
                    price_vals[tempIndex],
                    fun_vals[tempIndex],
                    s_fun_vals[tempIndex],
                    t_fun_vals[tempIndex],
                ) = ([], [], [], [])
            for i, deal in enumerate(all_candidate_lists):
                for k, pdgw in enumerate(PDGW):
                    totalRuns = (
                        counter * len(PDGW) * selected_deals_size
                        + i * len(PDGW)
                        + k
                        + 1
                    )
                    try:
                        IDEAL_VECTOR, NADIR_VECTOR = getIdealAndNadir(
                            i,
                            deal,
                            thirdObjectiveConstDec,
                            0.5,
                            X[0],
                            Y[0],  # TODO: Error? pdgw not being used
                        )

                        pref_obj = [2] if deal["Country"] in PREFERRED_COUNTRIES else []
                        ref = get_reference_points(IDEAL_VECTOR, NADIR_VECTOR, pref_obj)

                        print(
                            "\nRun: {} of {} \nDeal: {} \nCost: {} \nPrice: {} \nX: {} \nY: {} \nPDGW: {} \nIDEAL: {} \nNADIR: {}".format(
                                totalRuns,
                                maxRun,
                                i + 1,
                                all_costs[i],
                                all_prices[i],
                                x,
                                y,
                                pdgw,
                                IDEAL_VECTOR,
                                NADIR_VECTOR,
                            )
                        )

                        # print(IDEAL_VECTOR, NADIR_VECTOR)
                        x_val_3ob = OptimizeASF_three_objectivesModified(
                            deal,
                            all_three_objectivesConstDec,
                            probability_function,
                            [all_costs[i]],
                            [pdgw],
                            [single_bounds[i]],
                            ref,
                            IDEAL_VECTOR,
                            NADIR_VECTOR,
                            THRESHOLD,
                            x,
                            y,
                        )

                        x_val_3ob = float("%.6f" % np.round(x_val_3ob.x, 2))

                        fun_val = firstObjective(
                            deal, [x_val_3ob], [all_costs[i]], probability_function
                        )
                        fun_val = float("%.6f" % np.round(fun_val, 2))
                        s_fun_val = secondObjective(
                            deal, [x_val_3ob], probability_function
                        )
                        s_fun_val = float("%.6f" % np.round(s_fun_val, 2))
                        t_fun_val = thirdObjectiveConstDec(
                            deal,
                            [x_val_3ob],
                            probability_function,
                            [pdgw],
                            THRESHOLD,
                            x,
                            y,
                        )

                        t_fun_val = float("%.6f" % np.round(t_fun_val, 2))

                        fun_vals[k].append(fun_val)
                        s_fun_vals[k].append(s_fun_val)
                        t_fun_vals[k].append(t_fun_val)
                        price_vals[k].append(x_val_3ob)

                        print(
                            "\nPDGW: {} \n\t\tOptimal price: {} \n\t\tFirst obj: {}\n\t\tSecond obj: {}\n\t\tThird obj: {}".format(
                                pdgw, x_val_3ob, fun_val, s_fun_val, t_fun_val
                            )
                        )

                    except Exception as ex:
                        print(
                            " There was an exception for X: {} and Y: {} \n{}".format(
                                x, y, ex
                            )
                        )
                        exception_fileName = (
                            str(FOLDER_LABEL)
                            + "/Report/ConstDec/exceptionStackTrace.txt"
                        )
                        for tempIndex in range(k, 10):
                            fun_vals[tempIndex].append(-1)
                            s_fun_vals[tempIndex].append(-1)
                            t_fun_vals[tempIndex].append(-1)
                            price_vals[tempIndex].append(-1)
                        break
                        exc_type, exc_value, exc_trace = sys.exe_info()
                        exception_file = open(exception_fileName, "a+")
                        exception_file.write(exc_type + "\n")
                        exception_file.write(exc_value + "\n")
                        exception_file.write(exc_trace + "\n")
                        exception_file.close()

            temp_sln_dict = dict()
            for i in range(1, len(PDGW) + 1):
                temp_sln_dict["Optimal price at PDGW " + str(i / 10)] = price_vals[
                    i - 1
                ]
            for i in range(1, len(PDGW) + 1):
                temp_sln_dict["1st obj at PDGW " + str(i / 10)] = fun_vals[i - 1]

            for i in range(1, len(PDGW) + 1):
                temp_sln_dict["2nd obj at PDGW " + str(i / 10)] = s_fun_vals[i - 1]

            for i in range(1, len(PDGW) + 1):
                temp_sln_dict["3rd obj at PDGW " + str(i / 10)] = t_fun_vals[i - 1]

            df_third_single_third_const_dec = pd.DataFrame.from_dict(temp_sln_dict)
            # columns = df_third_single_third_const_dec.columns
            # multiple_index = [("Three obj 3rd obj const",col) for col in columns]
            # df_third_single_third_const_dec.columns = pd.MultiIndex.from_tuples(multiple_index)

            df_third_single_third_const_dec["Index"] = candid_indexes
            df_third_single_third_const_dec["X"] = [x] * selected_deals_size
            df_third_single_third_const_dec["Y"] = [y] * selected_deals_size

            df_third_single_third_const_dec.to_csv(
                str(FOLDER_LABEL)
                + "/Report/ConstDec/Three-Objectives-optimization-new-third-obj-constDec-"
                + str(counter)
                + ".csv"
            )
            df_third_single_third_const_dec.to_excel(
                str(FOLDER_LABEL)
                + "/Report/ConstDec/Three-Objectives-optimization-new-third-obj-constDec-"
                + str(counter)
                + ".xlsx"
            )
            counter += 1
print("Constant Decreasing part done")


counter = 0
maxRun = (len(X) * len(Y) - len(X)) * selected_deals_size * len(PDGW)
for x in X:
    for y in Y:
        if x != y:
            price_vals, fun_vals, s_fun_vals, t_fun_vals = {}, {}, {}, {}
            for k in range(0, 10):
                price_vals[k], fun_vals[k], s_fun_vals[k], t_fun_vals[k] = (
                    [],
                    [],
                    [],
                    [],
                )
            for i, deal in enumerate(all_candidate_lists):
                for k, pdgw in enumerate(PDGW):
                    # totalRuns = counter*len(PDGW)*selected_deals_size + i**len(PDGW) + k + 1
                    totalRuns = (
                        counter * len(PDGW) * selected_deals_size
                        + i * len(PDGW)
                        + k
                        + 1
                    )
                    try:
                        IDEAL_VECTOR, NADIR_VECTOR = getIdealAndNadir(
                            i, deal, thirdObjectiveExpDec, pdgw, X[0], Y[0]
                        )
                        pref_obj = [2] if deal["Country"] in PREFERRED_COUNTRIES else []
                        ref = get_reference_points(IDEAL_VECTOR, NADIR_VECTOR, pref_obj)

                        print(
                            "\nRun: {} of {} \nDeal: {} \nCost: {} \nPrice: {} \nX: {} \nY: {} \nPDGW: {} \nIDEAL: {} \nNADIR: {}".format(
                                totalRuns,
                                maxRun,
                                i + 1,
                                all_costs[i],
                                all_prices[i],
                                x,
                                y,
                                pdgw,
                                IDEAL_VECTOR,
                                NADIR_VECTOR,
                            )
                        )
                        x_val_3ob = OptimizeASF_three_objectivesModified(
                            deal,
                            all_three_objectivesExpDec,
                            probability_function,
                            [all_costs[i]],
                            [pdgw],
                            [single_bounds[i]],
                            ref,
                            IDEAL_VECTOR,
                            NADIR_VECTOR,
                            THRESHOLD,
                            x,
                            y,
                        )
                        x_val_3ob = float("%.6f" % np.round(x_val_3ob.x, 2))

                        fun_val = firstObjective(
                            deal, [x_val_3ob], [all_costs[i]], probability_function
                        )
                        fun_val = float("%.6f" % np.round(fun_val, 2))
                        s_fun_val = secondObjective(
                            deal, [x_val_3ob], probability_function
                        )
                        s_fun_val = float("%.6f" % np.round(s_fun_val, 2))
                        t_fun_val = thirdObjectiveExpDec(
                            deal,
                            [x_val_3ob],
                            probability_function,
                            [pdgw],
                            THRESHOLD,
                            x,
                            y,
                        )
                        t_fun_val = float("%.6f" % np.round(t_fun_val, 2))

                        fun_vals[k].append(fun_val)
                        s_fun_vals[k].append(s_fun_val)
                        t_fun_vals[k].append(t_fun_val)
                        price_vals[k].append(x_val_3ob)

                        print(
                            "X: {} Y: {} \n  PDGW: {} \n\t\tOptimal price: {}\n\t\tFirst obj: {}\n\t\tSecond obj: {}\n\t\tThird obj: {}".format(
                                x, y, pdgw, x_val_3ob, fun_val, s_fun_val, t_fun_val
                            )
                        )
                    except Exception as ex:
                        print(
                            " There was an exception for X: {} and Y: {} \n{}".format(
                                x, y, ex
                            )
                        )
                        exception_fileName = (
                            str(FOLDER_LABEL) + "/Report/ExpDec/exceptionStackTrace.txt"
                        )
                        for tempIndex in range(k, 10):
                            fun_vals[tempIndex].append(-1)
                            s_fun_vals[tempIndex].append(-1)
                            t_fun_vals[tempIndex].append(-1)
                            price_vals[tempIndex].append(-1)
                        break
                        exc_type, exc_value, exc_trace = sys.exe_info()
                        exception_file = open(exception_fileName, "a+")
                        exception_file.write(exc_type + "\n")
                        exception_file.write(exc_value + "\n")
                        exception_file.write(exc_trace + "\n")
                        exception_file.close()

            temp_sln_dict = dict()
            for i in range(1, len(PDGW) + 1):
                temp_sln_dict["Opt price at PDGW " + str(i / 10)] = price_vals[i - 1]
            for i in range(1, len(PDGW) + 1):
                temp_sln_dict["1st obj at PDGW " + str(i / 10)] = fun_vals[i - 1]

            for i in range(1, len(PDGW) + 1):
                temp_sln_dict["2nd obj at PDGW " + str(i / 10)] = s_fun_vals[i - 1]

            for i in range(1, len(PDGW) + 1):
                temp_sln_dict["3rd obj PDGW  at " + str(i / 10)] = t_fun_vals[i - 1]

            df_third_single_third_Exp_dec = pd.DataFrame.from_dict(temp_sln_dict)
            # columns = df_third_single_third_Exp_dec.columns
            # multiple_index = [("Three obj- 3rd obj Exp",col) for col in columns]
            # df_third_single_third_Exp_dec.columns = pd.MultiIndex.from_tuples(multiple_index)

            df_third_single_third_Exp_dec["Index"] = candid_indexes
            df_third_single_third_Exp_dec["X"] = [x] * selected_deals_size
            df_third_single_third_Exp_dec["Y"] = [y] * selected_deals_size

            df_third_single_third_Exp_dec.to_csv(
                str(FOLDER_LABEL)
                + "/Report/ExpDec/Three-Objectives-optimization-third-obj-expDec-"
                + str(counter)
                + ".csv"
            )
            df_third_single_third_Exp_dec.to_excel(
                str(FOLDER_LABEL)
                + "/Report/ExpDec/Three-Objectives-optimization-third-obj-expDec-"
                + str(counter)
                + ".xlsx"
            )
            counter += 1
print("Exponential decreasing part done")


# #### Result csv files


df_temp_merger = pd.merge(
    df_third_single_third_const_dec,
    df_third_single_third_Exp_dec,
    on="Index",
    how="inner",
)
df_temp_merger.to_csv(str(FOLDER_LABEL) + "/Report/Final report.csv")
df_temp_merger.to_excel(str(FOLDER_LABEL) + "/Report/Final report.xlsx")


def checkDesiredPatter(df, index, columns):
    """70% of the results are increasing and the once which are decreasing,
    should not decrease less than 70% of the previous value"""
    fouls = 0
    result = True
    prevCol = columns[0]
    restColumns = columns[1:]
    for col in restColumns:
        if df.iloc[index][prevCol] > df.iloc[index][col]:
            fouls += 1
        if (
            (df.iloc[index][col] < 0.7 * df.iloc[index][prevCol])
            or (fouls / 9.0 > 0.3)
            or (df.iloc[index][col] == -1)
        ):
            result = False
            break
        prevCol = col
    return result


# In[ ]:


def getAllFiles(folderName):
    return [
        join(folderName, f) for f in listdir(folderName) if isfile(join(folderName, f))
    ]


def selectFilesWithDesiredPattern(fileNameList):
    selectedFiles = []
    totalValidFiles = 0
    column_names = ["2nd obj at PDGW " + str(i / 10) for i in range(1, len(PDGW) + 1)]
    for fileName in fileNameList:
        try:
            count = 0
            df = pd.read_csv(fileName)
            # print("NO EXCEPTION")
            for index in range(0, len(df)):
                desiredDeal = checkDesiredPatter(df, index, column_names)
                if desiredDeal == True:
                    count += 1
                    totalValidFiles += 1
                if count != 0:
                    selectedFiles.append(fileName)
                    break
        except Exception as ex:
            print(" Exception, {}".format(str(ex)))
    print("\n\nTotal Valid Files: {}".format(totalValidFiles))
    return selectedFiles


constFolderName = FOLDER_LABEL + "/Report/ConstDec"
expFolderName = FOLDER_LABEL + "/Report/ExpDec"

allConstFiles = getAllFiles(constFolderName)
allExpFiles = getAllFiles(expFolderName)

totFiles = allConstFiles + allExpFiles

selectedFiles = selectFilesWithDesiredPattern(totFiles)
print(selectedFiles)
# print("Selected Files constant third objective: {}".format(selectedFilesConst))
# print("Selected Files Exponential third objective: {}".format(selectedFilesExp))


# In[ ]:


f = open(str(FOLDER_LABEL) + "/Report/selectedFiles.txt", "w")
for file in selectedFiles:
    f.write(file + "\n")
f.close()

f = open(str(FOLDER_LABEL) + "/Report/preferred_countries.txt", "w")
for country in PREFERRED_COUNTRIES:
    f.write(str(country) + "\n")
f.close()
