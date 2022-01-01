def firstObjective(df, prices, costs, probFun):
    try:
        probs = probFun(df, prices)
        return np.sum(
            [(price - cost) * prob for price, cost, prob in zip(prices, costs, probs)]
        )
    except Exception as ex:
        raise Exception(str(ex))


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


def all_three_objectivesConstDec(df, prices, costs, probFun, PDGW, threshold, x, y):
    try:
        return [
            firstObjective(df, prices, costs, probFun),
            secondObjective(df, prices, probFun),
            thirdObjectiveConstDec(df, prices, probFun, PDGW, threshold, x, y),
        ]
    except Exception as ex:
        raise Exception(str(ex))
