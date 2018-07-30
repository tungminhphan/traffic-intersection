import numpy as np

# Inequality for guard. Left-hand side is variable name, and is bounded on both sides (if one-sided inequality, has +/- infty on other side.)


class Inequality:
    def __init__(self, var, lower=-np.inf, upper=np.inf):
        self.var = var  # variable name
        self.lwrbnd = lower  # the lower bound
        self.uprbnd = upper  # the upper bound
        self.lwrstrict = False  # checks if lwrbnd is >= or >
        self.uprstrict = False

    def show(self):
        if self.lwrbnd > self.uprbnd:
            return False
        elif self.lwrbnd == -np.inf and self.uprbnd == np.inf:
            return True
        elif self.lwrbnd == self.uprbnd:
            return self.var + " = " + str(self.lwrbnd)
        elif self.lwrbnd == -np.inf and self.uprbnd != np.inf:
            return self.var + ' ≤ ' + str(self.uprbnd)
        elif self.lwrbnd != -np.inf and self.uprbnd == np.inf:
            return str(self.lwrbnd) + ' ≤ ' + self.var
        else:
            return str(self.lwrbnd) + ' ≤ ' + self.var + ' ≤ ' + str(self.uprbnd)


# returns a set of inequalities whose conjunction has the same truth value as the conjunction of two sets of inequalities
def conjunct(ineq_dict1, ineq_dict2):
    keys1 = set(ineq_dict1.keys())
    keys2 = set(ineq_dict2.keys())

    shared_keys = keys1 & keys2
    different_keys = (keys1 | keys2) - (keys1 & keys2)
    new_dict = dict()

    for key in shared_keys:
        ineq1 = ineq_dict1[key]
        ineq2 = ineq_dict2[key]
        new_ineq = Inequality(ineq1.var, max(ineq1.lwrbnd, ineq2.lwrbnd), min(
            ineq1.uprbnd, ineq2.uprbnd))  # take the conjunction of the two inqualities
        if new_ineq.show() == False:
            return False
        elif new_ineq.show() != True:
            new_dict[key] = new_ineq

    for key in different_keys:
        if key in ineq_dict1.keys():
            new_dict[key] = ineq_dict1[key]
        else:
            new_dict[key] = ineq_dict2[key]
    return new_dict


def dictionarize(ineq):
    ineq_dict = dict()
    ineq_dict[ineq.var] = ineq
    return ineq_dict


def pretty_print(ineq_dict):  # print contents of a dictionary of inequalities
    keys = sorted(ineq_dict.keys())
    for key in keys:
        print(ineq_dict[key].show())
