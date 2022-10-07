'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random


'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float


def get_entropy_of_dataset(df):
    #TODO
    entropydataset = 0
    res = df.iloc[:, -1]
    n = 0
    cou = {}
    for k in res:
        if k in cou:
            cou[k] += 1
        else:
            cou[k] = 1
        n += 1

    for k in cou:
        entropydataset -= (cou[k] / n) * (np.log(cou[k] / n) / np.log(2))

    return entropydataset


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float


def get_avg_info_of_attribute(df, attr):
    #TODO
    gro = df.groupby(by=attr)
    n = len(df)

    val = {}
    for key, value in gro:
        val[key] = np.array((value.iloc[:, -1].value_counts()))

    avg_info = 0
    for i in val:
        x = sum(val[i])
        avg_info -= (x / n) * sum((val[i] / x)
                                  * ((np.log(val[i] / x)) / np.log(2)))
    return avg_info


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float


def get_information_gain(df, attr):
    #TODO
    informationgain = get_entropy_of_dataset(
        df) - get_avg_info_of_attribute(df, attr)
    return informationgain


#input: pandas_dataframe
#output: ({dict},'str')

def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    #TODO
    value = {}
    for i in df.columns[:-1]:
        value[i] = get_information_gain(df, i)

    return (value, max(value, key=lambda x: value[x]))