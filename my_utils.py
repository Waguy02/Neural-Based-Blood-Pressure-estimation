import json
import os
from enum import Enum
from itertools import islice
import numpy as np

def read_json(path_json):
    with open(path_json, encoding='utf8') as json_file:
        return json.load(json_file)
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
def chunks(data, SIZE):
    """Split a dictionnary into parts of max_size =SIZE"""
    it = iter(data)
    for _ in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}

def sorted_dict(x, ascending=True):
    """
    Sort dict according to value.
    x must be a primitive type: int,float, str...
    @param x:
    @return:
    """
    return dict(sorted(x.items(), key=lambda item: (1 if ascending else -1) * item[1]))
def reverse_dict(input_dict):
    """
    Reverse a dictonary
    Args:
        input_dict:

    Returns:

    """
    inv_dict = {}
    for k, v in input_dict.items():
        inv_dict[v] = inv_dict.get(v, []) + [k]

    return inv_dict

def save_matrix(matrix,filename):
    with open(filename,'wb') as output:
        np.save(output,matrix)
def load_matrix(filename,auto_delete=False):
    with open(filename,'rb') as input:
        matrix=np.load(input)

    if auto_delete:
        os.remove(filename)
    return matrix



class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0