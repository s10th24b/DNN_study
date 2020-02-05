import tensorflow as tf
import numpy as np

def test(y,my_list):
    y+=1
    my_list.append(y)
    return zip(y,my_list)
y = 1
my_list = list()
print("y:",y)
print("my_list:",my_list)
res = test(y,my_list)
print("y:",y)
print("my_list:",my_list)
print("res:",res)

