from tabnanny import verbose
import numpy as np
import math

from intuition_fuzzy import IntuitiveFuzzy
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

data_file = "C:/Users/M4800/Documents/Zalo Received Files/Small Data/winequality-red.csv"
#att_nominal_cate = ["a1","a3","a4","a6","a7","a9","a10","a12","a14","a15","a17","a19","a20","d"] #german#
#att_nominal_cate = ["a1","a2","a3","a8","d"]
#att_nominal_cate = ["a1","a2","d"]
#att_nominal_cate = ["quality"] #winequality-red
#att_nominal_cate = ["class"]  #forest
att_nominal_cate = ["quality"] 
min_max_scaler = preprocessing.MinMaxScaler()


def preprocessing(pathfile):
    DS  = np.genfromtxt(pathfile, delimiter=",", dtype=object)[:,:]
    att = DS[0].astype(str)
    att_real = [i for i in att  if i not in att_nominal_cate ]

    DS[0] = att

    list_index_cate = [list(DS[0]).index(i) for i in att_nominal_cate]
    for i in list_index_cate:
        DS[1:, i] = LabelEncoder().fit_transform(DS[1:,i])

    DS[1:,:] = DS[1:,:].astype(float)
    if len(att_real) > 0 :
        list_index_real = [list(DS[0]).index(i) for i in att_real]
        DS[1:,list_index_real] = min_max_scaler.fit_transform(DS[1:,list_index_real])

    return DS

def split_data(data ,number: int =1):
    if number ==1:
        return [data]
    spt = int(len(data) / number)
    blk = spt*number
    arr = np.split(data[:blk],number)
    arr.append(data[blk:])
    return arr


if __name__ == "__main__":
    DS = preprocessing(data_file)
    number_split = 4
    DS_split = split_data(DS, number_split)
    
    W = []
   
    data = DS_split[0]
    for i in range(1, number_split):
        Fx = IntuitiveFuzzy(data, att_nominal_cate, np.vstack((data, DS_split[i])))
        Wx = Fx.remove_object()
        Wx = Fx.filter_icr(verbose=True)
        print(Wx)
        print(Fx.evaluate(Wx,10))
        W.append(Wx)
        data = np.vstack((data, DS_split[i]))

    
    
    

