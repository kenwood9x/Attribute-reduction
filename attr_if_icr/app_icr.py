from tabnanny import verbose
import numpy as np
import math

from intuition_fuzzy import IntuitiveFuzzy
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

data_file = "C:/Users/M4800/Documents/Zalo Received Files/Large Data/german.csv"
att_nominal_cate = ["a1","a3","a4","a6","a7","a9","a10","a12","a14","a15","a17","a19","a20","d"] #german#
#att_nominal_cate = ["a1","a2","a3","a8","d"]
#att_nominal_cate = ["a1","a2","d"]
#att_nominal_cate = ["quality"] #winequality-red
#att_nominal_cate = ["class"]  #forest
#att_nominal_cate = ["d"] 
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

def splitDS(DS,number_split):
    index_array = []
    number_object = DS.shape[0] - 1
    batch_split = math.ceil(number_object / number_split)
    batch = 0
    for i in range(number_split):
        if batch + batch_split >= number_object: batch = number_object + 1
        else: batch = batch + batch_split + 1
        index_array.append(batch)
    return index_array


if __name__ == "__main__":
    DS = preprocessing(data_file)
    ind_arr = splitDS(DS,5)
    
    DS1 = DS[:ind_arr[1]]
    F1 = IntuitiveFuzzy(DS1, att_nominal_cate, ind_arr[0:2])
    W1 = F1.remove_object()
    W1 = F1.filter_icr(verbose=True)
        
    DS2 = DS[:ind_arr[2]]
    F2 = IntuitiveFuzzy(DS2, att_nominal_cate, ind_arr[1:3])
    W2 = F2.remove_object()
    W2 = F2.filter_icr(verbose=True) 
    
    DS3 = DS[:ind_arr[3]]
    F3 = IntuitiveFuzzy(DS3, att_nominal_cate, ind_arr[2:4])
    W3 = F3.remove_object()
    W3 = F3.filter_icr(verbose=True)
    
    DS4 = DS[:ind_arr[4]]
    F4 = IntuitiveFuzzy(DS4, att_nominal_cate, ind_arr[3:5])
    W4 = F4.remove_object()
    W4 = F4.filter_icr(verbose=True)  
    
    
    
    print(W4[-1])
    print(F4.evaluate(W4,10))
    #print(ind_arr)
    #ind_arr = [3,271]

