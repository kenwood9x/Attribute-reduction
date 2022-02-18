""" Read Page 3 """
import numpy as np
import time

from functools import reduce 
from tabulate import tabulate
from sklearn import svm
from sklearn.model_selection import cross_val_score

class IntuitiveFuzzy(object):

    
    def __init__(self, dataframe, att_nominal_cate):
        super(IntuitiveFuzzy, self).__init__()

        print('[INFO] Initializing object ...')
        self.data = dataframe
        
        # Including decision. Assume last column is decision values
        self.attributes = list(dataframe[0])
        self.C = self.attributes[:-1]
        self.arr_cate = att_nominal_cate
        self.arr_real = [i for i in self.attributes  if i not in att_nominal_cate ]
        
        ### For filtering phase ###
        self.num_attr = len(self.data[0])
        self.num_objs = len(self.data[1:])
        self.num_class = len(set(self.data[1:,-1]))
        self.relational_matrices = self._get_single_attr_IFRM(self.data)
        self.ro = self._get_ro()
        self.time = self.filter(verbose = True)[0]

    def __str__(self):
        string = f"Attributes list : {str(self.attributes)}\n\n"
    
        for attr in self.attributes:
            string+=f'Relational matrix of {attr} : \n{str(self.relational_matrices[attr])}\n\n'

        return string
    
    def _get_single_attr_IFRM(self, data):
        """
            This function returns a dictionary of relational matrices corresponding to
            each single attributes
            Params :
                - data : The numpy DataFrame of sample data 
            Returns : 
                - result : The dictionary of relational matrices corresponding each attribute 
        """

        #Calculate fuzzy matrix d
        column_d = data[1:,-1]
        matrix_fuzzy_d = np.empty((self.num_objs, self.num_objs))
        for i in range(self.num_objs):
            for j in range(self.num_objs):
                mu = 1 - abs(column_d[i] - column_d[j])
                matrix_fuzzy_d[i][j] = mu 

        result = {}
        list_index_real = [list(self.attributes).index(i) for i in self.arr_real] 
        for k in range(len(self.attributes)):
            column = data[1:,k]
            std = np.std(column,ddof=1)
            rel_matrix = np.empty((2,self.num_objs, self.num_objs))

            if k in list_index_real:
                for i in range(self.num_objs):
                    for j in range(self.num_objs):

                        mu = 1 - abs(column[i] - column[j])
                        #v  = (1 - mu) / (1 + lamda * mu)

                        rel_matrix[0][i][j] = mu
                        #rel_matrix[1][i][j] = v
                if std == 0:
                    lamda = 1.0
                else:
                    lamda = np.sum(np.minimum(rel_matrix[0], matrix_fuzzy_d))/np.sum(matrix_fuzzy_d)/std
                    #lamda = 1.
                rel_matrix[1] = (1 - rel_matrix[0]) / (1 + lamda * rel_matrix[0])



            else:
                for i in range(self.num_objs):
                    for j in range(self.num_objs):

                        if column[i] == column[j]:
                            mu = 1.0
                            #v  = 0.0
                        else:
                            mu = 0
                        rel_matrix[0][i][j] = mu

                if std == 0:
                    lamda = 1.0
                else:
                    lamda = np.sum(np.minimum(rel_matrix[0], matrix_fuzzy_d))/np.sum(matrix_fuzzy_d)/std
                    #lamda = 1.0
                rel_matrix[1] = (1 - rel_matrix[0]) / (1 + lamda * rel_matrix[0])                    
            result[self.attributes[k]] = rel_matrix
        return result
        
    
    def _get_union_IFRM(self, IFRM_1, IFRM_2):
        """
            This function will return the intuitive  relational matrix of P union Q
            where P and Q are two Intuitionistic Matrix.
            Note : in the paper R{P union Q} = R{P} intersect R{Q}
            Params :
                - IFRM_1 : First Intuitionistic Fuzzy Matrix
                - IFRM_2 : Second Intuitionistic Fuzzy Matrix
            Returns :
                - result : The IFRM of P intersect Q
        """
        result = np.empty((2,self.num_objs, self.num_objs))

        result[0] = np.minimum(IFRM_1[0], IFRM_2[0])
        result[1] = np.maximum(IFRM_1[1], IFRM_2[1])

        return result

    def _get_multiple_attr_IFRM(self):
        """
            This function returns the intuitive relational matrix of set condition attribute. Matrix C
            Returns :
                - result : The relational matrix C
        """
        d = list(self.relational_matrices.values())
        matrix_C = reduce(self._get_union_IFRM,d[:-1])
        return matrix_C


    def _get_cardinality(self, IFRM):
        """
            Returns the caridnality of a Intuitionistic Matrix
            Params :
                - IFRM : An intuitive fuzzy relational matrix
            Returns :
                - caridnality : The caridnality of that parition 
        """
        ones = np.ones((self.num_objs, self.num_objs))
        caridnality = np.sum((ones + IFRM[0] - IFRM[1])/2)
        return caridnality


    def intuitive_partition_dist_d(self, IFRM):
        """
            This function returns the distance partition to d. 
            Params : IFRM is intuitiononstic fuzzy relation matrix 
            Returns :
                - result : A scalar representing the distance
        """
        IFRM_cardinality = self._get_cardinality(IFRM)
        IFRM_d = self._get_union_IFRM(IFRM,self.relational_matrices[self.attributes[-1]])
        IFRM_d_cardinality = self._get_cardinality(IFRM_d)
        return  (((1 / ((self.num_objs)**2)) * (IFRM_cardinality - IFRM_d_cardinality)) )
 

    def sig(self, IFRM, a):
        """
            This function measures the significance of an attribute a to the set of 
            attributes B. This function begin use step 2.
            Params :
                - IFRM : intuitionistic matrix relation
                - a : an attribute in C but not in B
            Returns :
                - sig : significance value of a to B
        """
        d1 = self.intuitive_partition_dist_d(IFRM)
        d2 = self.intuitive_partition_dist_d(self._get_union_IFRM(IFRM,self.relational_matrices[a]))

        sig = ((d1 - d2))

        return (sig)

    def _get_ro(self):
        num_attribute = self.num_attr 
        if self.num_class <= 4:
            if num_attribute >= 30: return 0.1
            elif 20 < num_attribute < 30: return 0.4
            else: return 0.2
        else:
            if num_attribute < 30: return 0.6
            else: return 0.7
        #return 0.25
        

    def condition_stop (self, IFRM_1, IFRM_2):
        """
        Condition of Algorithm
        """
        IFRM_1_d = self._get_union_IFRM(IFRM_1, self.relational_matrices[self.attributes[-1]])
        IFRM_2_d = self._get_union_IFRM(IFRM_2, self.relational_matrices[self.attributes[-1]])
        sup_muy = np.max(abs(IFRM_1_d[0] - IFRM_2_d[0]))
        sup_nuy = np.max(abs(IFRM_1_d[1] - IFRM_2_d[1]))

        return max(sup_muy,sup_nuy)

    def filter(self, verbose=False):
        """
            The main function for the filter phase
            Params :
                - verbose : Show steps or not
            Returns :
                - W : A list of potential attributes list
        """
        # initialization 
        lis = []
        matrix_C = self._get_multiple_attr_IFRM()
        B = []
        W = []
        d = 1
        D = self.intuitive_partition_dist_d(matrix_C)        
        ro = self._get_ro()
        
        if(verbose):
            print('\n----- Filtering phase -----')
            print('[INFO] Initialization for filter phase done ...')
            print('    --> Distance from B --> (B U {d}) : %.2f' % d)
            print('    --> Distance from C --> (C U {d}) : %.2f' % D)
            print('-------------------------------------------------------')

        # Filter phase 
        num_steps = 1
        max_sig = 0
        c_m = None
        start = time.time()
        for c in (self.C):
            SIG_B_c = 1 - self.intuitive_partition_dist_d(self.relational_matrices[c])
            if(SIG_B_c > max_sig):
                    max_sig = SIG_B_c
                    c_m = c

        B.append(c_m)
        W.append(B.copy())
        if(verbose):
            print(f'[INFO] Step {num_steps} completed : ')
            print(f'    --> Max(SIG_B_c) : {round(max_sig,3)}')
            print(f'    --> Selected c_m = {c_m}')
            print(f'    --> Distance from B -> (B U d) : {d}\n')
        IFRM_TG = self.relational_matrices[c_m]
        condition = self.condition_stop(IFRM_TG, matrix_C)

        while (condition >= 1 - ro and d > D):
            max_sig = 0
            c_m = None
            arr_value = []
            for c in np.setdiff1d(self.C,B):    
                SIG_B_c = self.sig(IFRM_TG, c)
                #print(f'SIG {c} = {SIG_B_c}')
                arr_value.append(SIG_B_c)

            c_m = np.setdiff1d(self.C,B)[np.argmax(arr_value)]        
            #for c in np.setdiff1d(self.C,B):    
            #    SIG_B_c = self.sig(IFRM_TG, c)
            #    if(SIG_B_c >= max_sig):
            #        max_sig = SIG_B_c
            #        c_m = c
            IFRM_TG = self._get_union_IFRM(IFRM_TG,self.relational_matrices[c_m])
            B.append(c_m)
            W.append(B.copy()) 

            # Re-calculate d
            d = self.intuitive_partition_dist_d(IFRM_TG)
            condition = self.condition_stop(IFRM_TG, matrix_C)

            if(verbose):
                print(f'[INFO] Step {num_steps + 1} completed : ')
                print(f'    --> Max(SIG_B_c) : {round(max_sig,2)}')
                print(f'    --> Selected c_m = {c_m}')
                print(f'    --> Distance from B -> (B U d) : {d}\n')

            # increase step number
            num_steps += 1
        finish = time.time() - start
        lis.append(finish)
        lis.append(W[-1])
        print("time process:",finish)
        return lis
    
    def evaluate(self, reduct, k=2):

        y_train = self.data[1:,-1]
        y_train = y_train.astype(int)
        X_train_original = self.data[1:,:-1]
        st_org = time.time()
        clf1 = svm.SVC(kernel='linear', C=1, random_state=42)
        scores_original_1 = round(cross_val_score(clf1, X_train_original, y_train, cv=k).mean(),3)
        
        fn_org = round(time.time() - st_org,3)


        attribute_reduct = reduct[-1]
        list_index = [list(self.data[0,:-1]).index(i) for i in attribute_reduct]

        X_train = self.data[1:,list_index]
        st_reduct = time.time() 
        clf1 = svm.SVC(kernel='linear', C=1, random_state=42)
        scores_1 = round(cross_val_score(clf1, X_train, y_train, cv=k).mean(),3)
        
        fn_reduct = round(time.time() - st_reduct,3)
        head = ["Size-O", "Size-R", "T-O", "T-R", "Acc-O", "Acc-R"]
        my_data = [[len(self.attributes)-1,len(reduct[-1]),fn_org,fn_reduct,scores_original_1,scores_1]]
        return tabulate(my_data, headers=head, tablefmt="grid")
        
