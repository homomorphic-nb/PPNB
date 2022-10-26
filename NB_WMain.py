import shutil
from tabnanny import check
import NB_WModules
import pandas as pd
import piheaan as heaan
import os
import time
import re
import numpy as np
import json
import natsort
import math
from sklearn.naive_bayes import CategoricalNB

params = heaan.ParameterPreset.FGb
context = heaan.make_context(params)
heaan.make_bootstrappable(context)

sk = heaan.SecretKey(context)
key_file_path = "./key"
os.makedirs(key_file_path, mode=0o775, exist_ok=True)
log_num_slot = heaan.get_log_full_slots(context)
num_slot = 1 << log_num_slot
sk.save(key_file_path+"/secretkey.bin")

sk = heaan.SecretKey(context,key_file_path+"/secretkey.bin")
key_generator = heaan.KeyGenerator(context, sk)
key_generator.gen_common_keys()
key_generator.save(key_file_path + "/")

keypack = heaan.KeyPack(context, key_file_path+"/")
keypack.load_enc_key()
keypack.load_mult_key()
eval = heaan.HomEvaluator(context,keypack)
dec = heaan.Decryptor(context)
enc = heaan.Encryptor(context)


def check_result(ctxt_path, y_class_num,datapath,real_):
    file_list2 = os.listdir(ctxt_path)
    file_list2 = natsort.natsorted(file_list2)
    ppnb_result = []
    for ii,f in enumerate(file_list2): 
        if f == 'w':
            continue
        else:
            result_path = ctxt_path+f+'/'
            results = NB_WModules.decrypt_result(result_path,y_class_num,key_file_path)
            ppnb_result.append(results)
    accuracy_sk_he = cal_accuracy_sk_he(ppnb_result,datapath)
    accuracy_sk_real = cal_accuracy_sk_real(real_,datapath)
    accuracy_he_real = cal_accuracy_he_real(ppnb_result,real_)
    return accuracy_sk_he, accuracy_sk_real, accuracy_he_real

def cal_accuracy_he_real(he_result,real_):
    count = 0
    total = len(he_result)
    for i in range(0,total):
        if he_result[i]==real_[i]:
            count+=1    
    return count/total

def cal_accuracy_sk_real(real_,datapath):
    tr = pd.read_csv(datapath+"train.csv")
    column = tr.columns
    X = tr[[i for i in column[:-1]]]
    Y = tr['label']

    catNB = CategoricalNB(alpha=0.01)
    catNB.fit(X, Y)

    te = pd.read_csv(datapath+"test.csv")
    test_X = te[[j for j in column[:-1]]]

    answer = list(catNB.predict(test_X))
    # print("answer : ", answer)
    # print("real : ",real_)

    count = 0
    total = len(answer)
    for i in range(0,total):
        if real_[i]==answer[i]:
            count+=1    
    return count/total

def cal_accuracy_sk_he(he_result,datapath):
    tr = pd.read_csv(datapath+"train.csv")
    column = tr.columns
    X = tr[[i for i in column[:-1]]]
    Y = tr['label']

    catNB = CategoricalNB(alpha=0.01)
    catNB.fit(X, Y)

    te = pd.read_csv(datapath+"test.csv")
    test_X = te[[j for j in column[:-1]]]

    answer = list(catNB.predict(test_X))
    # print("he_result:",he_result)

    count = 0
    total = len(answer)
    for i in range(0,total):
        if he_result[i]==answer[i]:
            count+=1    
    return count/total


def car_train():
    cell_col_max_list_car = '4,4,5,5,3,3,4'
    csv_data_path = './car_data/car_train.csv'
    data_ctxt_path = './car_ctxt/w/'
    model_ctxt_path = './car_train/w/'
    alpha = 0.01

    print("##TRAIN DATA ENCRYPT")
    NB_WModules.data_encrypt(csv_data_path,data_ctxt_path,cell_col_max_list_car,context,keypack)

    print("##LEARNING")
    NB_WModules.nb_learn(data_ctxt_path,eval, cell_col_max_list_car,alpha,context, keypack, model_ctxt_path)

    
def car_inference():
        cell_col_max_list_car = '4,4,5,5,3,3,4'
        test_data_path = "./car_data/test/"
        model_ctxt_path = './car_train/w/'
        y_class_num=4
        datapath = './car_data/car_'
        path100 = './car_train_log/'

        try:
            os.makedirs(name=path100, mode=0o775, exist_ok=True)
        except Exception as e:
            print("[Error] Could not make train table directory: ", e)
            return

        real_ = [1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 3, 1, 4, 1, 1, 2, 2, 3, 1, 2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 4, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 3, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 2, 4, 2, 1, 2, 4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 4, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 2, 1, 1, 1, 1, 1, 4, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 4, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 4, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 4, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 1, 1, 4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 2, 2, 4, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 4, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 2, 1, 3, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 4, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 3, 2, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 4, 1, 1, 1, 3, 1, 1, 4, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 3, 1, 2, 1, 1, 1, 2, 1, 1, 3, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 4, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 3, 2, 1, 1, 4, 2, 1, 2, 3, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 3, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 4, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 3, 2, 1, 2, 2, 2, 3, 1, 2, 2, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 4, 2, 2, 1, 1, 2, 1, 1, 1, 3, 1, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 3, 1, 2, 1, 1, 4, 1, 4, 2, 4, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 3, 2, 2, 1, 4, 1, 1, 1, 4, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 1, 1, 3, 1, 1, 1, 2, 2, 4, 4, 1, 1, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 4, 1, 2, 1, 1, 1, 3, 1, 1, 2, 1, 2, 1, 3, 2, 2, 2, 1, 1, 2, 1, 1, 3, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 2, 2, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 2, 2, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 4, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 4, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 2, 4, 1, 2, 2, 1, 1, 1, 4, 1, 1, 1, 1, 1, 2, 1, 2, 1, 4, 1, 1, 1, 1, 3, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2, 1, 3, 1, 1, 1, 2, 1, 2, 1, 3, 1, 2, 3, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 4, 1, 1, 1, 1, 1, 2, 4, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 4, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 3, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 3, 2, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 4, 1, 1, 1, 1, 4, 1, 1, 4, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 4, 2, 1, 4, 2, 3, 1, 1, 1, 1, 2, 2, 1, 4, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1, 1, 2, 1, 2, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 4, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, 2, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3, 2, 1, 2, 2, 3, 1, 1, 1, 1, 2, 1, 4, 1, 1, 4, 2, 3, 1, 1, 1, 4, 1, 1, 1, 1, 1, 2, 3, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 3, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 2, 4, 1, 2, 1, 2, 4, 2, 1, 2, 1, 2, 1, 4, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 4, 2]

        test_data_path = "./car_data/test/"
        file_list = [re.sub('.csv','',i) for i in os.listdir(test_data_path)]
        file_list = natsort.natsorted(file_list)

        print("##TEST DATA ENCRYPT")
        for j,k in enumerate(file_list):
            test_data = test_data_path+k +'.csv'
            test_ctxt_path = './car_ctxt/test/'+str(j)+'/'
            NB_WModules.inference_encrypt(test_data,test_ctxt_path,cell_col_max_list_car,context,keypack)


        infer_result = {}
        print("##INFERENCE")
        for j,k in enumerate(file_list):
            test_ctxt_path = './car_ctxt/test/'+str(j)+'/'
            NB_WModules.nb_predict(test_ctxt_path,model_ctxt_path,y_class_num,cell_col_max_list_car,key_file_path, eval)

        accuracy_sk_he, accuracy_sk_real, accuracy_he_real = check_result('./car_ctxt/test/',y_class_num,datapath,real_)

        infer_result["he-sk Accuracy"] = accuracy_sk_he
        infer_result["real-sk Accuracy"] = accuracy_sk_real
        infer_result["real-he Accuracy"] = accuracy_he_real

        with open(path100+'Inference.json', 'w') as f :
            f.write(json.dumps(infer_result, sort_keys=True, indent=4, separators=(',', ': ')))
        

def cancer_train():
    cell_col_max_list_cancer = '10,10,10,10,10,10,10,10,10,2'
    csv_data_path = './cancer_data/cancer_train.csv'
    data_ctxt_path = './cancer_ctxt/w/'
    model_ctxt_path = './cancer_train/w/'
    alpha = 0.01


    print("##TRAIN DATA ENCRYPT")
    NB_WModules.data_encrypt(csv_data_path,data_ctxt_path,cell_col_max_list_cancer,context,keypack)
    print("##LEARNING")
    NB_WModules.nb_learn(data_ctxt_path,eval, cell_col_max_list_cancer,alpha,context, keypack, model_ctxt_path)

def cancer_inference():
        cell_col_max_list_cancer = '10,10,10,10,10,10,10,10,10,2'
        test_data_path = "./cancer_data/test/"
        model_ctxt_path = './cancer_train/w/'
        datapath = './cancer_data/cancer_'
        y_class_num=2
        path100 = './cancer_train_log/'

        try:
            os.makedirs(name=path100, mode=0o775, exist_ok=True)
        except Exception as e:
            print("[Error] Could not make train table directory: ", e)
            return

        real_=[2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1]

        test_data_path = "./cancer_data/test/"
        file_list = [re.sub('.csv','',i) for i in os.listdir(test_data_path)]
        file_list = natsort.natsorted(file_list)

        print("##TEST DATA ENCRYPT")
        for j,k in enumerate(file_list):
            test_data = test_data_path+k +'.csv'
            test_ctxt_path = './cancer_ctxt/test/'+str(j)+'/'
            NB_WModules.inference_encrypt(test_data,test_ctxt_path,cell_col_max_list_cancer,context,keypack)

        infer_result = {}
        print("##INFERENCE")
        for j,k in enumerate(file_list):
            test_ctxt_path = './cancer_ctxt/test/'+str(j)+'/'
            NB_WModules.nb_predict(test_ctxt_path,model_ctxt_path,y_class_num,cell_col_max_list_cancer,key_file_path, eval)

        accuracy_sk_he, accuracy_sk_real, accuracy_he_real = check_result('./cancer_ctxt/test/',y_class_num,datapath,real_)
        infer_result["he-sk Accuracy"] = accuracy_sk_he
        infer_result["real-sk Accuracy"] = accuracy_sk_real
        infer_result["real-he Accuracy"] = accuracy_he_real

        with open(path100+'Inference.json', 'w') as f :
            f.write(json.dumps(infer_result, sort_keys=True, indent=4, separators=(',', ': ')))

def a1_train():
    cell_col_max_list_a1 = '66,2,2,2,2,2,2'
    csv_data_path = './a1_data/a1_train.csv'
    data_ctxt_path = './a1_ctxt/w/'
    model_ctxt_path = './a1_train/w/'
    alpha = 0.01

    print("##TRAIN DATA ENCRYPT")
    NB_WModules.data_encrypt(csv_data_path,data_ctxt_path,cell_col_max_list_a1,context,keypack)
    print("##LEARNING")
    NB_WModules.nb_learn(data_ctxt_path,eval, cell_col_max_list_a1,alpha,context, keypack, model_ctxt_path)

def a1_inference():
        cell_col_max_list_a1 = '66,2,2,2,2,2,2'
        test_data_path = "./a1_data/test/"
        model_ctxt_path = './a1_train/w/'
        y_class_num=2
        datapath = './a1_data/a1_'
        path100 = './a1_train_log/'

        try:
            os.makedirs(name=path100, mode=0o775, exist_ok=True)
        except Exception as e:
            print("[Error] Could not make train table directory: ", e)
            return

        real_=[2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1]  

        test_data_path = "./a1_data/test/"
        file_list = [re.sub('.csv','',i) for i in os.listdir(test_data_path)]
        file_list = natsort.natsorted(file_list)

        print("##TEST DATA ENCRYPT")
        for j,k in enumerate(file_list):
            test_data = test_data_path+k +'.csv'
            test_ctxt_path = './a1_ctxt/test/'+str(j)+'/'
            NB_WModules.inference_encrypt(test_data,test_ctxt_path,cell_col_max_list_a1,context,keypack)

        infer_result = {}
        print("##Inference")
        for j,k in enumerate(file_list):
            test_ctxt_path = './a1_ctxt/test/'+str(j)+'/'
            NB_WModules.nb_predict(test_ctxt_path,model_ctxt_path,y_class_num,cell_col_max_list_a1,key_file_path, eval)

        accuracy_sk_he, accuracy_sk_real, accuracy_he_real = check_result('./a1_ctxt/test/',y_class_num,datapath,real_)
        infer_result["he-sk Accuracy"] = accuracy_sk_he
        infer_result["real-sk Accuracy"] = accuracy_sk_real
        infer_result["real-he Accuracy"] = accuracy_he_real

        with open(path100+'Inference.json', 'w') as f :
            f.write(json.dumps(infer_result, sort_keys=True, indent=4, separators=(',', ': ')))

def a2_train():
    cell_col_max_list_a2 = '66,2,2,2,2,2,2'
    csv_data_path = './a2_data/a2_train.csv'
    data_ctxt_path = './a2_ctxt/w/'
    model_ctxt_path = './a2_train/w/'
    alpha = 0.01

    print("##TRAIN DATA ENCRYPT")
    NB_WModules.data_encrypt(csv_data_path,data_ctxt_path,cell_col_max_list_a2,context,keypack)
    print("##LEARNING")
    NB_WModules.nb_learn(data_ctxt_path,eval, cell_col_max_list_a2,alpha,context, keypack, model_ctxt_path)

def a2_inference():
        cell_col_max_list_a2 = '66,2,2,2,2,2,2'
        test_data_path = "./a2_data/test/"
        model_ctxt_path = './a2_train/w/'
        y_class_num=2
        datapath = './a2_data/a2_'
        path100 = './a2_train_log/'

        try:
            os.makedirs(name=path100, mode=0o775, exist_ok=True)
        except Exception as e:
            print("[Error] Could not make train table directory: ", e)
            return

        real_=[2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 2]

        test_data_path = "./a2_data/test/"
        file_list = [re.sub('.csv','',i) for i in os.listdir(test_data_path)]
        file_list = natsort.natsorted(file_list)

        print("##TEST DATA ENCRYPT")
        for j,k in enumerate(file_list):
            test_data = test_data_path+k +'.csv'
            test_ctxt_path = './a2_ctxt/test/'+str(j)+'/'
            NB_WModules.inference_encrypt(test_data,test_ctxt_path,cell_col_max_list_a2,context,keypack)

        infer_result = {}
        print("##INFERENCE")
        for j,k in enumerate(file_list):
            test_ctxt_path = './a2_ctxt/test/'+str(j)+'/'
            NB_WModules.nb_predict(test_ctxt_path,model_ctxt_path,y_class_num,cell_col_max_list_a2,key_file_path, eval)
    
        accuracy_sk_he, accuracy_sk_real, accuracy_he_real = check_result('./a2_ctxt/test/',y_class_num,datapath,real_)
        infer_result["he-sk Accuracy"] = accuracy_sk_he
        infer_result["real-sk Accuracy"] = accuracy_sk_real
        infer_result["real-he Accuracy"] = accuracy_he_real

        with open(path100+'Inference.json', 'w') as f :
            f.write(json.dumps(infer_result, sort_keys=True, indent=4, separators=(',', ': ')))