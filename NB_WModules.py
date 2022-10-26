from cgi import print_environ
from cmath import log
import piheaan as heaan
import os
import numpy as np
import pandas as pd
import math
import json
import re
import natsort
import NB_log

params = heaan.ParameterPreset.FGb
context = heaan.make_context(params)
heaan.make_bootstrappable(context)
log_num_slot = heaan.get_log_full_slots(context)
num_slot = 1 << log_num_slot

final_ctxt = heaan.Ciphertext(context)


dec = heaan.Decryptor(context)
sk = heaan.SecretKey(context,"./key/secretkey.bin")

#####################################################################################
###############           Naive Bayesian classifier LEARNING          ###############
#####################################################################################
def data_encrypt(csv_file_path, ctxt_path, cell_col_max_list,context, keypack):
    try:
        os.makedirs(name=ctxt_path, mode=0o775, exist_ok=True)
    except Exception as e:
        print("[Error] Could not make train table directory: ", e)
        return
    enc = heaan.Encryptor(context)
    tmp = pd.read_csv(csv_file_path)
    col_name = list(tmp.columns)
    col_max_list = [int(str(cell_col_max).strip()) for cell_col_max in str(cell_col_max_list).split(',')]
    for j, c in enumerate(col_name):
        if 'label' in c:
            m = col_max_list[-1]
        else:
            m = col_max_list[int(c[-1])]
        for n in range(1,m+1):
            _add_one_zero_column(tmp,c,n)
    data = tmp.drop(columns=col_name, axis=1)

    for cname in data.columns:
        msg = heaan.Message(log_num_slot,0)
        for index in range(data[cname].size):
            msg[index]=data[cname][index]
        if ("Unnamed" not in cname):
            c = heaan.Ciphertext(context)
            enc.encrypt(msg,keypack,c)
            c.save(ctxt_path+cname+".ctxt")

    json_opend={}
    json_opend["num_rows"] = len(data)
    json_opend["bin_col_names"] = list(data.columns)
    json_opend["bin_num_lists"] = []

    for i , c in enumerate(col_name):
        if 'label' not in c:
            json_opend["bin_num_lists"].append(c + ":" + ','.join([str(j) for j in range(1,col_max_list[int(c[-1])]+1)]))
        else: 
            json_opend["bin_num_lists"].append(c + ":" + ','.join([str(j) for j in range(1,col_max_list[-1]+1)]))

    with open(ctxt_path+"metadata.json", 'w') as f:
        f.write(json.dumps(json_opend, sort_keys=True, indent=4, separators=(',', ': ')))

def nb_learn(data_ctxt_path,eval, cell_col_max_list,alpha,context, keypack, model_ctxt_path):
    try:
        os.makedirs(name=model_ctxt_path, mode=0o775, exist_ok=True)
    except Exception as e:
        print("[Error] Could not make train table directory: ", e)
        return
    
    enc = heaan.Encryptor(context)
    col_max_list = [int(str(cell_col_max).strip()) for cell_col_max in str(cell_col_max_list).split(',')]

    with open(data_ctxt_path+"metadata.json",'r') as file:
            data_json_opend = json.load(file)
    data_cell_name_list = data_json_opend['bin_col_names']
    total_num = data_json_opend['num_rows']
    total_num_list = [total_num]*num_slot
    total_num_msg = heaan.Message(log_num_slot)
    for k in range(0,num_slot):
        total_num_msg[k] = total_num_list[k]
    # total_num_msg.set_data(total_num_list)
    total_num_ctxt = heaan.Ciphertext(context)
    enc.encrypt(total_num_msg, keypack, total_num_ctxt)
    total_num_ctxt.save(model_ctxt_path+'total_num.ctxt')

    data_cdict = load_ctxt(data_cell_name_list,data_ctxt_path,context)
    x_cols = []
    y_cols = []
    for i in data_cell_name_list:
        if 'X' in i:
            x_cols.append(i)
        else:
            y_cols.append(i)

    y_labels = col_max_list[-1]
    for y_label in range(1,y_labels+1):
        for i,p in enumerate(x_cols):
            # data_cdict[p].to_device()
            # data_cdict[str(y_label)+'Y_'].to_device()
            cname = str(y_label)+ "count"+p+"_"
            out_ctxt = heaan.Ciphertext(context)
            inv_ctxt = heaan.Ciphertext(context)
            rot_ctxt = heaan.Ciphertext(context)
            # out_ctxt.to_device()
            # inv_ctxt.to_device()
            # rot_ctxt.to_device()
            
            eval.mult(data_cdict[str(y_label)+'Y_'],data_cdict[p],out_ctxt)
            check_boot(out_ctxt,eval)
            
            rot_ctxt = rotate_sum(out_ctxt,eval)
            check_boot(rot_ctxt,eval)

            eval.add(rot_ctxt,alpha,rot_ctxt)

            rot_ctxt.save(model_ctxt_path+cname+".ctxt")
        yname = str(y_label)+'acountY_'
        rot_ctxt2 = heaan.Ciphertext(context)
        rot_ctxt2 = rotate_sum(data_cdict[str(y_label)+'Y_'],eval)
        check_boot(rot_ctxt2,eval)
        rot_ctxt2.save(model_ctxt_path+yname+".ctxt")

    fl = [re.sub('.ctxt','',i) for i in os.listdir(model_ctxt_path)]
    fl = natsort.natsorted(fl)

    data_cdict = load_ctxt(fl,model_ctxt_path,context)
    
    inverse_ctxt = heaan.Ciphertext(context)
    inverse_ctxt, inverse_index = make_inverse_ctxt(model_ctxt_path, eval, context,keypack,alpha,col_max_list)
    rotate_ctxt = make_rotate_ctxt(inverse_ctxt,inverse_index,eval,context,keypack)

    m_25_ctxt = make_25_ctxt(rotate_ctxt,eval)
    
    msg1 = [0]*num_slot
    for i in range(0,len(inverse_index)*25):
        msg1[i] = 1
    he_msg1 = heaan.Message(log_num_slot)
    for k in range(0,num_slot):
        he_msg1[k] = msg1[k]
    # he_msg1.set_data(data=msg1)
    # he_msg1.to_device()

    eval.mult(m_25_ctxt,he_msg1,m_25_ctxt)
    check_boot(m_25_ctxt,eval)
    find_log_ctxt = heaan.Ciphertext(context)

    find_log_ctxt = make_log_ctxt(m_25_ctxt, eval, context,keypack)

    final_result = heaan.Ciphertext(context)
    unit_size = 0
    for i in range(len(col_max_list)-1):
        unit_size+=int(col_max_list[i])
        
    unit_size+=1
    rs = [0]*32768
    st=0
    for _ in range(unit_size*col_max_list[-1]):
        rs[st]=1
        st +=25
    
    rs_msg = heaan.Message(log_num_slot)
    for k in range(0,num_slot):
        rs_msg[k] = rs[k]
    # rs_msg.set_data(rs)
    # rs_msg.to_device()
    
    eval.mult(find_log_ctxt, rs_msg, final_result)

    final_result = make_25_ctxt(final_result,eval)

    final_result.save(model_ctxt_path+"model.ctxt")

    flist = os.listdir(model_ctxt_path)
    for i in flist:
        if 'model' not in i:
            os.remove(model_ctxt_path+i)

#####################################################################################
###########################           INFERENCE          ############################
#####################################################################################
def inference_encrypt(csv_file_path, ctxt_path, cell_col_max_list,context, keypack):
    csv_file_path = str(csv_file_path)
    ctxt_path = str(ctxt_path)
    cell_col_max_list = str(cell_col_max_list)
    col_max_list = [int(str(cell_col_max).strip()) for cell_col_max in str(cell_col_max_list).split(',')]
    num_class = col_max_list[-1]
    try:
        os.makedirs(name=ctxt_path, mode=0o775, exist_ok=True)
    except Exception as e:
        print("[Error] Could not make train table directory: ", e)
        return
    enc = heaan.Encryptor(context)
    input_ = pd.read_csv(csv_file_path) 
    ctxt_out = heaan.Ciphertext(context)
    unit_size = 0
    for i in range(len(col_max_list)-1):
        unit_size+=col_max_list[i]

    unit_size+=1
    total_size = unit_size * num_class*25

    ndata = num_slot//total_size
    required_ctxt = len(input_)//ndata if (len(input_) % ndata ==0) else len(input_)//ndata+1 

    input_li = [[0]*num_slot for _ in range(required_ctxt)]
    cur_ctxt=0
    cur_idx=0
    cur_data=0
    for i in range(len(input_)):
        cur_item = input_.iloc[i].to_list()
        for k in range(num_class):
            input_li[int(cur_ctxt)][int(cur_idx+k*unit_size)*25]=1 
        for j in range(len(col_max_list)-1):
            for k in range(num_class):
                input_li[int(cur_ctxt)][int(cur_idx+cur_item[j]+k*unit_size)*25]=1
            cur_idx+=col_max_list[j]
        cur_idx+=1
        cur_data+=1
        if (cur_data == ndata):
            cur_data=0
            cur_ctxt+=1
            cur_idx=0

    for i in range(required_ctxt):
        msg_=heaan.Message(log_num_slot)
        for j in range(0,num_slot):
            msg_[j] = input_li[i][j]
        # msg_.set_data(data = input_li[i])
        enc.encrypt(msg_,keypack,ctxt_out)
        ctxt_out.save(ctxt_path+"/input_"+str(i)+"_NB.ctxt")

def nb_predict(test_ctxt_path,model_ctxt_path, ycn, cell_col_max_list,key_file_path,eval):
    y_class_num = int(ycn)
    cell_col_max_list = str(cell_col_max_list)
    col_max_list = [int(str(cell_col_max).strip()) for cell_col_max in str(cell_col_max_list).split(',')]
    unit_size=0
    for i in range(len(col_max_list)-1):
        unit_size+=col_max_list[i]
    unit_size+=1

    keypack = heaan.KeyPack(context, key_file_path+"/")
    enc = heaan.Encryptor(context)

    model_ctxt = heaan.Ciphertext(context)
    model_ctxt.load(model_ctxt_path+"model.ctxt")
    feature_length = len(col_max_list)-1

    log_ = [-1/(np.log(1/32768)*(feature_length+1))]*num_slot
    logs = heaan.Ciphertext(context)
    logs_m = heaan.Message(log_num_slot)
    for j in range(0,num_slot):
            logs_m[j] = log_[j]
    enc.encrypt(logs_m, keypack, logs)
    # logs.to_device()

    # model_ctxt.to_device()
    check_boot(model_ctxt,eval)

    input_ctxt = heaan.Ciphertext(context)
    input_ctxt.load(test_ctxt_path+"input_0_NB.ctxt")
    # input_ctxt.to_device()
    check_boot(input_ctxt,eval)

    eval.mult(model_ctxt,input_ctxt,input_ctxt)                
    check_boot(input_ctxt,eval)

    eval.mult(input_ctxt,logs,input_ctxt)                
    check_boot(input_ctxt,eval)

    u_size = unit_size
    rot_size = u_size//2 if (u_size % 2 ==0) else u_size//2+1 

    masku = [0]*num_slot 
    maskd = [0]*num_slot 
    
    num_iter = num_slot//(u_size*25)
    for i in range(num_iter):
        for j in range(u_size//2):
            maskd[i*u_size*25+j*25]=1
        for j in range(rot_size):
            masku[i*u_size*25+j*25]=1
    
    tmp_ctxt = heaan.Ciphertext(context)
    eval.left_rotate(input_ctxt,rot_size*25,tmp_ctxt)
    check_boot(tmp_ctxt,eval)

    masku_msg = heaan.Message(log_num_slot)
    for k in range(0,num_slot):
        masku_msg[k] = masku[k]
    # masku_msg.set_data(data = masku)
    # masku_msg.to_device()

    eval.mult(input_ctxt,masku_msg,input_ctxt)
    check_boot(input_ctxt,eval)

    maskd_msg = heaan.Message(log_num_slot)
    for o in range(0,num_slot):
        maskd_msg[o] = maskd[o]
    # maskd_msg.set_data(data = maskd)
    # maskd_msg.to_device()
    eval.mult(tmp_ctxt,maskd_msg,tmp_ctxt)
    check_boot(tmp_ctxt,eval)

    eval.add(input_ctxt,tmp_ctxt,input_ctxt)
    while(rot_size>1):
        rot_size = rot_size//2 if (rot_size % 2 ==0) else rot_size//2+1     
        eval.left_rotate(input_ctxt,rot_size*25,tmp_ctxt)
        check_boot(tmp_ctxt,eval)
        eval.add(input_ctxt,tmp_ctxt,input_ctxt)

    input_ctxt_duplicate = heaan.Ciphertext(input_ctxt)
    # input_ctxt_duplicate.to_device()

    result_ctxt = heaan.Ciphertext(context)
    # result_ctxt.to_host()
    tmp_msg = [0]*num_slot

    for i in range(y_class_num):
        tmp_msg[i]=1

        tmp_ = heaan.Ciphertext(context)
        he_message=heaan.Message(log_num_slot)
        for k in range(0,log_num_slot):
            he_message[k] = 0
        
        enc.encrypt(he_message,keypack, tmp_)
        # tmp_.to_device()

        one_msg = heaan.Message(log_num_slot)
        msg=[0]*num_slot
        msg[i]=1
        for o in range(0,num_slot):
            one_msg[o] = msg[o]
        # one_msg.set_data(msg)
        # one_msg.to_device()
        if i ==0:
            eval.mult(input_ctxt, one_msg, result_ctxt)
            check_boot(result_ctxt,eval)
        else : 
            eval.left_rotate(input_ctxt_duplicate,25*(unit_size)-1,input_ctxt_duplicate)
            check_boot(input_ctxt_duplicate,eval)

            eval.mult(input_ctxt_duplicate,one_msg,tmp_)
            check_boot(tmp_,eval)

            eval.add(tmp_, result_ctxt,result_ctxt)

    tmp_ctxt2 = heaan.Ciphertext(context)
    he_message2=heaan.Message(log_num_slot)
    for k in range(0,log_num_slot):
        he_message2[k] = tmp_msg[k]
    enc.encrypt(he_message2,keypack,tmp_ctxt2)
    # tmp_ctxt2.to_device()

    eval.mult(result_ctxt,tmp_ctxt2,result_ctxt)
    check_boot(result_ctxt,eval)
    
    final_result = heaan.Ciphertext(context)
    # final_result.to_device()
    if y_class_num != 2:
        final_result = findMaxPos(result_ctxt,context,keypack,log_num_slot,y_class_num,eval)

    elif y_class_num ==2 : 
        result_duplicate = heaan.Ciphertext(result_ctxt)
        # result_duplicate.to_device()
        eval.left_rotate(result_duplicate,1,result_duplicate)
        check_boot(result_duplicate,eval)
        eval.mult(result_duplicate,-1,result_duplicate)
        check_boot(result_duplicate,eval)
        eval.add(result_duplicate,result_ctxt,result_ctxt)
        
        heaan.math.approx.sign(eval,result_ctxt,final_result)

    final_result.save(test_ctxt_path+"result.ctxt")

def decrypt_result(model_ctxt_path,ycn,key_file_path):
    dec = heaan.Decryptor(context)
    sk = heaan.SecretKey(context,key_file_path+"/secretkey.bin")
    y_class_num = int(ycn)

    result_ctxt= heaan.Ciphertext(context)
    result_msg = heaan.Message(log_num_slot)

    result_ctxt.load(model_ctxt_path+'result.ctxt')
    # result_ctxt.to_host()
    dec.decrypt(result_ctxt,sk,result_msg)

    if y_class_num==2:
        num = round(result_msg[0].real)
        if num ==1:
            return 1
        else : 
            return 2
    else : 
        num_list = []
        for i in range(y_class_num):
            num_list.append(round(result_msg[i].real))
        return (num_list.index(max(num_list)))+1

#####################################################################################
########################           INNER FUNCTIONS         ##########################
#####################################################################################
def make_rotate_ctxt(input_ctxt, index_dict,eval,context,keypack):
    enc = heaan.Encryptor(context)
    one_ = [0]*num_slot
    one_msg = heaan.Message(log_num_slot)
    for i in range(0,num_slot):
        one_msg[i] = one_[i]
    # one_msg.set_data(one_)

    return_ctxt = heaan.Ciphertext(context)
    enc.encrypt(one_msg,keypack,return_ctxt)

    duplicate_ctxt = heaan.Ciphertext(input_ctxt)
    # duplicate_ctxt.to_device()

    # input_ctxt.to_device()
    # return_ctxt.to_device()

    for i in range(len(index_dict)):
        tmp_ctxt = heaan.Ciphertext(context)
        enc.encrypt(one_msg,keypack,tmp_ctxt)
        mask_ = [0]*num_slot
        mask_[i*25]=1
        mask_msg = heaan.Message(log_num_slot)
        for k in range(0,num_slot):
            mask_msg[k] = mask_[k]
        # mask_msg.set_data(mask_)
        # mask_msg.to_device()
        if i ==0:
            eval.mult(input_ctxt,mask_msg, tmp_ctxt)
            check_boot(tmp_ctxt,eval)
            eval.add(return_ctxt, tmp_ctxt,return_ctxt)
        else:
            eval.right_rotate(duplicate_ctxt,24,duplicate_ctxt)
            check_boot(duplicate_ctxt,eval)
            eval.mult(duplicate_ctxt,mask_msg, tmp_ctxt)
            check_boot(tmp_ctxt,eval)
            check_boot(duplicate_ctxt,eval)
            eval.add(return_ctxt, tmp_ctxt,return_ctxt)

    return return_ctxt

def make_25_ctxt(ctxt,eval):
    origin = heaan.Ciphertext(ctxt)
    tmp_ = heaan.Ciphertext(ctxt)
    eval.right_rotate(ctxt,1,tmp_)
    check_boot(ctxt,eval)
    eval.add(ctxt, tmp_,ctxt)

    eval.right_rotate(ctxt,2,tmp_)
    check_boot(ctxt,eval)
    eval.add(ctxt, tmp_,ctxt)

    eval.right_rotate(ctxt,4,tmp_)
    check_boot(ctxt,eval)
    eval.add(ctxt, tmp_,ctxt)

    ctxt_8_ = heaan.Ciphertext(ctxt)
    eval.right_rotate(ctxt_8_,16,ctxt_8_)
    check_boot(ctxt_8_,eval)

    eval.right_rotate(ctxt,8,tmp_)
    check_boot(ctxt,eval)
    eval.add(ctxt, tmp_,ctxt)
    eval.add(ctxt_8_,ctxt,ctxt)

    eval.right_rotate(origin,24,origin)
    check_boot(origin,eval)

    eval.add(ctxt,origin,ctxt)
    check_boot(ctxt,eval)

    return ctxt

def make_inverse_ctxt(model_ctxt_path, eval, context,keypack,alpha, col_max_list):
    enc = heaan.Encryptor(context)

    f_list = os.listdir(model_ctxt_path)
    file_list = []
    for i in f_list:
        if 'total' not in i:
            file_list.append(re.sub('.ctxt','',i))
    file_list = natsort.natsorted(file_list)

    model_cdict = load_ctxt(file_list,model_ctxt_path,context)
    
    one_ = [0]*num_slot
    one_msg = heaan.Message(log_num_slot)
    for i in range(0,num_slot):
        one_msg[i] = one_[i]
    # one_msg.set_data(one_)

    inverse_index = {}

    inverse_ctxt = heaan.Ciphertext(context)
    enc.encrypt(one_msg, keypack, inverse_ctxt)
    # inverse_ctxt.to_device()

    count_ctxt = heaan.Ciphertext(context)
    enc.encrypt(one_msg, keypack, count_ctxt)
    # count_ctxt.to_device()

    for i,n in enumerate(file_list):        
        if 'log' not in n : 
            inverse_index[i] = n
    
    total_num_ctxt = heaan.Ciphertext(context)
    total_num_ctxt.load(model_ctxt_path+'total_num.ctxt')
    # total_num_ctxt.to_device()

    for i,n in enumerate(inverse_index):
        tmp = [0]*num_slot
        tmp[i]=1
        tmp_msg = heaan.Message(log_num_slot)
        for k in range(0,num_slot):
            tmp_msg[k] = tmp[k]
        # tmp_msg.set_data(data = tmp)
        # tmp_msg.to_device()
        # model_cdict[inverse_index[n]].to_device()

        if 'X' in inverse_index[n]:

            eval.mult(model_cdict[inverse_index[n]],tmp_msg,model_cdict[inverse_index[n]])
            check_boot(model_cdict[inverse_index[n]],eval)

            eval.add(model_cdict[inverse_index[n]],count_ctxt,count_ctxt)

        else:

            tmp_ctxt1 = heaan.Ciphertext(context)
            enc.encrypt(one_msg, keypack, tmp_ctxt1)
            # tmp_ctxt1.to_device()
            eval.mult(model_cdict[inverse_index[n]],tmp_msg,tmp_ctxt1)
            check_boot(tmp_ctxt1,eval)

            eval.add(tmp_ctxt1,count_ctxt,count_ctxt)

    tmp_ctxt = heaan.Ciphertext(context)
    enc.encrypt(one_msg, keypack, tmp_ctxt)
    # tmp_ctxt.to_device()

    tc = [0]*num_slot

    unit_size = 0        
    for i in range(len(col_max_list)-1):
        unit_size+=col_max_list[i]
    unit_size+=1

    y_loca = 0
    for _ in range(col_max_list[-1]):
        tc[y_loca]=1
        y_loca+=unit_size

    y_loc_msg = heaan.Message(log_num_slot)
    for k in range(0,num_slot):
        y_loc_msg[k] = tc[k]
    # y_loc_msg.set_data(tc)
    # y_loc_msg.to_device()
    eval.mult(total_num_ctxt,y_loc_msg,total_num_ctxt)
    eval.add(total_num_ctxt,inverse_ctxt,inverse_ctxt)

    for i in range(1,col_max_list[-1]+1):
        tc = [0]*num_slot
        for j in range(unit_size*(i-1),unit_size*i):
            if j ==unit_size*(i-1):
                pass
            else : 
                tc[j]=1
        y_tmp_msg = heaan.Message(log_num_slot)
        for k in range(0,num_slot):
            y_tmp_msg[k] = tc[k]
        # y_tmp_msg.set_data(tc)
        # y_tmp_msg.to_device()
        eval.mult(model_cdict[str(i)+'acountY_'], y_tmp_msg,model_cdict[str(i)+'acountY_'])
        eval.add(model_cdict[str(i)+'acountY_'],inverse_ctxt,inverse_ctxt)

    nc = [0]*num_slot
    unit_ = [0]*unit_size

    cur=1
    for j in range(len(col_max_list[:-1])):
        ni = col_max_list[j]
        for i in range(cur,cur+ni):
            unit_[i]=alpha*ni
        cur+=ni

    unit_msg = unit_*col_max_list[-1]
    for i,ui in enumerate(unit_msg):
        nc[i]=ui

    nc_msg = heaan.Message(log_num_slot)
    for k in range(0,num_slot):
        nc_msg[k] = nc[k]
    # nc_msg.set_data(nc)
    nc_ctxt = heaan.Ciphertext(context)
    enc.encrypt(nc_msg,keypack,nc_ctxt)
    # nc_ctxt.to_device()

    eval.add(nc_ctxt,inverse_ctxt,inverse_ctxt)


    eval.mult(inverse_ctxt, 0.0001,inverse_ctxt)
    eval.mult(count_ctxt,0.0001,count_ctxt)
    check_boot(inverse_ctxt,eval)
    check_boot(count_ctxt,eval)

    heaan.math.approx.inverse(eval, inverse_ctxt, inverse_ctxt,greater_than_one = False)
    check_boot(inverse_ctxt,eval)

    eval.mult(inverse_ctxt, count_ctxt, inverse_ctxt)

    return inverse_ctxt, inverse_index

def make_log_ctxt(m_25_ctxt, eval, context,keypack):
    key_file_path = "./key"
    sk = heaan.SecretKey(context,key_file_path+"/secretkey.bin")
    dec = heaan.Decryptor(context)

    check_boot(m_25_ctxt,eval)
    log_ctxt = NB_log.find_log(m_25_ctxt,eval,keypack,log_num_slot,context,dec,sk)

    return log_ctxt

def findMax4(c, context, keypack, logN, ndata,eval):
    enc = heaan.Encryptor(context)
    if (ndata==1): 
        return c
    check_boot(c,eval)
    if (ndata % 4 !=0):
        i=ndata
        msg = heaan.Message(logN-1,0)
        while (i % 4 !=0):
            msg[i]=0.00000
            i+=1
        ndata=i
        eval.add(c,msg,c)

    ms1=heaan.Message(log_num_slot)
    for i in range(num_slot):
        ms1[i]=0
    for i in range(ndata//4):
        ms1[i]=1

    msg1 = heaan.Ciphertext(context)
    enc.encrypt(ms1,keypack,msg1)
    # msg1.to_device()

    ca = heaan.Ciphertext(context)
    cb = heaan.Ciphertext(context)
    cc = heaan.Ciphertext(context)
    cd = heaan.Ciphertext(context)
    c1 = heaan.Ciphertext(context)
    c2 = heaan.Ciphertext(context)
    c3 = heaan.Ciphertext(context)
    c4 = heaan.Ciphertext(context)
    c5 = heaan.Ciphertext(context)
    c6 = heaan.Ciphertext(context)
    ctmp1 = heaan.Ciphertext(context)
    ctmp2 = heaan.Ciphertext(context)
    # ca.to_device()
    # cb.to_device()
    # cc.to_device()
    # cd.to_device()
    # c1.to_device()
    # c2.to_device()
    # c3.to_device()
    # c4.to_device()
    # c5.to_device()
    # c6.to_device()
    # ctmp1.to_device()
    # ctmp2.to_device()

    eval.mult(c,msg1,ca)
    eval.left_rotate(c,ndata//4,ctmp1)
    eval.mult(ctmp1,msg1,cb)
    eval.left_rotate(c,ndata//2,ctmp1)
    eval.mult(ctmp1,msg1,cc)
    eval.left_rotate(c,ndata*3//4,ctmp1)
    eval.mult(ctmp1,msg1,cd)

    check_boot(ca,eval)
    check_boot(cb,eval)
    check_boot(cc,eval)
    check_boot(cd,eval)

    eval.sub(ca,cb,c1)
    eval.sub(cb,cc,c2)
    eval.sub(cc,cd,c3)
    eval.sub(cd,ca,c4)
    eval.sub(ca,cc,c5)
    eval.sub(cb,cd,c6)

    eval.right_rotate(c2,ndata//4,ctmp1)
    eval.add(ctmp1,c1,ctmp1)

    eval.right_rotate(c3,ndata//2,ctmp2)
    eval.add(ctmp1,ctmp2,ctmp1)

    eval.right_rotate(c4,ndata*3//4,ctmp2)
    eval.add(ctmp1,ctmp2,ctmp1)

    eval.right_rotate(c5,ndata,ctmp2)
    eval.add(ctmp1,ctmp2,ctmp1)

    eval.right_rotate(c6,5*ndata//4,ctmp2)
    eval.add(ctmp1,ctmp2,ctmp1)

    c0 = heaan.Ciphertext(context)
    heaan.math.approx.sign(eval,ctmp1,c0)
    check_boot(c0,eval)

    c0_c = heaan.Ciphertext(c0)
    
    mkmsg = [1.0]*num_slot
    mkall = heaan.Message(log_num_slot)
    for k in range(0,num_slot):
        mkall[k] = mkmsg[k]
    # mkall.set_data(mkmsg)
    # mkall.to_device()
    eval.add(c0,mkall,c0)
    eval.mult(c0,0.5,c0)
    check_boot(c0,eval)

    ceq = heaan.Ciphertext(context)
    # ceq.to_device()
    eval.square(c0_c,ceq)
    check_boot(ceq,eval)
    eval.negate(ceq,ceq)
    eval.add(ceq,mkall,ceq)

    mk1 = heaan.Ciphertext(msg1)
    # mk1.to_device() 
    mk2 = heaan.Ciphertext(context)
    mk3 = heaan.Ciphertext(context)
    mk4 = heaan.Ciphertext(context)
    mk5 = heaan.Ciphertext(context)
    mk6 = heaan.Ciphertext(context)
    # mk2.to_device()
    # mk3.to_device()
    # mk4.to_device()
    # mk5.to_device()
    # mk6.to_device()
  
    eval.right_rotate(mk1,ndata//4,mk2)
    eval.right_rotate(mk2,ndata//4,mk3)
    eval.right_rotate(mk3,ndata//4,mk4)
    eval.right_rotate(mk4,ndata//4,mk5)
    eval.right_rotate(mk5,ndata//4,mk6)

    c_neg = heaan.Ciphertext(c0)
    # c_neg.to_device()
    eval.negate(c0,c_neg)

    eval.add(c_neg,mkall,c_neg)

    c0n = heaan.Ciphertext(c0)
    # c0n.to_device()
    eval.mult(c0n,mk1,ctmp1)
    ctxt=c0n
    c_ab = heaan.Ciphertext(ctmp1)

    c0=heaan.Ciphertext(ctxt)
    eval.mult(c_neg,mk4,ctmp2)
    eval.left_rotate(ctmp2,(3*ndata)//4,ctmp2)
    eval.mult(ctmp1,ctmp2,ctmp1)

    eval.mult(c0,mk5,ctmp2)
    eval.left_rotate(ctmp2,ndata,ctmp2)

    cca = heaan.Ciphertext(context)
    eval.mult(ctmp1,ctmp2,cca)

    eval.mult(c_neg,mk1,ctmp1)
    eval.mult(c0,mk2,ctmp2)
    eval.left_rotate(ctmp2,ndata//4,ctmp2)
    c_bc = heaan.Ciphertext(ctmp2)
    eval.mult(ctmp1,ctmp2,ctmp1)

    eval.mult(c0,mk6,ctmp2)
    eval.left_rotate(ctmp2,ndata*5//4,ctmp2)
    ccb = heaan.Ciphertext(context)
    eval.mult(ctmp1,ctmp2,ccb)

    eval.mult(c_neg,mk2,ctmp1)
    eval.left_rotate(ctmp1,ndata//4,ctmp1)
    eval.mult(c0,mk3,ctmp2)
    eval.left_rotate(ctmp2,ndata//2,ctmp2)
    c_cd = heaan.Ciphertext(ctmp2)
    # c_cd.to_device()
    eval.mult(ctmp1,ctmp2,ctmp1)
    eval.mult(c_neg,mk5,ctmp2)
    eval.left_rotate(ctmp2,ndata,ctmp2)
    ccc = heaan.Ciphertext(context)
    # ccc.to_device()
    eval.mult(ctmp1,ctmp2,ccc)

    eval.mult(c_neg,mk3,ctmp1)
    eval.left_rotate(ctmp1,ndata//2,ctmp1)
    eval.mult(c0,mk4,ctmp2)
    eval.left_rotate(ctmp2,3*ndata//4,ctmp2)
    cda = heaan.Ciphertext(ctmp2)
    eval.mult(ctmp1,ctmp2,ctmp1)

    eval.mult(c_neg,mk6,ctmp2)
    eval.left_rotate(ctmp2,5*ndata//4,ctmp2)

    ccd = heaan.Ciphertext(context)
    # ccd.to_device()
    eval.mult(ctmp1,ctmp2,ccd)

    check_boot(cca,eval)
    check_boot(ccb,eval)
    check_boot(ccc,eval)
    check_boot(ccd,eval)

    eval.mult(cca,ca,cca)
    eval.mult(ccb,cb,ccb)
    eval.mult(ccc,cc,ccc)
    eval.mult(ccd,cd,ccd)
    
    cout = heaan.Ciphertext(cca)
    eval.add(cout,ccb,cout)
    eval.add(cout,ccc,cout)
    eval.add(cout,ccd,cout)

    check_boot(cout,eval)

    cneq = heaan.Ciphertext(context)
    # cneq.to_device()
    eval.negate(ceq,cneq)
    eval.add(cneq,mkall,cneq)
    cneq_da = heaan.Ciphertext(context)
    # cneq_da.to_device()
    eval.left_rotate(cneq,(3*ndata)//4,cneq_da)
    eval.mult(cneq_da,mk1,cneq_da)

    cneq_bc = heaan.Ciphertext(context)
    # cneq_bc.to_device()
    eval.left_rotate(cneq,(ndata)//4,cneq_bc)
    eval.mult(cneq_bc,mk1,cneq_bc)

    ceq_ab = heaan.Ciphertext(context)
    # ceq_ab.to_device()
    eval.mult(ceq,mk1,ceq_ab)

    ceq_bc = heaan.Ciphertext(context)
    # ceq_bc.to_device()
    eval.left_rotate(ceq,(ndata)//4,ceq_bc)
    eval.mult(ceq_bc,mk1,ceq_bc)


    ceq_cd = heaan.Ciphertext(context)
    # ceq_cd.to_device()
    eval.left_rotate(ceq,(ndata)//2,ceq_cd)
    eval.mult(ceq_cd,mk1,ceq_cd)

    ceq_da = heaan.Ciphertext(context)
    # ceq_da.to_device()
    eval.negate(cneq_da,ceq_da)
    eval.add(ceq_da,mk1,ceq_da)
    check_boot(ceq,eval)
    check_boot(ceq_ab,eval)
    check_boot(ceq_bc,eval)
    check_boot(ceq_cd,eval)
    check_boot(ceq_da,eval)
    
    eval.mult(ceq_ab,ceq_bc,ctmp2)
    eval.mult(ctmp2,c_cd,ctmp1)
    eval.bootstrap(ctmp1, ctmp1)
    c_cond3 = heaan.Ciphertext(ctmp1)

    eval.mult(ceq_bc,ceq_cd,ctmp1)
    eval.mult(ctmp1,cda,ctmp1)
    eval.add(c_cond3,ctmp1,c_cond3)

    eval.mult(ceq_cd,ceq_da,ctmp1)
    eval.mult(ctmp1,c_ab,ctmp1)
    eval.add(c_cond3,ctmp1,c_cond3)

    eval.mult(ceq_ab,ceq_da,ctmp1)
    eval.mult(ctmp1,c_bc,ctmp1)
    eval.add(c_cond3,ctmp1,c_cond3)

    c_cond4 = heaan.Ciphertext(context)
    eval.mult(ctmp2,ceq_cd,c_cond4)

    eval.bootstrap(c_cond3, c_cond3)
    eval.bootstrap(c_cond4, c_cond4)

    c_tba = heaan.Ciphertext(context)
    eval.mult(c_cond3,0.333333333,c_tba)
    eval.bootstrap(c_tba, c_tba)
    eval.add(c_tba,mkall,c_tba)
    eval.bootstrap(c_cond4, c_cond4)
    eval.add(c_cond4,mkall,ctmp1)   
    eval.mult(c_tba,ctmp1,c_tba)
    eval.mult(cout,c_tba,cout)
    eval.bootstrap(cout, cout)

    return findMax4(cout, context, keypack, logN, ndata//4,eval)

def findMaxPos(c,context,kpack,logN,ndata,eval):
    cmax = findMax4(c,context,kpack,logN,ndata,eval)
    ctmp = heaan.Ciphertext(context)
    for i in range(logN-1):
        eval.right_rotate(cmax,pow(2,i),ctmp)
        eval.add(cmax,ctmp,cmax)
    eval.sub(c,cmax,c)
    c_red = heaan.Ciphertext(context) 
    eval.add(c,0.000001,c)
    # heaan.math.approx.greater_than_zero(eval,c,c_red)
    greater_than_zero_(eval,c,c_red)
    
    c_out=selectRandomOnePos(c_red,context,kpack,logN,ndata,eval)
    return c_out

def greater_than_zero_(eval,ctxt,result_ctxt):
    tmp = heaan.Ciphertext(context)
    tmp2 = heaan.Ciphertext(context)

    heaan.math.approx.sign(eval,ctxt,tmp)
    check_boot(tmp,eval)

    eval.add(tmp, 1, tmp2)

    eval.mult(tmp,tmp2,tmp)
    check_boot(tmp,eval)

    eval.mult(tmp, 0.5, result_ctxt)
    check_boot(result_ctxt,eval)


    

def selectRandomOnePos(c_red,context,kpack,logN,ndata,eval):
    enc = heaan.Encryptor(context)
    m0 = heaan.Message(logN,0.0)
    c_sel = heaan.Ciphertext(context)
    enc.encrypt(m0,kpack,c_sel)
    # c_sel.to_device()
    rando = np.random.permutation(ndata)
    ctmp1 = heaan.Ciphertext(c_red)
    # ctmp1.to_device()
 
    if (ctmp1.level-4< eval.min_level_for_bootstrap):
         eval.bootstrap(ctmp1,ctmp1)
    ctmp2 = heaan.Ciphertext(context)
    # ctmp2.to_device()

    m0[0]=1.0
    # m0.to_device()
    for l in rando:
        if (l>0):
            
            if (ctmp1.level-4< eval.min_level_for_bootstrap):
                eval.bootstrap(ctmp1,ctmp1)

            if (c_sel.level-4< eval.min_level_for_bootstrap):
                eval.bootstrap(c_sel,c_sel)

            eval.left_rotate(ctmp1,l,ctmp1)
            eval.mult(c_sel,ctmp1,ctmp2)
            eval.sub(ctmp1,ctmp2,ctmp1)
            eval.mult(ctmp1,m0,ctmp2)
            eval.right_rotate(ctmp1,l,ctmp1)
            eval.add(c_sel,ctmp2,c_sel)
        else:

            if (ctmp1.level - 4< eval.min_level_for_bootstrap):
                eval.bootstrap(ctmp1,ctmp1)

            if (c_sel.level-4< eval.min_level_for_bootstrap):
                eval.bootstrap(c_sel,c_sel)
            eval.mult(c_sel,ctmp1,ctmp2)
            eval.sub(ctmp1,ctmp2,ctmp1)
            eval.mult(ctmp1,m0,ctmp2)
            eval.add(c_sel,ctmp2,c_sel)

    if (ctmp1.level - 4 < eval.min_level_for_bootstrap):
        eval.bootstrap(ctmp1,ctmp1)

    return ctmp1

def _add_one_zero_column(tmp, column_name,n):
    result = [0]*len(tmp)
    for i,cc in enumerate(tmp[column_name]):
        if cc == n:
            result[i] = 1
        if "feature" in column_name : 
            tmp["X_"+column_name[-1]+"_"+str(n)]=result
        else :
            tmp[str(n)+"Y_"] = result        

def load_ctxt(fn_list,ctxt_path,context):
    out_cdict={}
    for cname in fn_list:
        ctxt = heaan.Ciphertext(context)
        ctxt.load(ctxt_path+cname+".ctxt")
        out_cdict[cname]=ctxt
    return out_cdict

def rotate_sum(input_ctxt,eval):
    tmp_ctxt = heaan.Ciphertext(context)
    for i in range(int(np.log2(num_slot))):
        eval.left_rotate(input_ctxt, 2**i, tmp_ctxt)
        check_boot(tmp_ctxt,eval)
        eval.add(input_ctxt, tmp_ctxt, input_ctxt)
    return input_ctxt

def check_boot(x,eval):
    if x.level==3:
        eval.bootstrap(x, x)
    elif x.level<3:
        exit(0)
    return x

def print_ctxt(c,dec,sk,logN,size):
    # c.to_host()
    m=heaan.Message(logN)
    dec.decrypt(c,sk,m)
    for i in range(size):
        print(i,m[i])
        if (math.isnan(m[i].real)):
            print ("nan detected..stop")
            exit(0)
    # c.to_device()