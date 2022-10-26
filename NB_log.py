import piheaan as heaan
import os
import time
import numpy as np

import mpmath

LOG2_HI = float.fromhex('0x1.62e42fee00000p-1')
LOG2_LOW = float.fromhex('0x1.a39ef35793c76p-33')
L1 = float.fromhex('0x1.5555555555593p-1')
L2 = float.fromhex('0x1.999999997fa04p-2')
L3 = float.fromhex('0x1.2492494229359p-2')
L4 = float.fromhex('0x1.c71c51d8e78afp-3')
L5 = float.fromhex('0x1.7466496cb03dep-3')
L6 = float.fromhex('0x1.39a09d078c69fp-3')
L7 = float.fromhex('0x1.2f112df3e5244p-3')
SQRT2_HALF = float.fromhex('0x1.6a09e667f3bcdp-1')
CTX = mpmath.MPContext()
CTX.prec = 200  # Bits vs. default of 53

log_num_slot=15
num_slot = 1 << log_num_slot

##All1e1
l=[0]*(1<<log_num_slot)
for j in range((1<<log_num_slot)//25):
    for i in range(1,24):
        l[j*25+i]=1
    l[j*25]=0

he_all1e1 = heaan.Message(log_num_slot)
# he_all1e1.to_host()
for i in range(0,num_slot):
    he_all1e1[i] = l[i]
# he_all1e1.set_data(data=l)
# he_all1e1.to_device()


## Table
table = [0]*(1<<log_num_slot)
for j in range((1<<log_num_slot)//25):
    table[j*25+0] = -16.6366323334
he_table = heaan.Message(log_num_slot)
for i in range(0,num_slot):
    he_table[i] = table[i]
# he_table.set_data(data=table)
# he_table.to_device()


l=[0]*(1<<log_num_slot)
for j in range((1<<log_num_slot)//25):
    l[j*25+0]=l[j*25+1]=l[j*25+2]=1 
    for i in range(3,25):
        l[j*25+i]=0 

he_all0e3 = heaan.Message(log_num_slot)
# he_all0e3.to_host()
for i in range(0,num_slot):
    he_all0e3[i] = l[i]
# he_all0e3.set_data(data=l)
# he_all0e3.to_device()

l = [0]*(1 << log_num_slot)
for j in range((1<<log_num_slot)//25): 
    for i in range(24):
        l[j*25+i]=1

he_all1 = heaan.Message(log_num_slot)
for i in range(0,num_slot):
    he_all1[i] = l[i]
# he_all1.set_data(data=l)
# he_all1.to_device()

l=[0]*(1<<log_num_slot)
for j in range((1<<log_num_slot)//25):
    l[j*25+0]=1 
    for i in range(1,25):
        l[j*25+i]=0 

he_all0e1 = heaan.Message(log_num_slot)
for i in range(0,num_slot):
    he_all0e1[i] = l[i]
# he_all0e1.set_data(data=l)
# he_all0e1.to_device()

msg_pr = [0] * num_slot
for j in range(num_slot//25):
    for i in range(1,25):
        msg_pr[j*25+i] = 2**((-1)*(24-i))

he_msg_pr = heaan.Message(log_num_slot)
for i in range(0,num_slot):
    he_msg_pr[i] = msg_pr[i]
# he_msg_pr.set_data(data=msg_pr)

msg_pr2 = [0]*num_slot
for j in range(num_slot//25):
    for i in range(1,25):
        msg_pr2[j*25+i] = 2**(23-i)*1.4142135623730950488016887242

he_msg_pr2 = heaan.Message(log_num_slot)
for i in range(0,num_slot):
    he_msg_pr2[i] = msg_pr2[i]
# he_msg_pr2.set_data(data=msg_pr2)

msg_pr3 = [0]*num_slot
for j in range(num_slot//25):
    for i in range(1,25):
        msg_pr3[j*25+i] = 0.693147180559945309417232121458*(23.5-i)

he_msg_pr3 = heaan.Message(log_num_slot)
for i in range(0,num_slot):
    he_msg_pr3[i] = msg_pr3[i]
# he_msg_pr3.set_data(data=msg_pr3)


def log_approx(x):
    f = x - 1
    k = 0

    s = f / (2 + f)
    s2 = s * s
    s4 = s2 * s2
    # Terms with odd powers of s^2.
    t1 = s2 * (L1 + s4 * (L3 + s4 * (L5 + s4 * L7)))
    # Terms with even powers of s^2.
    t2 = s4 * (L2 + s4 * (L4 + s4 * L6))
    R = t1 + t2
    hfsq = 0.5 * f * f
    return k * LOG2_HI - ((hfsq - (s * (hfsq + R) + k * LOG2_LOW)) - f)

## Need make sure 2^(-1/2)~2^(1/2)
def HE_log_approx(ctxt, keypack, eval_, log_num_slot, context):
    check_boot(ctxt,eval_)

    ctxt_x1 = heaan.Ciphertext(context)
    ctxt_x1_inv = heaan.Ciphertext(context)
    # ctxt_x1.to_device()
    eval_.add(ctxt,1,ctxt_x1) ## f+2 = x+1  
    heaan.math.approx.inverse(eval_,ctxt_x1,ctxt_x1_inv,init=0.1,num_iter=16,greater_than_one=True) ## 1/x+1 
    check_boot(ctxt_x1_inv,eval_)
    eval_.sub(ctxt_x1,2,ctxt_x1) ### x-1 = f   
    s = heaan.Ciphertext(context)
    s2 = heaan.Ciphertext(context)
    s4 = heaan.Ciphertext(context)
    tmp = heaan.Ciphertext(context)
    tmp2 = heaan.Ciphertext(context)
    # tmp.to_device()
    # tmp2.to_device()

    eval_.mult(ctxt_x1_inv,ctxt_x1,s) ## ctxt_x1 = f = x-1, s = f/(f+2) 
    check_boot(s,eval_)

    eval_.square(s,s2)
    check_boot(s2,eval_)
    eval_.square(s2,s4)
    check_boot(s4,eval_)

    eval_.mult(s4,L7,tmp)
    check_boot(tmp,eval_)
    eval_.add(tmp,L5,tmp)

    eval_.mult(tmp,s4,tmp)
    check_boot(tmp,eval_)
    eval_.add(tmp,L3,tmp)

    eval_.mult(tmp,s4,tmp)
    check_boot(tmp,eval_)
    eval_.add(tmp,L1,tmp)

    eval_.mult(tmp,s2,tmp) ## tmp=t1 
    check_boot(tmp,eval_)

    eval_.mult(s4,L6,tmp2)
    check_boot(tmp2,eval_)
    eval_.add(tmp2,L4,tmp2)

    eval_.mult(tmp2,s4,tmp2)
    check_boot(tmp2,eval_)
    eval_.add(tmp2,L2,tmp2)

    eval_.mult(tmp2,s4,tmp2)
    check_boot(tmp2,eval_)

    R = heaan.Ciphertext(context)
    # R.to_device()
    eval_.add(tmp,tmp2,R)
    
    eval_.square(ctxt_x1,ctxt_x1_inv) ## ctxt_x1_inv = hfsq
    check_boot(ctxt_x1_inv,eval_)
    
    eval_.mult(ctxt_x1_inv,0.5,ctxt_x1_inv)
    check_boot(ctxt_x1_inv,eval_)

    eval_.sub(ctxt_x1,ctxt_x1_inv,tmp) ## tmp = f- hsfq
    eval_.add(ctxt_x1_inv,R,tmp2) ## tmp 2 = hfsq + R
    eval_.mult(tmp2,s,tmp2) ## tmp2 = s*(hfsq+R)
    check_boot(tmp2,eval_)
    eval_.add(tmp,tmp2,tmp) ## f-hsfq + s*(hfsq+R) 
    check_boot(tmp,eval_)

    return tmp


def check_boot(c,eval_):
    if c.level==3:
        eval_.bootstrap(c,c)
    elif c.level<3:
        print("ciphertext level is less than 3.. exiting..\n")
        exit(1)
    return

## Suppose every set of 25 slots has the same values to be logged
def find_log(ctxt1, eval_, keypack, log_num_slot, context, dec, sk):
    ctxt = heaan.Ciphertext(ctxt1)
    # ctxt.to_device()
    # he_msg_pr.to_device()
    # he_msg_pr2.to_device()
    # he_msg_pr3.to_device()

    cf = heaan.Ciphertext(context)
    cf1 = heaan.Ciphertext(context)
    # cf.to_device()
    # cf1.to_device()

    eval_.negate(ctxt,ctxt)
    eval_.add(ctxt,he_msg_pr,cf)
    check_boot(cf,eval_)

    eval_.negate(ctxt,ctxt)
    
    result_msg = heaan.Message()
    # result_msg.to_device()
    '''
    cf.to_host()
    dec.decrypt(cf,sk,result_msg)
    print("===before_approx_sign===")
    print(list(result_msg)[:50])
    cf.to_device()
    '''
    ## Good 1
    #heaan.math.approx.sign(eval_, cf, cf,numiter_g=24, numiter_f=10)
    ## Good when very small
    #heaan.math.approx.sign(eval_, cf, cf,numiter_g=75, numiter_f=15)
    heaan.math.approx.sign(eval_, cf, cf,numiter_g=24, numiter_f=15)
    check_boot(cf,eval_)
    
    
    eval_.left_rotate(cf,1,cf1)
    check_boot(cf1,eval_)
    ## add , /2 , square
    eval_.add(cf1,cf,cf)
    eval_.mult(cf,0.5,cf)
    check_boot(cf,eval_)
    eval_.square(cf,cf)
    check_boot(cf,eval_)
    ## 1-()
    ## all are 1s
    eval_.negate(cf,cf)
    eval_.add(cf,he_all1,cf)
    '''
    cf.to_host()
    dec.decrypt(cf,sk,result_msg)
    print("===find_position===")
    print(list(result_msg)[:50])
    cf.to_device()
    '''

    eval_.mult(ctxt,cf,ctxt)
    check_boot(ctxt,eval_)


    eval_.mult(ctxt,he_msg_pr2,ctxt)
    check_boot(ctxt,eval_)

    '''
    ctxt.to_host()
    dec.decrypt(ctxt,sk,result_msg)
    print("===before_approx===")
    print(list(result_msg)[:50])
    ctxt.to_device()
    '''

    eval_.bootstrap(ctxt,ctxt)

    ctxtr = HE_log_approx(ctxt, keypack, eval_, log_num_slot, context)
    check_boot(ctxtr,eval_)
    
    '''
    ctxtr.to_host()
    dec.decrypt(ctxtr,sk,result_msg)
    print("===after_approx===")
    print(list(result_msg)[:50])
    ctxtr.to_device()
    '''

    eval_.sub(ctxtr, he_msg_pr3, ctxtr)

    eval_.mult(ctxtr,cf,ctxtr)
    check_boot(ctxtr,eval_)
    ## All 1 except 1st
    eval_.mult(ctxtr,he_all1e1,ctxtr)
    check_boot(ctxtr,eval_)

    tmp2 = heaan.Ciphertext(context)
    ## he_table
    eval_.mult(cf,he_table,tmp2)
    check_boot(tmp2,eval_)
    eval_.add(ctxtr,tmp2,ctxtr)
    check_boot(ctxtr,eval_)

    tmp = heaan.Ciphertext(context)
    # tmp.to_device() 
   
    '''
    ctxtr.to_host()
    dec.decrypt(ctxtr,sk,result_msg)
    print("===before_sumup_ctxtr===")
    print(list(result_msg)[:50])
    ctxtr.to_device()
    '''

    ### Sum up all the values per 25 slots
    eval_.left_rotate(ctxtr,12,tmp)
    check_boot(tmp,eval_)
    eval_.add(ctxtr,tmp,ctxtr)
    check_boot(ctxtr,eval_)

    eval_.left_rotate(ctxtr,6,tmp)
    check_boot(tmp,eval_)
    eval_.add(ctxtr,tmp,ctxtr)
    check_boot(ctxtr,eval_)
    
    eval_.left_rotate(ctxtr,3,tmp)
    check_boot(tmp,eval_)
    eval_.add(ctxtr,tmp,ctxtr)
    check_boot(ctxtr,eval_)

    ## All except 1st three
    eval_.mult(ctxtr,he_all0e3,ctxtr)
    check_boot(ctxtr,eval_)

    eval_.left_rotate(ctxtr,2,tmp)
    check_boot(tmp,eval_)
    eval_.add(ctxtr,tmp,ctxtr)

    eval_.left_rotate(ctxtr,1,tmp)
    check_boot(tmp,eval_)
    eval_.add(ctxtr,tmp,ctxtr)
    
    ## All zero except 1st

    eval_.mult(ctxtr,he_all0e1,ctxtr)
    check_boot(ctxtr,eval_)
  
    '''
    cf.to_host()
    dec.decrypt(cf,sk,result_msg)
    print("===before_sumup_cf===")
    print(list(result_msg)[:50])
    cf.to_device()
    '''

    ### Sum up all the values per 24 slots
    eval_.left_rotate(cf,12,tmp)
    check_boot(tmp,eval_)
    eval_.add(cf,tmp,cf)

    eval_.left_rotate(cf,6,tmp)
    check_boot(tmp,eval_)
    eval_.add(cf,tmp,cf)
    
    eval_.left_rotate(cf,3,tmp)
    check_boot(tmp,eval_)
    eval_.add(cf,tmp,cf)
   
    ## All zero except 1st three
    eval_.mult(cf,he_all0e3,cf)
    check_boot(cf,eval_)

    eval_.left_rotate(cf,2,tmp)
    check_boot(tmp,eval_)
    eval_.add(cf,tmp,cf)

    eval_.left_rotate(cf,1,tmp)
    check_boot(tmp,eval_)
    eval_.add(cf,tmp,cf)
   
    ## All zero except 1st
    eval_.mult(cf,he_all0e1,cf)
    check_boot(cf,eval_)

    heaan.math.approx.inverse(eval_,cf,cf,init=0.1,num_iter=9,greater_than_one=True) ## 1/x+1 
    check_boot(cf,eval_)

    eval_.mult(ctxtr,cf,ctxtr)
    check_boot(ctxtr,eval_) 
    return ctxtr

