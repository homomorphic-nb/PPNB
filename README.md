# Privacy-Preserving Naive Bayesian Classifier

This repository contains code and data for "Fully homomorphic privacy-preserving Naive Bayes machine learning and classification"  
 
It is impossible to open HEaaN code to the public, these codes are re-constructed by using pi-heaan.  

Pi-heaan is the simulator version of HEaaN.  

Some subroutines in HEaaN is not programmed in pi-heaan, therefore, we programmed by using pi-heaan with the same process.  

Costs such as communication and time might be different which is represented in the paper.  
-----------------------------------------------------------------------------
## Brief explanation of codes
- run.sh : shell script for running the main
- NB_WMain.py : main code for training and classfy each data
- NB_WModule.py : functions for encrypt, train, inference 
- NB_log.py : approximate logarithm function
- data_devide.py : devide the test data 

## Procedure to training and inference the data

1. Install pi-heaan
```console
pip install pi-heaan
```
2. Devide the test data : change the directory name in 'data_devide.py'
```console
python3 data_devide.py
```
3. Run shell script
```console
sh run.sh
``` 
