# Privacy Preserving Naive Bayesian Classifier

#### This repository contains code and data for "Fully Homomorphic Privacy-Preserving Naive Bayes Machine Learning and Classification"

- run.sh : shell script for running the main
- NB_WMain.py : main code for training and classfy each data
- NB_WModule.py : functions for encrypt, train, inference 
- NB_log.py : approximate logarithm function
- data_devide.py : devide the test data 

## Pi-heaan is the simulator version of HEaaN used in the paper
### Please install pi-heaan before running the code.
```console
pip install pi-heaan
```

# Procedure to training and inference the data

1. Install pi-heaan
2. Devide the test data : change the directory name in 'data_devide.py'
```console
python3 data_devide.py
```
3. Run shell script
```console
sh run.sh
``` 
