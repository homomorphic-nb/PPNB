import pandas as pd
import os
csv_file_path4 = './a1_data/'
save_path = csv_file_path4+'test/'
try:
    os.makedirs(name=save_path, mode=0o775, exist_ok=True)
except Exception as e:
    print("[Error] Could not make train table directory: ", e)

tmp = pd.read_csv(csv_file_path4+'a1_test.csv')

for i in range(0,len(tmp)):
    csv_path =save_path +'a1_test'+str(i)+'.csv'
    tmp.iloc[[i]].to_csv(csv_path,index=False)
