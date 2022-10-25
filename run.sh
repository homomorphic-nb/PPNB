echo "a1_train/inference"
python3 -c 'import NB_WMain; NB_WMain.a1_train()'
python3 -c 'import NB_WMain; NB_WMain.a1_inference()' 

echo "a2_train/inference"
python3 -c 'import NB_WMain; NB_WMain.a2_train()'
python3 -c 'import NB_WMain; NB_WMain.a2_inference()'

echo "cancer_train/inference"
python3 -c 'import NB_WMain; NB_WMain.cancer_train()'
python3 -c 'import NB_WMain; NB_WMain.cancer_inference()'

echo "car_train/inference"
python3 -c 'import NB_WMain; NB_WMain.car_train()'
python3 -c 'import NB_WMain; NB_WMain.car_inference()'
