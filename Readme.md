Authors: 

Yun Song,
Dr.,
School of Computer and Communication Engineering, 
Changsha University of Science and Technology
Email: sonie@126.com

Cong Jiang,
Master Degree Candidate,
School of Computer and Communication Engineering, 
Changsha University of Science and Technology
Email: jc2428@qq.com

=====================================================================\
Overview:

This package contains the implemention of algorithm TARGCN.  

To run the code： on the TARGCN/model
"python Run_xx.py --mode train/test --test_para epoch_xx"
Run_xx.py: Run_03.py to run PEMS03 dataset, Run_D4.py to run PEMSD4 dataset, Run_D8.py to run PEMSD8 dataset
--mode : train or test TARGCN
--test_para : Specifies the name of the saved model parameter file

To modify the hyperparameters you need to change the parameters in 'TARGCN/model/conf/PEMSD4.conf' 
, "TARGCN/model/conf/PEMSD8.conf" and 'TARGCN/model/conf/PEMS03.conf'.
  
The code is associated with the following paper: 
Yun Song, Cong Jiang, Fan Wendong, Deng Zelin, Bai Xinke, "TARGCN: Temporal 
Attention Recurrent Graph Convolutional Neural Network for Traffic Prediction," 
   
Please cite this paper if you use this code. 
 
For further information, please contact: sonie@126.com and jc2428@qq.com

=====================================================================

Copyright (C) 2022 Yun Song, Cong Jiang, Fan Wendong, Den Zelin,Bai Xinke

=====================================================================
