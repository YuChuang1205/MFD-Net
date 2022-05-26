#coding=gbk
'''
Created on 2021年6月21日

@author: 余创
'''
from utils import *
import os
import skimage.io
from sklearn import metrics
from scipy import interpolate
from keras.models import load_model



batch_size = 32
root_path = os.path.abspath('.')
data_path = os.path.join(root_path,'data')

model_path = os.path.join(root_path,'model')
make_dir(model_path)

model_out_path = os.path.join(model_path,'match_model_best.hdf5')
model = load_model(model_out_path)



img_file_list =['field.npy','forest.npy','indoor.npy','mountain.npy','oldbuilding.npy','street.npy','urban.npy','water.npy']
label_file_list = ['field_label.npy','forest_label.npy','indoor_label.npy','mountain_label.npy','oldbuilding_label.npy','street_label.npy','urban_label.npy','water_label.npy']
  
  
 
  

fpr95_list = []
AUC_list = []
  
  
for i in range(len(img_file_list)):
    print("加载数据中.........................................")
    img =np.load(os.path.join(data_path,img_file_list[i]))
    label = np.load(os.path.join(data_path,label_file_list[i]))
    print("---------------------------------------------------------")
    print(img.shape)
    print(label.shape)
    img = img/255.0   #归一化
    img_0 = img[:, 0].copy()
    img_1 = img[:, 1].copy()
    img_0 = np.expand_dims(img_0, axis=3)
    img_1 = np.expand_dims(img_1, axis=3)
    loss = model.evaluate([img_0, img_1,img_0, img_1,img_0, img_1,img_0, img_1],[label,label,label,label,label],batch_size=batch_size,verbose=1)
    img_name = img_file_list[i].split('.')[0]
    print("{}测试的loss:{}".format(img_name,loss))
    print("{}测试的accuracy:{}".format(img_name,loss[6]))
    
      
    out = model.predict([img_0, img_1,img_0, img_1,img_0, img_1,img_0, img_1],batch_size=batch_size)
    out = out[0]  

    out0_1 = np.where(out>0.5,1,0)
      
    
    
    #ROC曲线
    fpr,tpr,thresholds = metrics.roc_curve(label,out)
    #指标FPR95
    fpr95 = float(interpolate.interp1d(tpr,fpr)(0.95))
    #指标AUC
    area = metrics.roc_auc_score(label,out)
      
      
      
      
    
    fpr95_list.append(fpr95)
    AUC_list.append(area)
    print("{}测试的fpr95输出:{}".format(img_name,fpr95))
    print("{}测试的AUC输出:{}".format(img_name,area))
      
  
     
      

fpr95_mean = np.mean(fpr95_list)
AUC_mean = np.mean(AUC_list)
  
  
print("所有测试集的Fpr95_mean输出:",fpr95_mean)
print("所有测试集的AUC_mean输出:",AUC_mean)
print("---------------------------------------------------------")
print("Done!!!!!!!!!!!!!!!!!!")
