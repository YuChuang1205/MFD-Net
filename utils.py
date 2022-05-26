#coding=gbk
'''
Created on 2021年6月21日

@author: 余创
'''
import os 
import numpy as np 





#判断目录是否存在，不存在则创建
def make_dir(path):
    if os.path.exists(path)==False:
        os.makedirs(path)
        
       



