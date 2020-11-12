import glob
import cv2
import numpy as np
import os
import math
import vg

data_path = 'data/BHP_Raw/TapData/P1'
window_size = 5
validation_split = 0.8

label_list = sorted(glob.glob(data_path+"/3d/*.npy")) 

for ind, l in enumerate(label_list):
    labels = np.load(l)[-300:]
    _, f = os.path.split(l)
    sess_name = f[:-8]
    img_dir = data_path + '/frames/' + sess_name
    img_list = sorted(glob.glob(img_dir+'/*.png'))
    imgs = []
    for i in img_list:
        frame = cv2.imread(i,cv2.IMREAD_UNCHANGED)
        frame = cv2.resize(frame,(224,224))
        imgs.append(frame)
    imgs = np.array(imgs)
    imgs = imgs.reshape(-1,window_size,  imgs.shape[1],imgs.shape[2],imgs.shape[3])
    labels = labels.reshape(-1,window_size,labels.shape[1])
    if ind == 0:
        x = imgs
        y = labels
    else:
        x = np.concatenate((x,imgs))
        y = np.concatenate((y,labels))
    print(ind,x.shape,y.shape)

def preprocess_angle(y):

    temp = y[:,4,:] 
    temp = temp.reshape(-1,25,3)
    temp = temp[:,5:,:]
    y=[]
    for f in temp:
        for i in range(4): #finger
            for j in range(1,4):
                v1 = f[5*i+j] - f[5*i+j-1] #get previous bone vector
                v2 = f[5*i+j+1] - f[5*i+j] #get current bone vector
                if j == 1:  #MCP
                    angle = vg.signed_angle(v1, v2, look=np.array([0,-1,0]))
                    y.append(angle)
                angle = vg.signed_angle(v1, v2, look=np.array([-1,0,0]))
                y.append(angle)
    y = np.array(y).reshape(-1,16)
    return y
#     #temp[0]

final_y = preprocess_angle(y)
train_x = x[:int(len(x)*validation_split)]/255.0
train_y = final_y[:int(len(final_y)*validation_split)]/90.0
test_x = x[int(len(x)*validation_split):]/255.0
test_y = final_y[int(len(final_y)*validation_split):]/90.0

print(train_x.shape,train_y.shape)
print(test_x.shape,test_y.shape)

cond_name = data_path.split('/')[-2]
usr_name = data_path.split('/')[-1]

np.savez('data/train/'+cond_name +'_'+usr_name+'_window5.npz',train_x,train_y)
np.savez('data/test/'+cond_name +'_'+usr_name+'_window5.npz',test_x,test_y)
