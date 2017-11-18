import scipy.io
import numpy as np

data = scipy.io.loadmat("realitymining.mat")['s']
affilation = data['my_affil']
data_mat = data['data_mat']

'''
### Classifier explanation ###
### 2 categories: sloan or no_sloan ###
### sloan = 1; no_sloan = 0 ###
'''
affilation_list = []
matdata_list = []
for i in range(len(affilation[0])):
    if len(affilation[0][i]) > 0 and len(data_mat[0][i]) > 0:
        affilation_list += [affilation[0][i][0][0]]
        matdata_list += [data_mat[0][i].tolist()]
        if affilation_list[len(affilation_list)-1] == 'sloan' or affilation_list[len(affilation_list)-1] == 'sloan_2':
            affilation_list[len(affilation_list)-1] = [1]
        else:
            affilation_list[len(affilation_list)-1] = [0]
print(affilation_list)

# 1 – home, 2 – work, 3 – elsewhere, 0 – no signal, NaN – phone is off
home, work, elsewhere, no_signal, phone_off = 0, 0, 0, 0, 0
work = 0
elsewhere = 0
no_signal = 0
phone_off = 0
frequency = []
all_frequency = []
for subject in range(len(matdata_list)):
    for hours in range(24):
        for elements in range(len(matdata_list[subject][hours])):
            if matdata_list[subject][hours][elements] == 1:
                home += 1
            elif matdata_list[subject][hours][elements] == 2:
                work += 1
            elif matdata_list[subject][hours][elements] == 3:
                elsewhere += 1
            elif matdata_list[subject][hours][elements] == 0:
                no_signal += 1
            else:
                phone_off += 1
        frequency += [home/len(matdata_list[subject][hours]) if home !=0 else 0, 
                     work/len(matdata_list[subject][hours]) if  work !=0 else 0, 
                     elsewhere/len(matdata_list[subject][hours]) if elsewhere !=0 else 0, 
                     no_signal/len(matdata_list[subject][hours]) if no_signal !=0 else 0,
                     phone_off/len(matdata_list[subject][hours]) if phone_off !=0 else 0]
        home, work, elsewhere, no_signal, phone_off = 0, 0, 0, 0, 0
    all_frequency += [frequency]
    frequency = []
features = np.array(all_frequency)
print(features)