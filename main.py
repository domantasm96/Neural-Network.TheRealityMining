import nltk
import scipy.io

'''
### Classifier explanation ###
### 2 categories: sloan or no_sloan ###
### sloan = 1; no_sloan = 0 ###
'''
data = scipy.io.loadmat("realitymining.mat")['s']
affilation = data['my_affil']
my_hours = data['my_hours']
data_mat = data['data_mat']
category = []
# my_predictable = data['my_predictable']
# my_hangouts = data['my_hangouts']
# my_group = data['my_group']
# my_regular = data['my_regular']
# my_predictable = data['my_predictable']
# my_travel = data['my_travel']

affilation_list = []
data_list = []
hours_list = []
for aff in affilation:
    for i in aff:
        print(i)
        if(len(i) > 0):
            affilation_list += i[0][0].tolist()
        else:
            affilation_list += ['None']

print(affilation_list)
print(nltk.FreqDist(affilation_list).most_common())

for mat in data_mat:
    for i in mat:
        if(len(i) > 0):
            data_list += [i[0].tolist()]
        else:
            data_list += [['None']]

for hours in my_hours:
    for i in hours:
        if(len(i) > 0):
            hours_list += i[0][0].tolist()
        else:
            hours_list += ['None']

# Change values: Sloan = 1; no_sloan = 0
for index in range(len(affilation_list)):
	if(affilation_list[index] == 'None'):
		continue
	if(affilation_list[index] == 'sloan' or affilation_list[index] == 'sloan_2'):
		affilation_list[index] = 1
	else:
		affilation_list[index] = 0
print(affilation_list)
print(nltk.FreqDist(affilation_list).most_common())