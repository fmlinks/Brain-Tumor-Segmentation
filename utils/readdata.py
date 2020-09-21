import numpy as np
import os

a = os.listdir("./data/HGG")
b = os.listdir("./data/LGG")

print(np.shape(a))
print(np.shape(b))


# read HGG data

path = './data/HGG/'

import nibabel as nib

all_data_HGG = []
modality = ['t1', 't2', 't1ce', 'flair', 'seg']

for j in a:
    name = j
    all_modality_data = []

    for i in modality:
        data = nib.load(path + name + '/' + name + '_' + i + '.nii.gz').get_data()
        all_modality_data.append(data)
        print(i, np.shape(all_modality_data))

    all_modality_data_summary = np.transpose(all_modality_data, (3, 1, 2, 0))
    print(i, np.shape(all_modality_data_summary))

    if j == a[0]:
        all_data_HGG = all_modality_data_summary
    else:
        all_data_HGG = np.concatenate([all_data_HGG, all_modality_data_summary], axis=0)

    print(j, np.shape(all_data_HGG))

print('all_data_HGG', np.shape(all_data_HGG))




# read LGG data


path = './data/LGG/'

import nibabel as nib

all_data_LGG = []
modality = ['t1', 't2', 't1ce', 'flair', 'seg']
for j in b:
    name = j
    all_modality_data = []

    for i in modality:
        data = nib.load(path + name + '/' + name + '_' + i + '.nii.gz').get_data()
        all_modality_data.append(data)
        print(i, np.shape(all_modality_data))

    all_modality_data_summary = np.transpose(all_modality_data, (3, 1, 2, 0))
    print(i, np.shape(all_modality_data_summary))

    if j == b[0]:
        all_data_LGG = all_modality_data_summary
    else:
        all_data_LGG = np.concatenate([all_data_LGG, all_modality_data_summary], axis=0)

    print(j, np.shape(all_data_LGG))

print('all_data_LGG', np.shape(all_data_LGG))





# concat HGG and LGG
import numpy as np
all_data = np.concatenate([all_data_HGG, all_data_LGG], axis=0)
print('all_data', np.shape(all_data))
del all_data_HGG
del all_data_LGG



# shuffle all data

import random
random.shuffle(all_data)



# normalize all data

# modality = ['t1', 't2', 't1ce', 'flair', 'seg']
all_data_t1 = all_data[:,:,:,0]
all_data_t2 = all_data[:,:,:,1]
all_data_t1ce = all_data[:,:,:,2]
all_data_flair = all_data[:,:,:,3]

all_data_seg = all_data[:,:,:,4]
all_data_seg = all_data_seg[:,:,:,np.newaxis]

mean_t1 = np.mean(all_data_t1)
std_t1 = np.std(all_data_t1)
mean_t2 = np.mean(all_data_t2)
std_t2 = np.std(all_data_t2)
mean_t1ce = np.mean(all_data_t1ce)
std_t1ce = np.std(all_data_t1ce)
mean_flair = np.mean(all_data_flair)
std_flair = np.std(all_data_flair)

print('mean_t1', mean_t1)
print('mean_t2', mean_t2)
print('mean_t1ce', mean_t1ce)
print('mean_flair', mean_flair)
print('std_t1', std_t1)
print('std_t2', std_t2)
print('std_t1ce', std_t1ce)
print('std_flair', std_flair)




data_types = ['t1', 't2', 't1ce', 'flair']
norm_dict = {i: {'mean': 0.0, 'std': 1.0} for i in data_types}

norm_dict['t1']['mean'] = mean_t1
norm_dict['t1']['std'] = std_t1
norm_dict['t2']['mean'] = mean_t2
norm_dict['t2']['std'] = std_t2
norm_dict['t1ce']['mean'] = mean_t1ce
norm_dict['t1ce']['std'] = std_t1ce
norm_dict['flair']['mean'] = mean_flair
norm_dict['flair']['std'] = std_flair

print(norm_dict)


all_data_t1_norm = (all_data_t1 - norm_dict['t1']['mean'])/norm_dict['t1']['std']
all_data_t2_norm = (all_data_t2 - norm_dict['t2']['mean'])/norm_dict['t2']['std']
all_data_t1ce_norm = (all_data_t1ce - norm_dict['t1ce']['mean'])/norm_dict['t1ce']['std']
all_data_flair_norm = (all_data_flair - norm_dict['flair']['mean'])/norm_dict['flair']['std']
print(np.shape(all_data_t1_norm))

all_data_t1_norm = all_data_t1_norm[:,:,:,np.newaxis]
all_data_t2_norm = all_data_t2_norm[:,:,:,np.newaxis]
all_data_t1ce_norm = all_data_t1ce_norm[:,:,:,np.newaxis]
all_data_flair_norm = all_data_flair_norm[:,:,:,np.newaxis]

print(np.shape(all_data_t1_norm))
print(np.shape(all_data_t2_norm))
print(np.shape(all_data_t1ce_norm))
print(np.shape(all_data_flair_norm))



all_data = np.concatenate([all_data_t1_norm,
                           all_data_t2_norm,
                           all_data_t1ce_norm,
                           all_data_flair_norm,
                           all_data_seg], axis=3)
print(np.shape(all_data))



# seperate all data to 5 parts

print(np.shape(all_data))
p = all_data.shape[0]//5
print(p)

part0 = all_data[:p, :, :, :]
part1 = all_data[p:2*p, :, :, :]
part2 = all_data[2*p:3*p, :, :, :]
part3 = all_data[3*p:4*p, :, :, :]
part4 = all_data[4*p:, :, :, :]

print(np.shape(part0))
print(np.shape(part1))
print(np.shape(part2))
print(np.shape(part3))
print(np.shape(part4))



# seperate dataset to data and label

X0 = part0[:,:,:,0:4]
Y0 = part0[:,:,:,4]
Y0 = Y0[:,:,:,np.newaxis]

X1 = part1[:,:,:,0:4]
Y1 = part1[:,:,:,4]
Y1 = Y1[:,:,:,np.newaxis]

X2 = part2[:,:,:,0:4]
Y2 = part2[:,:,:,4]
Y2 = Y2[:,:,:,np.newaxis]

X3 = part3[:,:,:,0:4]
Y3 = part3[:,:,:,4]
Y3 = Y3[:,:,:,np.newaxis]

X4 = part4[:,:,:,0:4]
Y4 = part4[:,:,:,4]
Y4 = Y4[:,:,:,np.newaxis]

print('X0', np.shape(X0))
print('Y0', np.shape(Y0))

print('X1', np.shape(X1))
print('Y1', np.shape(Y1))

print('X2', np.shape(X2))
print('Y2', np.shape(Y2))

print('X3', np.shape(X3))
print('Y3', np.shape(Y3))

print('X4', np.shape(X4))
print('Y4', np.shape(Y4))



# save numpy

np.save('X0.npy', X0)
np.save('Y0.npy', Y0)

np.save('X1.npy', X1)
np.save('Y1.npy', Y1)

np.save('X2.npy', X2)
np.save('Y2.npy', Y2)

np.save('X3.npy', X3)
np.save('Y3.npy', Y3)

np.save('X4.npy', X4)
np.save('Y4.npy', Y4)

