from __future__ import division

import sys
import glob2
import os
import shutil
import pandas as pd

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
master_dir = os.path.join(parent_directory, 'ADNI')
master_list = os.path.join(master_dir, 'introduce csv file name here')
write_dir = os.path.join(parent_directory, 'data')
label_dict = ['AD', 'MCI', 'CN']

if __name__ == '__main__':
    adni_data = glob2.glob(master_dir + '/**/*.nii')
    # We use only Scaled and not Scaled_2 images
    adni_data_scaled = [i for i in adni_data if 'Scaled_2' not in i]

    adni_list = pd.read_csv(master_list, skipinitialspace=True, dtype=str)

    completed = 0
    total = len(adni_data_scaled)
    for adni in adni_data_scaled:
        img_id = adni.split('_')[-1].split('.')[0]
        
        relevant_rows = adni_list[adni_list['Image Data ID'] == img_id]
        if (len(relevant_rows) != 1):
            print(img_id)
            print('Mismatch error')
            completed += 1
            continue
        
        label = relevant_rows['Group'].values[0]
        if (label not in label_dict):
            print(img_id)
            print(label)
            print('Label not found')
            completed += 1
            continue
        
        file_root = adni.split('/')[-1]
        curr_dir = os.path.join(write_dir, label, file_root)

        shutil.copyfile(adni, curr_dir)
        completed += 1
        print('Completed ' + str(completed / total * 100) + '%')


    

