
from __future__ import print_function

import os
import sys
import shutil
from multiprocessing import Pool, cpu_count
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def unwarp_bias_field_correction(arg, **kwarg):
    return bias_field_correction(*arg, **kwarg)

def bias_field_correction(src_path, dst_path):
    print("N4ITK on: ", src_path)
    try:
        n4 = N4BiasFieldCorrection()
        n4.inputs.input_image = src_path
        n4.inputs.output_image = dst_path

        n4.inputs.dimension = 3
        n4.inputs.n_iterations = [100, 100, 60, 40]
        n4.inputs.shrink_factor = 3
        n4.inputs.convergence_threshold = 1e-4
        n4.inputs.bspline_fitting_distance = 300
        res = n4.run()
        print(res.outputs)
        
    except RuntimeError:
        print("\tFailed on: ", src_path)

    return

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
data_dir = os.path.join(parent_directory, 'data')

data_src_dir = os.path.join(data_dir, 'ADNIBrain')
data_dst_dir = os.path.join(data_dir, 'ADNIDenoise')
data_labels = ['AD', 'CN', 'MCI']

data_src_paths, data_dst_paths = [], []
for label in data_labels:
    src_label_dir = os.path.join(data_src_dir, label)
    dst_label_dir = os.path.join(data_dst_dir, label)
    if not os.path.exists(dst_label_dir):
        create_dir(dst_label_dir)
    else:
        if len(os.listdir(dst_label_dir))>0:
            cont = input("DESTINTATION FOLDER IS NOT EMPTY!!!!! CHECK IF YOUR ARE RUNNING THE CORRECT FILE. Do you want to continue (Y/N)?").upper()
            while cont != "Y" and cont != "N":
                cont = input("Invalid input. Please enter Y for yes and N for no. Do you want to continue (Y/N)?").upper()
            if cont == "N":
                exit(1)
            else:
                shutil.rmtree(dst_label_dir)
                create_dir(dst_label_dir)
        else:    
            shutil.rmtree(dst_label_dir)
            create_dir(dst_label_dir)

    for subject in os.listdir(src_label_dir):
        data_src_paths.append(os.path.join(src_label_dir, subject))
        data_dst_paths.append(os.path.join(dst_label_dir, subject))

# Test
#bias_field_correction(data_src_paths[0], data_dst_paths[0])

'''
batches = [0,200,400,600,800,1000,1200,1400,1614]

if __name__ == '__main__':
    # Multi-processing
    for i in range(0, len(batches)-1):
        cont = input(f"Starting processing from {batches[i]} to {batches[i+1]}. Do you want to continue (Y/N)?").upper()
        while cont != "Y" and cont != "N":
            cont = input("Invalid input. Please enter Y for yes and N for no. Do you want to continue (Y/N)?").upper()
        if cont == "N":
            print(f"last known processed file: {batches[i]}")
            exit(1)
        else:
            paras = zip(data_src_paths[batches[i]:batches[i+1]], data_dst_paths[batches[i]:batches[i+1]])
            pool = Pool(processes=cpu_count())
            pool.map(unwarp_bias_field_correction, paras)'''

if __name__ == '__main__':
    # Multi-processing
    paras = zip(data_src_paths, data_dst_paths)
    pool = Pool(processes=cpu_count())
    pool.map(unwarp_bias_field_correction, paras)