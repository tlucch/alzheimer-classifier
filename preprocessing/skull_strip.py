from __future__ import print_function

import os
import sys
import subprocess
import shutil
from multiprocessing import Pool, cpu_count

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def bet(src_path, dst_path, frac="0.5"):
    command = ["bet", src_path, dst_path, "-R", "-f", frac, "-g", "0"]
    subprocess.call(command)
    return

def unwarp_strip_skull(arg, **kwarg):
    return strip_skull(*arg, **kwarg)

def strip_skull(src_path, dst_path, frac="0.4"):
    print("Working on :", src_path)
    try:
        bet(src_path, dst_path, frac)
    except RuntimeError:
        print("\tFailed on: ", src_path)

    return

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
data_dir = os.path.join(parent_directory, 'data')

data_src_dir = os.path.join(data_dir, 'ADNIReg')
data_dst_dir = os.path.join(data_dir, 'ADNIBrain')
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
# strip_skull(data_src_paths[0], data_dst_paths[0])

if __name__ == '__main__':
    # Multi-processing
    paras = zip(data_src_paths, data_dst_paths)
    pool = Pool(processes=cpu_count())
    pool.map(unwarp_strip_skull, paras)