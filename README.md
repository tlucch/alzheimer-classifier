# MRI Preprocessing

## Introduction

Before creating a model, MRI images need to be preprocessed. The main reason behind this is that MRI images have a lot of information that is not necessary for the classification model and that can even affect the classification. For example, MRI images are not images of just the brain, they also have the skull, the eyes and other tissues. In addition, MRI machines generate noise in the images, which medical professionals know they have to ignore, but machines do not, so it's important to remove it before training a model.

## Important information

We will be using software that is external to python which is not entirely compatible with Windows. So it will be better if we use Linux or MacOS. In case you need to use Windows yes or yes, you will need to install the Windows Subsystem for Linux (WSL) which is a feature of Windows that allows developers to run a Linux environment without the need for a separate virtual machine or dual booting; and do all the working within this environment and not in windows. This means, for example, that all commands will not be run in the powershell terminal but on the ubuntu terminal that is installed when WSL is installed; that the root folder has to be in cloned in the WSL environment and not in a Windows folder; that if you are using Visual Studio Code, you will also need to install the WSL extension, etc.

The installation and setup of WSL can be found here: https://learn.microsoft.com/en-us/windows/wsl/install

## Folder Tree

Before starting with the preprocessing steps is important to have the folders organized in the following way:

```
+-- Root
|   +-- ADNI #Folder containing all the ADNI files and the .csv with metadata
|   +-- atlas #Folder containing the altas reference file
|   +-- preprocessing
		|   +-- compile_adni.py
		|   +-- register.py
		|   +-- skull_strip.py
		|   +-- bias_correct.py
|   ...
```

So, in our root folder we have three folders: ADNI and preprocessing. Inside the ADNI folder we have to put all our NiFTi (MRI) files and the .csv that contains the metadata of each file (specially the class label). Inside the preprocessing, we have our 4 preprocessing python files, one for each step in our pipeline. Finally, inside the atlas folder we will put our atlas reference file for the affine registration (the explantation on this is found on step 2).

## Requirements

For the preprocessing, we need to install some dependencies:

```
glob2==0.4.1
matplotlib==3.8.0
nipype==1.8.6
```

You can install them manually or simply run the following command in the terminal:

```
pip install -r requirements_preprocessing.txt
```

## Steps

1. Separate NiFTi files into subdirectories according to class label
2. Perform affine registration
3. Skull stripping
4. N4 enhanced bias correction

### Step 1: Compile ADNI

The first step in our preprocessing pipeline consists in matching NiFTi files to the master metadata csv file provided by ADNI and divide each of them into sudirectories with their corresponding class label (AD, MCI, CN). For that, we first have to open our `compile_adni.py` file with our favorite IDE and replace at line 12 the `“introduce csv file name here”` with the name of the master metadata csv file.

Now we can run the python file, just by introducing in the terminal:

```
python compile_adni.py
```

After running this file, a new folder will be created inside the root folder called `data`. Inside this folder we can find a new folder called `ADNI` which has 3 subdirectories (AD, CN, MCI), each of them containing the corresponding NiFTi files already segregated.

## Step 2: Register

Our next step in the preprocessing stage is to perform the affine registration transformation. The goal of affine registration is to find the affine transformation that best maps one data set (e.g., image, set of points) onto another. In many situations data is acquired at different times, in different coordinate systems, or from different sensors. Such data can include sparse sets of points and images both in 2D and 3D, but the concepts generalize also to higher dimensions and other primitives. Registration means to bring these data sets into alignment, i.e., to find the “best” transformation that maps one set of data onto another, here using an affine transformation.

#### Installing and Setting up FSL

The first thing we have to do to perform this transformation is to install FSL. FSL is a comprehensive library of analysis tools for FMRI, MRI and diffusion brain imaging data created by the University of Oxford. You can find the installation process for MacOS, Linux and Windows (WSL) here: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Windows

After completing the installation, make sure that FSL is set up in your PATH environment variables. If not set the FSL bin folder path into the PATH environment variables.

#### Performing the transformation

Now that we have FSL already installed, we can perform the affine registration. For this, we will use two of the FSL tools: `fslreorient2std` and `flirt`. The first one is a simple tool designed to reorient an image to match the orientation of the standard template images (MNI152) so that they appear "the same way around" in FSLeyes. The later is a fully automated robust and accurate tool for linear (affine) intra- and inter-modal brain image registration. 

FSL doesn't work with python, its run thought the terminal. So, for example if we wanted to run the `fslreorient2std` tool on one of our files we should open a terminal a right the following command:

```
fslreorient2std <source file path> <destination path>
```

We only use python so we don’t have to type one by one the files and run them one by one. But the only thing we are doing in python is running the commands in the terminal automatically. Moreover, we can use the multiprocess functions in python so that our CPU uses all its cores and processes multiple files at once.

To perform the transformation, we have to perform two easy tasks. The first one is to search for our reference NiFTi file for the affine registration and put it in the atlas folder. FSL comes with lots of reference images for this transformation, which our found in the following directory: `FSL/data/standard`. We have to select the MNI152 that goes with the type of MRI we are working with. In my case, all my MRI files were 1mm T1 MRIs, so I selected the MNI152_T1_1mm.nii file, but if for example your nifti files are 2mm T1 you should use the MNI152_T1_2mm.nii file. Once the file is inside the atlas folder, open the `register.py` file in the preprocessing folder and in line 66 change `“name of MNI152 file”` with the name of the atlas reference file you selected. In my case it should be `“MNI152_T1_1mm.nii”`.

Once this is done, we can now run the `register.py` file by introducing in the terminal:

```
python register.py
```

This will create a new folder inside the data folder called `ADNIReg` which has 3 subdirectories (AD, CN, MCI), each of them containing the corresponding NiFTi files already transformed.

### Step 3: Skull Stripping

MRI scans are not only of the brain, they come will lots of other information. Like the skull, the eyes, the nose, the tongue, etc., etc., etc. Although, we humans, are able to know that all that visual information has to be ignored and just focus on the brain, computers don't have this ability so lots of unnecessary information is feed into the machine learning model which can lead to a more extense training and a loss in the model accuracy. That's why one of the steps in our preprocessing is performing skull stripping, which is the process of removing all this unnecessary information and just keeping the brain.

In order to perform this, we will use another tool from FSL called `bet`. As we said before, the FSL tools are run in the terminal so we will use python to perform it automatically in all files and speeding up things using multiprocessing.

In this step we don’t to perfom any special task, just run the file using the following command on the terminal:

```
python skull_strip.py
```

This will use all the files in the `data/ADNIReg` folder, perform skull stripping to them and store them in a new folder inside the data directory called `ADNIBrain`. As before, inside we can find 3 subdirectories (AD, CN, MCI), each of them containing the corresponding NiFTi files.

### Step 4: Bias Correction

N4 bias correction is a technique used to eliminate intensity inconsistencies, often referred to as "bias fields," from medical imaging. A medical professional is capable of ignoring or identify this bias or inconsistencies, but a machine no, so they might arise from different sources like variations in scanner settings or the patient's physical structure. Such bias fields can greatly impact the precision and dependability of image analysis. Correcting these fields enhances the overall image clarity and minimizes the impact of external variables on the analysis process.

To perform this bias correction, we will use a library called Advanced Normalization Tools (ANTs), which is a very popular toolkit for processing medical images. The installation process for Linux/WSL and MacOS can be found in the following link: https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS.  

Once ANTs is installed and  the path to the bin folder is already set into the PATH environment variable, we can proceed to perform the N4 bias correction. For this, we just have to run the `bias_correct.py` file in the same way we runned the other files. Keep in mind that this transformation is the most expensive one computationally, so thats why you can choose to modify a little bit the python file so that it runs in batches and not all at once. For this just unncoment lines 72 to 87, comment lines 89 to 93 and change the batches variable to generate the amount of batches you want. Take into consideration that you need to know the total amount of files and subtract one. So, for example, if you want to make batches that process 200 files at a time and your total amount of files is 1615 lines 72 to 93 should look like this:

```python
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
            pool.map(unwarp_bias_field_correction, paras)

'''
if __name__ == '__main__':
    # Multi-processing
    paras = zip(data_src_paths, data_dst_paths)
    pool = Pool(processes=cpu_count())
    pool.map(unwarp_bias_field_correction, paras)'''
```

if you want to run it all at once, just keep everything as it is.

# Building Alzheimer Classificator

## Introduction

Now that we have our preprocessing done, we can start creating our classificator. As we have seen in the preprocessing we are working with 3 different classes: Alzheimers Disease (AD), Mid Cognitive Impairment (MCI) and Control (CN). The issue with building an MRI classificator is that the data comes in 3D format and almost all convolutional neural networks are build to receive 2D data as input. Luckily for us we can use MONAI library to help us with this issue as it has lots of 3D convolution neural networks. In this case we will use MONAI’s DenseNet 121.

## Requirements

The first step in our journey is to install the necessary requirements:

```
matplotlib==3.8.0
monai==1.3.0
nipype==1.8.6
numpy==1.26.1
torch==2.1.0
scikit-learn==1.3.2
```

You can install them manually or simply run the following command in the terminal:

```
pip install -r requirements_nn.txt
```

After installing the necessary requirements, we will need to install CUDA, which is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs). In other words, it allows us to GPU for our training. We will be ussing version 11.6 which can be installed following these instructions: https://developer.nvidia.com/cuda-11-6-0-download-archive

We also need to install cudNN (CUDA for neural networks) which is another NVIDIA ilbrary that alows us to use cuda for neural networks. Here you can find the explanation on how to install it: https://developer.nvidia.com/cudnn

## Building the trainer

Now that we have everything ready, we can start building our trainer! The first stpe is to import the necessary libraries:

```python
import logging
import os
import sys
import shutil
import tempfile

import matplotlib.pyplotas plt
import torch
from torch.utils.tensorboardimport SummaryWriter
import numpyas np
from sklearn.model_selectionimport train_test_split
from sklearn.metrics import accuracy_score

import monai
from monai.appsimport download_and_extract
from monai.configimport print_config
from monai.dataimport DataLoader, ImageDataset
from monai.transformsimport (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)

pin_memory= torch.cuda.is_available()
device= torch.device("cuda"if torch.cuda.is_available()else "cpu")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()
```

Next we have to set the data directory. In our case all our files are located in the `data/ADNIDenoise` folder so we write the following:

```python
parent_directory **=** r"data/ADNIDenoise"
```

The next step is to create a list of the paths to each file and a list of its corresponding label:

```python
dir_labels= ["AD", "MCI", "CN"]
images= []
labels= []

label_count= 0

for iin dir_labels:
    dir_path= os.path.join(parent_directory, i)
    file_names= os.listdir(dir_path)
for fin file_names:
        images.append(os.path.join(dir_path, f))
for jin range(0, len(file_names)):
        labels.append(label_count)
    label_count+= 1
```

Now we can create a training, a validating and a testing data set and make the necessary transformations to it. For that, we will write the following lines:

```python
# Define transforms
train_transforms= Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])
val_transforms= Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

#divide train, test and validate
train_paths, temp_paths, train_labels, temp_labels= train_test_split(images, labels, test_size=0.4, random_state=42)
valid_paths, test_paths, valid_labels, test_labels= train_test_split(temp_paths, temp_labels, test_size=0.5, random_state=42)
```

Afterward, we need to create our DataLoaders. A `DataLoader` is a Pytorch data primitive that allow you to use pre-loaded datasets as well as your own data. `DataLoader` wraps an iterable to enable easy access to the samples. We will create a `DataLoader` for our training and validation sets:

```python
# create a training data loader
train_ds= ImageDataset(image_files=train_paths, labels=train_labels, transform=train_transforms)
train_loader= DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=pin_memory)

# create a validation data loader
val_ds= ImageDataset(image_files=valid_paths, labels=valid_labels, transform=val_transforms)
val_loader= DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=pin_memory)
```

Now its time to set up our DenseNet model and, in case you have a pre-trained model or you want to continue training an existing model, just uncomment the last two lines and add the path to your .pth file:

```
model= monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels= 3).to(device)
#checkpoint= torch.load("path to your .pth")
#model.load_state_dict(checkpoint)
```

As we said before, we are using DenseNet 121, but MONAI comes with lots of different neural networks. Here you can find all the available models: https://docs.monai.io/en/stable/networks.html#nets . If you want to use any of them just change `.DenseNet121` with the model you would like to use and adapt the hyperparameters that go with the model you choose.

Finally, we can start our training. Write the following code and adjust the parameters according to your case, like the amount of epochs:

```python
# Create DenseNet121, CrossEntropyLoss and Adam optimizer
loss_function= torch.nn.CrossEntropyLoss()

optimizer= torch.optim.Adam(model.parameters(), 1e-4)

# start a typical PyTorch training
val_interval= 2
best_metric=-1
best_metric_epoch=-1
epoch_loss_values= []
metric_values= []
writer= SummaryWriter()
max_epochs= 20

for epochin range(max_epochs):
    print("-"* 10)
    print(f"epoch {epoch+ 1}/{max_epochs}")
    model.train()
    epoch_loss= 0
    step= 0

for batch_datain train_loader:
        step+= 1
        inputs, labels= batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs= model(inputs)
        loss= loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss+= loss.item()
        epoch_len= len(train_ds)// train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len* epoch+ step)

    epoch_loss/= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch+ 1} average loss: {epoch_loss:.4f}")

if (epoch+ 1)% val_interval== 0:
        model.eval()

        num_correct= 0.0
        metric_count= 0
for val_datain val_loader:
            val_images, val_labels= val_data[0].to(device), val_data[1].to(device)
with torch.no_grad():
                val_outputs= model(val_images)
                value= torch.eq(val_outputs.argmax(dim=1), val_labels)
                metric_count+= len(value)
                num_correct+= value.sum().item()

        metric= num_correct/ metric_count
        metric_values.append(metric)

if metric> best_metric:
            best_metric= metric
            best_metric_epoch= epoch+ 1
            torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
            print("saved new best metric model")

        print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
        print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
        writer.add_scalar("val_accuracy", metric, epoch+ 1)

print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()
```

This code will run all the epochs and store the one that has the best metric into a .pth file in the root folder called "best_metric_model_classification3d_array.pth". You can run this code as many times as you want in order to keep training the model.

## Testing

In order to test our model accuracy, we first need to create a DataLoader for the test dataset, so we run the following code:

```python
test_ds **=** ImageDataset(image_files**=**test_paths, labels**=**test_labels, transform**=**val_transforms)
test_loader **=** DataLoader(test_ds, batch_size**=**2, num_workers**=**2, pin_memory**=**pin_memory)
```

Then we load the .pth file we want to test:

```python
model.load_state_dict(torch.load("best_metric_model_classification3d_array.pth"))
model.eval()
```

And finally, we test our model accuracy:

```
predictions= []
ground_truth= []

for test_datain test_loader:
    test_images, test_labels= test_data[0].to(device), test_data[1].to(device)
with torch.no_grad():
        test_outputs= model(test_images)
# Perform any post-processing on the test_outputs if neededpredictions.extend(test_outputs.cpu().numpy())# Assuming you want to store the predictionsground_truth.extend(test_labels.cpu().numpy())

# Convert the list of predictions and ground truth to numpy arrays if not already done
predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

accuracy = accuracy_score(ground_truth, predictions.argmax(axis=1))
print(f"Test Accuracy: {accuracy:.4f}")
```
