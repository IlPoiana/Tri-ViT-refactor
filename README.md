# Tri-Vit Refactor
A reconstruction of the Tri-Vit ( see [references](#references)) model for brain age estimation.
The original repo is missing:
- The environment used
- The model weights
- The preprocessing pipeline

Also different models trained and used only in the comparison made in the paper are left in the codebase, resulting in useless libraries and packages import for the actual Tri-Vit model.
This project objective is:
1. Reconstruct Tri-ViT preprocessing pipeline
2. Train Tri-ViT and test it 

## Table of Contents
- [Tri-Vit Refactor](#tri-vit-refactor)
  - [Table of Contents](#table-of-contents)
  - [1. Preprocessing (FSL)](#1-preprocessing-fsl)
    - [Dataset split setup](#dataset-split-setup)
    - [Installation (Apptainer)](#installation-apptainer)
    - [Running (Apptainer)](#running-apptainer)
    - [1.1 Brain Extraction](#11-brain-extraction)
    - [1.2 Bias Field Correction](#12-bias-field-correction)
    - [1.3 Non-linear Brain Image Registration](#13-non-linear-brain-image-registration)
    - [1.4 Voxel Normalization](#14-voxel-normalization)
    - [1.5 Registration to isotropic spatial resolution of 2mm](#15-registration-to-isotropic-spatial-resolution-of-2mm)
  - [2. Triamese-ViT](#2-triamese-vit)
      - [2.1 Setup](#21-setup)
      - [Workspace setup](#workspace-setup)
      - [Excel](#excel)
      - [Spearman (differentiable) loss](#spearman-differentiable-loss)
    - [2.2 Training](#22-training)
      - [Execution](#execution)
  - [3. Future Works](#3-future-works)
  - [References](#references)
      - [T1 IXI Dataset](#t1-ixi-dataset)
      - [ABIDE-II Dataset](#abide-ii-dataset)
      - [FSL](#fsl)
      - [FreeSurfer](#freesurfer)
      - [Triamese-ViT](#triamese-vit)


## 1. Preprocessing (FSL)
*"To ensure compatibility and mitigate the potential effects of protocol variability for the different datasets, we applied a standardized preprocessing protocol using FSL 5.10 [37] to the MRI scans. This protocol included several steps: brain extraction [38], bias field correction, nonlinear registration to the MNI standard space, and normalization of voxel values within the brain area by subtracting the mean and dividing by the standard deviation. We also used ComBat harmonization on the datasets to adjust for scanner and site-specific effects while preserving biological variability. After preprocessing, all MRI scans were resized to a voxel dimension of 91 × 109 × 91 with an isotropic spatial resolution of 2 mm."* 

Under the `preprocessing/` directory there are `run_preprocessing.sh` and `cluster_preprocessing.sh`, for single process and multi-process(slurm cluster) preprocessing respectively

### Dataset split setup 
The samples tipically are not divided into the (train,eval,test) datasets, you can run:
```bash
python3 split_scripts.py \
--source-dir /path/to/dataset
```

to generate under the `./data` directory three sub directories: `IXI_train`, `IXI_eval`, `IXI_test` used later for the model training.
This can be done also after the preprocessing tuning the input/output directories of the scripts

```bash

# - Cluster example -
# The output is stored in a `preprocessed/` dir and also logs are printed for each job spawned
FILES_PER_JOB=20 sbatch cluster_preprocessing.sh "../data/IXI_train"

# - Single job example -
# `run_preprocessing.sh` applies the complete FSL preprocessing pipeline to MRI images:
# Usage: ./run_preprocessing.sh <START_INDEX> <NUM_FILES>
# Default: processes 1 file starting from index 0
# You can set the IXI_DATASET variable to specify the input directory (defaults to ../data/IXI_test)

cd preprocessing
IXI_DATASET="../data/IXI_train" ./run_preprocessing.sh 0 5

```

Running the datasets preprocessing is made through the FSL library. Is possible to run it locally even though I suggest to do it on a HPC system due to its high workload. More infos on the [dedicated section](#installation-apptainer)

[FSL reference](https://fsl.fmrib.ox.ac.uk/fsl/docs/index.html)


| Step in paper                       | FSL tool                                     |
| ----------------------------------- | -------------------------------------------- |
| Brain extraction                    | `bet`                                        |
| Bias field correction               | `fast` (bias correction mode)                |
| Nonlinear registration to MNI space | `fnirt` (usually with `flirt` pre-alignment) |
| Voxel normalization (z-score)       | `fslmaths`                                   |


### Installation (Apptainer)

- [FSL singularity images](https://singularityhub.github.io/singularityhub-archive/collection/MPIB-singularity-fsl/)
- [Apptainer](https://apptainer.org/docs/user/main/)
- [Singularity compatibility](https://apptainer.org/docs/user/main/singularity_compatibility.html#singularity-command-symlink)

```bash
singularity pull shub://MPIB/singularity-fsl:5.0.10
```

### Running (Apptainer)
```bash
# Running a shell inside the singularity image
apptainer shell \
--cleanenv \
singularity-fsl_5.0.10.sif
```


> [!Note]
> Set `export OMP_NUM_THREADS=8` to enable **multithreading** (in theory)


### 1.1 Brain Extraction
[Reference](https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/bet.html)
*"BET (Brain Extraction Tool) deletes non-brain tissue from an image of the whole head. It can also estimate the inner and outer skull surfaces, and outer scalp surface, if you have good quality T1 and T2 input images."*

```bash
#inside the apptainer image
bet <input.nii.gz> <output_name>

# use the `-m` flag in the end to generate the brain mask separately(used after in the pipeline)
```

The `output_name` is extended with `.nii.gz` automatically

### 1.2 Bias Field Correction 
[Reference](https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/fast.html)


```bash
#inside the apptainer image
#    -b      output estimated bias field
#    -B      output bias-corrected image
#    -S,--channels   number of input images (channels); default 1
#    -o,--out    output basename
fast -b -B -o <files> #the -B flag performs the bias-field correction
```

### 1.3 Non-linear Brain Image Registration
[Reference](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/index.html)

Image registration is done for standardizing different images to a reference one by applying transformations (geometrical) to them. Non-linear image registration is a class of tranformation algorithms that uses non-linear transformations together with the linear for registering images.
FLIRT (FMRIB's linear Image Registration Tool) is a tool for high quality linear brain image registration. We will use FNIRT which is the non-linear counterpart.

> [!Note]
> Remember that **fnirt is not diffeomorphic**, which means that is: "A diffeomorphic mapping from a space *U* to a space *V* is one which has exactly one position in *U* for each position in *V*, which also means that it is invertible"

**What image to use as a reference?**
*"use a standard template brain image that represents the average anatomy of a population, such as the MNI (Montreal Neurological Institute) template or others derived from neuroimaging datasets."*

**Brain Atlases**: *"Brain atlases provide spatial reference systems for neuroscience that allow navigation, characterisation and analysis of information based on anatomical location."* [cit.](https://www.humanbrainproject.eu/en/science-development/focus-areas/brain-atlases/)

> [!Note]
> FSL provides under `data/atlases` different standard atlases for brain registration

[FSL atlases](https://fsl.fmrib.ox.ac.uk/fsl/docs/other/datasets.html?h=atlas) 


```bash
fnirt --in=input_image --config=T1_2_MNI152_2mm.cnf --iout=
# fnirt --ref=target_image --in=input_image
# --config=T1_2_MNI152_2mm.cnf
# --ref=$FSLDIR/data/standard/MNI152_T1_2mm_brain_mask.nii.gz not necessary with configuration loaded
# --iout=
```
- $\lambda$ "fudge factor" is an hyperparameter that has to be found empirically, high values (Like 30) move the registration away from the reference instead lower values typically set it closer
- `--config=my_config_file` is for loading a configuration file. They encurage the use of one among the presents in the library
  - "When you specify --config=my_file (i.e., without explicit path or extension) fnirt will search for ./my_file, ./my_file.cnf, ${FSLDIR}/etc/flirtsch/my_file and ${FSLDIR}/etc/flirtsch/my_file.cnf, in that order, and use the first one that is found."
- Execution tricks:
  - Setting the env variable `OMP_NUM_THREADS=8` enables the multhithread of the op 
  - `--subsamp` number and resolution of sub-sampling registration. Higher number of values(4 to 8 ex.) and lower resolution factors (1 minimum) means slower.
  - `--miter` max number of non-linear iterations, lower means faster

More details about fnirt [here](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/user_guide.html#principles)

We can use FNIRT or the newer [MMORF](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/mmorf.html#installing-mmorf), MMORF is not available in FSL 5.0.10 but is faster and does a more precise registration

### 1.4 Voxel Normalization
```bash
# Step 1: Calculate global mean and std across all voxels
mean_val=$(fslstats input.nii.gz -M) #-m all voxels(also zero valued) 
std_val=$(fslstats input.nii.gz -S)

# Step 2: Subtract mean and divide by std
fslmaths input.nii.gz -sub $mean_val -div $std_val output_zscore.nii.gz
```

### 1.5 Registration to isotropic spatial resolution of 2mm
Using linear interpolation referring to the standard MNI reference image in FSL.

> [!Note]
> This step is needed due to the precedent interpolation to 1mm MNI space, **if you choose the 2mm MNI reference ATLAS(the same used in this pipeline) when doing the** [non-linear registration](#13-non-linear-brain-image-registration) **you can skip this step**
>
> To do it check the `run_preprocessing`(remove this step and change the input-output destination directory) bash file and the `my_fnirt.cfg`(change the MNI ref)

```bash
flirt -in your_T1_image.nii.gz -ref your_T1_image.nii.gz -out your_resampled_T1.nii.gz -applyisoxfm 2
```

## 2. Triamese-ViT
They have used **T1 structural MRI** scans from IXI and ABIDE [(ref)](#references)

I'm removing all the test models used for comparison in the paper.  

#### 2.1 Setup
#### Workspace setup
A conda environment is provided through the `tri-vit-environment.yml` file.

> [!Warning]
> The environment is a reconstruction of what have been used in the paper codebase. It has been working for the development and testing of this project but serves more as a guideline of the required libraries  

#### Excel
Datasets metadata is set through an excel file, which must contain at least:
- subject (univoque)id, matching the one into the dataset
- subject age

An example(and also what has been used for this project) is available on the [IXI](#t1-ixi-dataset) dataset.

For using a custom excel file, inside `load_data.py` is possible to add a new `DatasetName` type and insert the new logic in the `__init__` section, to match the expected dataloader format. (IXI dataset is already present and taken as default value)

#### Spearman (differentiable) loss
>[!Warning]
> This loss is not used as described in the paper, it have been used instead the `mse`

The Spearman correlation coefficient tells how samples paired values are correlated in a monotonic but non-linear way.
It actually compare the **rank** for each sample from X to the rank from Y. The rank of a variable is an arbitrary ordering strategy of that variable, which means that exists an ordering relationship in the set (for example age, number of elements etc. etc.)

The ranking operations is non-differentiable, so a small model approximating the ranking function is the adopted solution from the authors.

> [!Warning]Sorter
> If using the ranking aux_loss(default!) is necessary to have the "pretrained SoDeep sorter network weights"


There are 4 kinds of models used as "sorter" part for the Spearman loss:

### 2.2 Training
The `Training.py` file can be considered the entry point of this project. It (1)train from scratch a model then (2) saves the best one(based on the eval set) and finally (3) test it on the test set and returns the model performance


#### Execution
Main flags for `Training.py`:
1. `--train_folder `
2. `--valid_folder `
3. `--test_folder  `
4. `--sorter       `
5. `--excel_path   `

```bash
#to populate the MACRO passed as argument to the next program or if you want to use other data paths just be sure to create the `training_loss/` directory at the same depth level as Training.py
# --- 
bash run_setup.sh 
# ---

# Training for IXI example
python3 Training.py \
--train_folder $IXI_TRAIN_PREPROCESSED \
--valid_folder $IXI_EVAL_PREPROCESSED \
--test_folder $IXI_TEST_PREPROCESSED \
--excel_path IXI.xls
```

## 3. Future Works
- ABIDE-II integration and testing
- [ComBat harmonization](https://github.com/Jfortin1/ComBatHarmonization) between datasets (see [Tri-ViT](#triamese-vit))
- Attention map and occlusion analysis code refactor from the original codebase (respectively `mask.py` and `Attention_map_new_try.py`) 

## References

#### T1 IXI Dataset 
  - web:  https://brain-development.org/ixi-dataset/
  - data: http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar
  - excel:http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls 

#### ABIDE-II Dataset
  - web: https://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html
  - data: download each source file separately, there is a available a univoque `csv` file

#### FSL
  - singularity images: https://singularityhub.github.io/singularityhub-archive/collection/MPIB-singularity-fsl/
  - doc: https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index

#### FreeSurfer
  - doc: https://surfer.nmr.mgh.harvard.edu/fswiki

#### Triamese-ViT
  - paper : https://ieeexplore.ieee.org/document/11016176
  - gitHub: https://github.com/zhangz59/Triamese-ViT (Original repo)


