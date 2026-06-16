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
  - [Preprocessing](#preprocessing)
    - [1 FSL preprocessing](#1-fsl-preprocessing)
      - [Installation (Apptainer)](#installation-apptainer)
      - [Running (Apptainer)](#running-apptainer)
      - [1.1 Brain Extraction](#11-brain-extraction)
      - [1.2 Bias Field Correction](#12-bias-field-correction)
      - [1.3 Non-linear Brain Image Registration](#13-non-linear-brain-image-registration)
      - [1.4 Voxel Normalization](#14-voxel-normalization)
      - [1.5 Registration to isotropic spatial resolution of 2mm](#15-registration-to-isotropic-spatial-resolution-of-2mm)
    - [2 ComBat preprocessing](#2-combat-preprocessing)
  - [Triamese-ViT](#triamese-vit)
      - [Spearman (differentiable) loss](#spearman-differentiable-loss)
    - [1. Training](#1-training)
      - [Setup](#setup)
    - [2. Inference](#2-inference)
  - [References](#references)


## Preprocessing
*"To ensure compatibility and mitigate the potential effects of protocol variability for the different datasets, we applied a standardized preprocessing protocol using FSL 5.10 [37] to the MRI scans. This protocol included several steps: brain extraction [38], bias field correction, nonlinear registration to the MNI standard space, and normalization of voxel values within the brain area by subtracting the mean and dividing by the standard deviation. We also used ComBat harmonization on the datasets to adjust for scanner and site-specific effects while preserving biological variability. After preprocessing, all MRI scans were resized to a voxel dimension of 91 × 109 × 91 with an isotropic spatial resolution of 2 mm."* 

### 1 FSL preprocessing
Running the datasets preprocessing is made through the FSL library. Is possible to run it locally even though I suggest to do it on a HPC system due to its high workload. More infos on the [dedicated section](#installation-apptainer)

[FSL reference](https://fsl.fmrib.ox.ac.uk/fsl/docs/index.html)


| Step in paper                       | FSL tool                                     |
| ----------------------------------- | -------------------------------------------- |
| Brain extraction                    | `bet`                                        |
| Bias field correction               | `fast` (bias correction mode)                |
| Nonlinear registration to MNI space | `fnirt` (usually with `flirt` pre-alignment) |
| Voxel normalization (z-score)       | `fslmaths`                                   |


#### Installation (Apptainer)

- [FSL singularity images](https://singularityhub.github.io/singularityhub-archive/collection/MPIB-singularity-fsl/)
- [Apptainer](https://apptainer.org/docs/user/main/)
- [Singularity compatibility](https://apptainer.org/docs/user/main/singularity_compatibility.html#singularity-command-symlink)

```bash
singularity pull shub://MPIB/singularity-fsl:5.0.10
```

#### Running (Apptainer)
```bash
# Running a shell inside the singularity image
apptainer shell \
--cleanenv \
singularity-fsl_5.0.10.sif
```


> [!Note]
> Set `export OMP_NUM_THREADS=8` to enable **multithreading** (in theory)


#### 1.1 Brain Extraction
[Reference](https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/bet.html)
*"BET (Brain Extraction Tool) deletes non-brain tissue from an image of the whole head. It can also estimate the inner and outer skull surfaces, and outer scalp surface, if you have good quality T1 and T2 input images."*

```bash
#inside the apptainer image
bet <input.nii.gz> <output_name>

# use the `-m` flag in the end to generate the brain mask separately(used after in the pipeline)
```

The `output_name` is extended with `.nii.gz` automatically

#### 1.2 Bias Field Correction 
[Reference](https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/fast.html)


```bash
#inside the apptainer image
#    -b      output estimated bias field
#    -B      output bias-corrected image
#    -S,--channels   number of input images (channels); default 1
#    -o,--out    output basename
fast -b -B -o <files> #the -B flag performs the bias-field correction
```

#### 1.3 Non-linear Brain Image Registration
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

#### 1.4 Voxel Normalization
```bash
# Step 1: Calculate global mean and std across all voxels
mean_val=$(fslstats input.nii.gz -M) #-m all voxels(also zero valued) 
std_val=$(fslstats input.nii.gz -S)

# Step 2: Subtract mean and divide by std
fslmaths input.nii.gz -sub $mean_val -div $std_val output_zscore.nii.gz
```

#### 1.5 Registration to isotropic spatial resolution of 2mm
Using linear interpolation referring to the standard MNI reference image in FSL.

```bash
flirt -in your_T1_image.nii.gz -ref your_T1_image.nii.gz -out your_resampled_T1.nii.gz -applyisoxfm 2
```

### 2 ComBat preprocessing
[original implementation](https://github.com/Jfortin1/ComBatHarmonization)

This is a dataset harmonization done for MRI scans coming from different machinery

## Triamese-ViT
They have used **T1 structural MRI** scans from IXI and ABIDE [(ref)](#references)

I'm removing all the test models used for comparison in the paper.

#### Spearman (differentiable) loss
>[!Warning]
> This loss is not used as described in the paper, it have been used instead the `mse`

The Spearman correlation coefficient tells how samples paired values are correlated in a monotonic but non-linear way.
It actually compare the **rank** for each sample from X to the rank from Y. The rank of a variable is an arbitrary ordering strategy of that variable, which means that exists an ordering relationship in the set (for example age, number of elements etc. etc.)

The ranking operations is non-differentiable, so a small model approximating the ranking function is the adopted solution from the authors.

> [!Warning]Sorter
> If using the ranking aux_loss(default!) is necessary to have the "pretrained SoDeep sorter network weights"


There are 4 kinds of models used as "sorter" part for the Spearman loss:

### 1. Training
#### Setup
run `run_setup.sh` to initialize the workspace for the training

Flags for 
- `Training.py`:
  1. --train_folder
  2. --valid_folder
  3. --sorter 
  4. --excel_path


```bash
# Training for IXI example
python3 Training.py \
--train_folder $IXI_TRAIN_PREPROCESSED \
--valid_folder $IXI_EVAL_PREPROCESSED \
--test_folder $IXI_TEST_PREPROCESSED \
--excel_path IXI.xls
```

- `Prediction`:
  1. --train_folder
  2. --valid_folder
  3. --test_folder
  4. --excel_path

```bash
#put the prediction command 
``` 

### 2. Inference

```bash
IXI_DATASET="../data/IXI_validate" source run_preprocessing.sh 0 1 
```

## References

- **T1 IXI Dataset** 
  - web:  https://brain-development.org/ixi-dataset/
  - data: http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar
  - excel:http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls 


- **FSL**
  - singularity images: https://singularityhub.github.io/singularityhub-archive/collection/MPIB-singularity-fsl/
  - doc: https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index

- **FreeSurfer**
  - doc: https://surfer.nmr.mgh.harvard.edu/fswiki

- **Triamese-ViT**
  - paper : https://ieeexplore.ieee.org/document/11016176
  - gitHub: https://github.com/zhangz59/Triamese-ViT (Original repo)


