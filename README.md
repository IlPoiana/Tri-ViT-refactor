## Objectives
- Reconstruct Tri-ViT preprocessing pipeline
- Train Tri-ViT and test it 
- Understand FreeSurfer and find areas of integration with Tri-ViT 

**Free Surfer tasks mapping to DL**
For doing a segmentation or a region classification task, probably is better to see stuff like YOLO or SAM

I can see instead a better outcome for directly extracting a statistical map and see how it performs.
Maybe a joint training from freeSurfer output as ground truth?


## Preprocessing
*"To ensure compatibility and mitigate the potential effects of protocol variability for the different datasets, we applied a standardized preprocessing protocol using FSL 5.10 [37] to the MRI scans. This protocol included several steps: brain extraction [38], bias field correction, nonlinear registration to the MNI standard space, and normalization of voxel values within the brain area by subtracting the mean and dividing by the standard deviation. We also used ComBat harmonization on the datasets to adjust for scanner and site-specific effects while preserving biological variability. After preprocessing, all MRI scans were resized to a voxel dimension of 91 × 109 × 91 with an isotropic spatial resolution of 2 mm."* 

### 1 FSL preprocessing
*"This protocol included several steps: brain extraction [38], bias field correction, nonlinear registration to the MNI standard space, and normalization of voxel values within the brain area by subtracting the mean and dividing by the standard deviation."*

TO DO
Easier to implement to harder, more complex handling to easier handling
1. Isolated apptainer/singularity running FSL, called from a dedicated bash/python script
  1. Singularity with FSL (and ComBat?)
  2. Script to run in apptainer that does the preprocessing 
  3. Script to run the apptainer for all the images 

**Installation**
[FSL singularity images](https://singularityhub.github.io/singularityhub-archive/collection/MPIB-singularity-fsl/)
```bash
singularity pull shub://MPIB/singularity-fsl:5.0.10
```

**Running (Apptainer)**
```bash
apptainer shell \
--cleanenv \
singularity-fsl_5.0.10.sif

## Cluster
srun --job-name=1488762 apptainer shell \
--cleanenv \
singularity-fsl_5.0.10.sif
```


> [!Note]
> Set `export OMP_NUM_THREADS=8` to enable **multithreading**(in theory)


[FSL reference](https://fsl.fmrib.ox.ac.uk/fsl/docs/index.html)
1. Brain extraction
2. Bias field correction
3. Non-linear registration to the MNI standard space
4. Voxel normalization (mean and std. dev)

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

TO REMOVE
```bash
cp $FSLDIR/etc/flirtsch/T1_2_MNI152_2mm.cnf ./preprocessing && \
cp $FSLDIR/data/standard/MNI152_T1_2mm_brain_mask.nii.gz ./preprocessing
```



#### 1.4 Voxel Normalization


| Step in paper                       | FSL tool                                     |
| ----------------------------------- | -------------------------------------------- |
| Brain extraction                    | `bet`                                        |
| Bias field correction               | `fast` (bias correction mode)                |
| Nonlinear registration to MNI space | `fnirt` (usually with `flirt` pre-alignment) |
| Voxel normalization (z-score)       | `fslmaths`                                   |

```bash
# Step 1: Calculate global mean and std across all voxels
mean_val=$(fslstats input.nii.gz -M) #-m all voxels(also zero valued) 
std_val=$(fslstats input.nii.gz -S)

# Step 2: Subtract mean and divide by std
fslmaths input.nii.gz -sub $mean_val -div $std_val output_zscore.nii.gz
```

#### 1.5 Linear interpolation for Isotropic spatial res 2mm
```bash
flirt -in your_T1_image.nii.gz -ref your_T1_image.nii.gz -out your_resampled_T1.nii.gz -applyisoxfm 2
```

---


**Other dependencies (idk if I need those)**

*Python*
- wxPython: The fsl.utils.idle module has functionality  to schedule functions on the wx idle loop.
- indexed_gzip: The fsl.data.image.Image class can use indexed_gzip to keep large compressed images on disk instead of decompressing and loading them into memory.
- trimesh/rtree: The fsl.data.mesh.TriangleMesh class has some methods which use trimesh to perform geometric queries on the mesh.
- Pillow: The fsl.data.bitmap.Bitmap class uses Pillow to load image files.

*Non-python*
- The fsl.data.dicom module requires the presence of Chris Rorden's dcm2niix program.
- The rtree library assumes that libspatialindex is installed on your system.
- The fsl.transform.x5 module uses h5py, which requires libhdf5.


**Neurodocker for building singularity images** 

```bash
neurodocker generate singularity \
    --pkg-manager apt \
    --base-image debian:bullseye \
    --fsl version=5.0.10 \
    --user nonroot > fsl_singularity

sudo singularity build fsl_5.0.10.sif fsl_singularity
```

### 2 ComBat preprocessing
This is a Dataset 

[original implementation](https://github.com/Jfortin1/ComBatHarmonization)
Harmonization done for scans coming from different machinery

## Triamese-ViT
They have used **T1 structural MRI** scans from IXI and ABIDE[(ref)](#references)

I'm removing all the useless models because I'm getting a lot of interferences with the dependencies


#### Spearman (differentiable) loss
>[!Warning]
> This loss is not used as described in the paper, it have been used instead the `mse`

The Spearman correlation coefficient tells how samples paired values are correlated in a monotonic but non-linear way.
It actually compare the **rank** for each sample from X to the rank from Y. The rank of a variable is an arbitrary ordering strategy of that variable, which means that exists an ordering relationship in the set (for example age, number of elements etc. etc.)

The ranking operations is non-differentiable, so a small model approximating the ranking function is the adopted solution from the authors.

> [!Warning]sorter
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
python3 Training.py --train_folder $IXI_TEST_PREPROCESSED --valid_folder $IXI_EVAL_PRE --excel_path IXI.xls
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

## TO REMOVE

**Setup for testing**
```bash
export SAMPLE_NAME="IXI012-HH-1211-T1.nii.gz"
export IXI_TEST_PRE="$(pwd)/data/IXI_test_pre"
export SAMPLE_PATH="${IXI_TEST_PRE}/voxel/${SAMPLE_NAME}"
```

```bash
IXI_DATASET="../data/IXI_validate" source run_preprocessing.sh 0 1 
```

## References

T1 IXI Dataset (581 healthy subjects)
- web:  https://brain-development.org/ixi-dataset/
- data: http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar
- excel:http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls 

NeuroDocker
- gitHub: https://github.com/ReproNim/neurodocker
- doc   : https://repronim.org/neurodocker/user_guide/examples.html 

FSL
- doc: https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index

FreeSurfer
- doc: https://surfer.nmr.mgh.harvard.edu/fswiki

Triamese-ViT
- paper : https://ieeexplore.ieee.org/document/11016176
- gitHub: https://github.com/zhangz59/Triamese-ViT

web: 
data:


----

USEFUL COMMANDS

rsync -v -a --exclude="*.pdf" /home/emanuele/SIV_project/project emanuele.poiana@baldo.disi.unitn.it:/home/emanuele.poiana/SIV_project/

Visualize slices of the MRI 

```bash
export SAMPLE_PATH="data/IXI_test_pre"
#fast
python visualize.py $SAMPLE_PATH/fast/IXI012-HH-1211-T1_restore.nii.gz $SAMPLE_PATH/images/
#fnirt
python visualize.py $SAMPLE_PATH/fnirt/IXI012-HH-1211-T1.nii.gz $SAMPLE_PATH/images/
```


MEETING 1
- Multi soggetto, ok taglia i dataset filtra per singolo soggetto
- Riassumi doc di FreeSurfer.
- Allena Triamese-ViT con i dati del paper


MEETING 2

- TriVit: richiede preprocessing in FSL, non ho i parametri usati per il preprocessing
- TriVit: non ho il modello pre-trainato
- Free Surfer: differenti entry point di ViT?

MEETING 3
1. Preprocessing, fatto come indicato, ma non mi sembra super preciso, si potrebbe:
  - Tenere 1mm e passare tutte le immagini(più pesante da processare)
  - Tenere 1mm, e passare metà delle immagini
  - Non fare la seconda registrazione ma solo la prima
2. Training



TO DO 
part 1
1. crea una pipeline per estrarre le immagini ViT-ready

part 2
1. Traina ViT per vedere se impara qualcosa
2. Testa Freesurfer e classifica gli outputs
part 3 definisci con il prossimo meeting

Env creation

1. conda create -n tri-vit python=3.10
2. conda install -c intel mkl==2024.0.0 (downgrade for torch compatibility)
3. conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
4. conda install opencv matplotlib scikit-learn scipy pandas
5. conda install tensorboardX -c conda-forge

torch installation. 
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
