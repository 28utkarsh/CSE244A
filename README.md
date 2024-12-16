# CSE 244A - Final Project

This project assumes that you have installed python3 in your system.

### Step 1: Setup the environment variable and clone the repository

```bash
export REPO_ROOT=~/CSE244A
git clone https://github.com/28utkarsh/CSE244A.git $REPO_ROOT
```

### Step 2: Download and setup the dataset

```bash
cd $REPO_ROOT
kaggle competitions download -c ucsc-cse-244-a-2024-fall-final-project
unzip ucsc-cse-244-a-2024-fall-final-project.zip
```

### Step 3: Install dependencies

```bash
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install timm pandas
```

### Step 4: Train the model

```bash
cd $REPO_ROOT
python3 train.py
```

### Step 5: Run inference

To run inference from the pretrained checkpoint, please download the checkpoint from this [link](https://drive.google.com/drive/folders/1ajQU7UMafcidQoq8j1LXVPApmekDm5Qt?usp=drive_link).

```bash
cd $REPO_ROOT
python3 inference.py
```
