# VisioWell

## Overview
VisioWell is a high-throughput imaging platform to study division synchronicity of stem cells.  
The paper has been submitted and it can be found at bioRxiv: insert link.     
The computational pipeline of VisioWell is designed to automatically retrieve cell division times for stem cells in brightfield images.  
It is composed of three main computational steps:  
(A) Microwell identification and naming  
(B) Cell segmentation and curve correction  
(C) Retrieval of division times  
At the end, (D) statistical analysis are conducted on a notebook to compare synchronicity division across experimental conditions.  
Necessarily, (A), leveraging cupy, has to run in environment different than (B), leveraging pytorch. Dependencies are detailed for each task.

## Local environments 
### Dependencies
The dependencies are set up with poetry, and they can be found in:  
docker/step0 and docker/step1, docker/notebook respectively for tasks: (A); (B), (C); (D).

To create **locally** the environments without using docker, please consider the following codes to insert in the terminal.  
For the environment associated to task (A):  
```bash
conda create -n envstep0 python=3.10 -y
conda activate envstep0
cd "docker/step0"
pip install poetry
poetry config virtualenvs.create false --local
poetry install --no-root
conda install -c conda-forge cupy=13.6 --solver=libmamba -y 
```

For the environment associated to tasks (B, C): 
Uncomment the commented lines to recreate the .txt files 
```bash
conda create -n cellposesam_inference python=3.10 -y
conda activate cellposesam_inference
cd "docker/step1"
#pip install poetry
#poetry self add poetry-plugin-export
#poetry export --without-hashes --format=requirements.txt --output requirements.txt
#findstr /v /i "torch torchvision torchaudio nvidia- triton cuda" requirements.txt > #requirements_filtered.txt
pip install --no-cache-dir -r requirements_filtered.txt
pip install --no-cache-dir torch==2.9.1+cu130 torchvision==0.24.1+cu130 --index-url https://download.pytorch.org/whl/cu130

```

For the environment associated to analysis (D):  
```bash
conda create -n notebookenv python=3.10 -y
conda activate notebookenv
cd "docker/notebook"
pip install poetry 
poetry config virtualenvs.create false --local
poetry install --no-root
```

## Launching the steps separately
Example for step0, after inserting the data to analyse in the data/ folder, choosing a rotation_angle and choosing a results directory where the results will be automatically saved: 
```bash
set PYTHONPATH=src
python -m src.visiowell_pipeline.pipeline.step0_findwells --folder_path data/ --rotation_angle 8 --dir_results "results/reproducibility/MB07082025/Scene009/step0"
```

```bash
set PYTHONPATH=src
python -m src.visiowell_pipeline.pipeline.step1_predictions --path_exp "results/reproducibility/MB07082025/Scene009/step0" --dir_results "results/reproducibility/MB07082025/Scene009/step1"
```


## Docker images
The docker images, containing the isolated environments, are available at the following links:
For task(A), envstep0:
For tasks(B,C), cellposesam_inference:
For analysis(D), notebookenv: 
Docker allows to avoid any manual local environment setting.

### Launching code with Docker images
For task (A), adapt "D:\user\folder" to the folder containing the data folder:
```bash
docker run --rm -it --gpus all -v D:\user\folder:/app/visiowell visiowell-step0:latest bash
poetry run python3 -m src.visiowell_pipeline.pipeline.step0_findwells \
  --folder_path /app/visiowell/data \
  --rotation_angle 0 \
  --dir_results /app/visiowell/results/try1/step0
```

## Nextflow
A nextflow pipeline is set up to be compatible with the local environments (done) and docker (work in progress). 
Nextflow runs one step (A,B,C) after the other and it automatically takes care of selecting the right environment for the task.

## Computational requirements
Overall, GPU acceleration is required.  
On local machines, the code was ran on a GPU NVIDIA RTX 5090.  
Task (A) works with CuPy if a GPU is available, otherwise in numpy. GPU is strongly recommended.  
Task (B) requires a GPU, as segmentation is based on CellPoseSAM.   
Task (C) does not require GPU acceleration.   
Analysis (D) does not require GPU acceleration.

