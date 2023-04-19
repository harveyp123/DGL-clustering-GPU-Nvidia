# DGL-clustering
An example for DGL cluster/subgraph manipulation

### 1. Download and install the CUDA 11.6 environment, add the environment to path
#### 1.1. Download the cuda 11.6 library, you can refer to https://developer.nvidia.com/cuda-toolkit-archive, refer to "runfile (local)" for download link
```bash
mkdir cuda_install
cd cuda_install
wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run
```

#### 1.2 Install the cuda 11.6 library to ```/opt/cuda-11.6``` path
```bash
sudo sh cuda_11.6.2_510.47.03_linux.run --silent --toolkit --installpath=/opt/cuda-11.6 --override
```

#### 1.3 Set up environment variables:
To switch between CUDA versions, you'll need to update your environment variables. You can create a shell script or an alias to easily switch between versions.

Create a script (e.g. “switch_cuda.sh”):
```bash
#!/bin/bash
export CUDA_HOME=/opt/cuda-$1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Make the script executable:
```bash
chmod +x switch_cuda.sh
```

Now, you can easily switch between CUDA versions by running:
```bash
source switch_cuda.sh <version>
```

In this example, you need to activate CUDA 11.6:
```bash
source switch_cuda.sh 11.6
```

#### 1.4 Verify installation:
To check if the correct version is active, run:
```bash
nvcc --version
```
This should display the currently active CUDA version.

### 2. Steps for building an environment for both torch+tensorflow+dgl+pyg

#### 2.1. Create a new conda environment
```bash
conda create --prefix ${HOME}/.conda/envs/torch_tf_pyg_dgl python=3.8
```

#### 2.2. Activate the environment
```bash
conda activate torch_tf_pyg_dgl 
```

#### 2.3. Install pytorch on CUDA 11.6 (please have CUDA 11.6 install)
```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

#### 2.4 Install the pyg library according to https://github.com/pyg-team/pytorch_geometric
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu116.html 
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu116.html 
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
pip install torch-geometric
```

#### 2.5 Install the dgl and tensorflow:
```bash
conda install -c "dglteam/label/cu116" dgl
pip install tensorflow==2.12.*
```

### 3. Run the example: 
```bash
python GNN_partition_dgl.py
```
The code will create a ```test``` folder to contain the partitioned graph output. 

### 4. Run the GCoD code: 
Reference: https://github.com/GATECH-EIC/GCoD
Some minor package compatibility issues are fixed. 
```bash
cd GCoD
bash GCN_cora.sh
```

### 5. Run the pyg example:
2-layer GCN on Cora, Citeseer and PubMed datasets
Reference: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py
