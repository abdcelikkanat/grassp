### GraS2P
This is a PyTorch implementation of the paper "Continuous-time Graph Representation with Sequential Survival Process".

### Installation
- Initialize a new conda environment and activate it.
```
conda create -n grassp python=3.11
conda activate grassp
```
- You can install all the required packages by running the following command.
```
pip install -r requirements.txt
```

**Note** : Please visit the [PyTorch website](https://pytorch.org/get-started/locally/) for the installation details of the PyTorch library (version Stable 2.0.0 with possible CUDA 11.7 support) regarding your system.


#### Usage
- You can train the model for the Synthetic-mu dataset with CPU processors by typing the following command:
```
python run.py --edges ../datasets/synthetic_n=100_seed=16/synthetic_n=100_seed=16.edges --model_path ./synthetic_n=100_seed=16.model --device cpu
```
- To see more detailed list of options, please type: 
```
python run.py --help
```
- You can generate the animations for the learned embedding by
```
python animate.py --edges datasets/synthetic_n=100_seed=16/synthetic_n=100_seed=16.edges --model_path synthetic_n=100_seed=16.model --anim_path ynthetic_n=100_seed=16.mp4

