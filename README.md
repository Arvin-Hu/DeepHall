<p align="center">
  <img src=".github/img/deephall.svg" width="200">
</p>
<h1 align="center">DeepHall</h1>

Simulating the fractional quantum Hall effect (FQHE) with neural network variational Monte Carlo.

This repository contains the codebase for the paper [Describing Landau Level Mixing in Fractional Quantum Hall States with Deep Learning](https://doi.org/10.1103/PhysRevLett.134.176503). If you use this code in your work, please [cite our paper](CITATIONS.bib).

Currently, DeepHall supports running simulations with spin-polarized electrons on a sphere and has been tested with 1/3 and 2/5 fillings.

## Installation

DeepHall requires Python `>=3.11` and JAX `0.4.35`. It is highly recommended to install DeepHall in a separate virtual environment.

```bash
# Remember to activate your virtual environment
git clone https://github.com/bytedance/DeepHall
cd DeepHall
pip install -e .                  # Install CPU version
pip install -e ".[cuda12]"        # Download CUDA libraries from PyPI
pip install -e ".[cuda12_local]"  # Or, use local CUDA libraries
```

To customize JAX installation, please refer to the [JAX documentation](https://jax.readthedocs.io/en/latest/installation.html).

## Performing Simulations

### Command Line Invocation

You can use the `deephall` command to run FQHE simulations. The configurations can be passed to DeepHall using the `key=value` syntax (see [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-a-dot-list)). A simple example would be:

```bash
deephall 'system.nspins=[6,0]' system.flux=15 optim.iterations=100
```

In this example, we place 6 electrons on a sphere with a total flux $2Q=15$ through the spherical surface. The radius of the sphere is implicitly set as $\sqrt{Q}=\sqrt{15/2}$. This configuration corresponds to 1/3 filling. (Remember that the particle–flux relation on the sphere geometry is $2Q = N / \nu - \mathcal{S}$, where $\mathcal{S}=3$ for 1/3 filling.) The energy output includes only the kinetic part and the electron–electron interactions.

If you just want to test the installation, an even simpler example is the non-interacting case with a smaller network and batch size:

```bash
deephall 'system.nspins=[3,0]' system.flux=2 system.interaction_strength=0 optim.iterations=100 network.psiformer.num_layers=2 batch_size=100
```

Details of available settings are available at [config.py](deephall/config.py).

### Python API

You can also use DeepHall from your Python script. For example:

```python
from deephall import Config, train

config = Config()
config.system.nspins = (3, 0)
config.system.flux = 2
config.system.interaction_strength = 0.0
config.optim.iterations = 100
config.network.psiformer.num_layers = 2
config.batch_size = 100

train(config)
```

## Output

By default, the results directory is named like `DeepHall_n3l2_xxxxxx_xx:xx:xx`. You can configure the output location with the `log.save_path` config, which can be any writable path on the local machine or a remote path supported by [universal_pathlib](https://github.com/fsspec/universal_pathlib).

In the results directory, the file you will need most of the time is `train_stats.csv`, which contains the energy, angular momentum, and other useful quantities per step. The checkpoint files like `ckpt_000099.npz` store Monte Carlo walkers and neural network parameters so that the wavefunction can be analyzed, and the training can be resumed.

## Wavefunction Analysis with NetObs

DeepHall contains a `netobs_bridge` module to calculate the pair correlation function, overlap with the Laughlin wavefunction, and the one-body reduced density matrix. With [NetObs](https://github.com/bytedance/netobs) installed:

```bash
# Energy
netobs deephall unused energy --with steps=2000 --net-restore save_path/ckpt_000099.npz --ckpt save_path/energy
# Overlap
netobs deephall unused deephall@overlap --with steps=50 --net-restore save_path/ckpt_000099.npz --ckpt save_path/overlap
# Pair correlation function
netobs deephall unused deephall@pair_corr --with steps=100000 --net-restore save_path/ckpt_000099.npz --ckpt save_path/pair_corr
# 1-RDM
netobs deephall unused deephall@one_rdm --with steps=20000 --net-restore save_path/ckpt_000099.npz --ckpt save_path/1rdm
```

## Adding a New Neural Network Wavefunction

To add a custom neural network wavefunction, follow these steps:

### Step 1: Create the Network Implementation

Add a new file in the `deephall/networks/` directory, e.g., `deephall/networks/mynet.py`. You can refer to the existing implementation in `deephall/networks/psiformer.py` as a template.

### Step 2: Configure the Network

Update the configuration file `deephall/config.py`:
- Define a new dataclass. Create a dataclass `MyNet` to store the configurations specific to your network. For example:
  ```python
  @dataclass
  class MyNet:
      hidden_dim: int = 128
      num_layers: int = 3
  ```
- Add the dataclass to the `Network` config. Include your dataclass in the `Network` configuration by adding a line like:
  ```python

  @dataclass
  class Network:
      ...
      mynet: MyNet = field(default_factory=MyNet)
  ```
- Extend the `NetworkType` enum. Add a new entry in the `NetworkType` enum to identify your network, e.g.:
  ```python
  class NetworkType(StrEnum):
      ...
      mynet = "mynet"
  ```
### Step 3: Register the Network

Add a construction function in `deephall/networks/__init__.py`. Register your network by adding a conditional block to instantiate it based on the `NetworkType`. For example:
 ```python
if network.type == NetworkType.mynet:
    return MyNet(network.mynet.hidden_dim, network.mynet.num_layers)
```

For more details, commit [d5dc18c](https://github.com/bytedance/DeepHall/commit/d5dc18c) serves as an example for adding a new network.

## Citing Our Paper

If you use this code in your work, please cite the following paper:

```bib
@article{PhysRevLett.134.176503,
  title = {Describing Landau Level Mixing in Fractional Quantum Hall States with Deep Learning},
  author = {Qian, Yubing and Zhao, Tongzhou and Zhang, Jianxiao and Xiang, Tao and Li, Xiang and Chen, Ji},
  journal = {Phys. Rev. Lett.},
  volume = {134},
  issue = {17},
  pages = {176503},
  numpages = {8},
  year = {2025},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.134.176503},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.134.176503}
}
```
