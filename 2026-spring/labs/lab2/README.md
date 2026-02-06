# Tutorial 2: Habitat Navigation Demo
##### [Advanced Topics in Embodied Learning and Vision](https://elvcourse.org/)
######  2026-02-03
###### Ellis Brown

Tutorial materials adapted from Chris Hoang's materials from the [SP25 course](https://elvcourse.org/2025/).
Originally derived from https://aihabitat.org/tutorial/2020 and https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html



### 1. Setup Tutorial Materials

1. download 
```bash
# create tutorial2 dir
export T2_DIR=$SCRATCH/elv/tutorial2
mkdir -p $T2_DIR
cd $T2_DIR

# TODO
wget /path/to/course/materials.tar.gz $T2_DIR/

# TODO: untar

```


2. Download Habitat test scenes
```bash
# runs in background, unzips to $T2_DIR/data
cd $T2_DIR
(wget -q https://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip && unzip -q habitat-test-scenes.zip) &
# wait # (optionally wait)
```

3. Clone `habitat-lab`
```bash
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git $T2_DIR/habitat-lab
```

4. Transfer Habitat demo files to the habitat data dir
```bash
wait # ensure bg download/unzip has finished

cp $T2_DIR/habitat-demo.json.gz $T2_DIR/data/datasets/pointnav/habitat-test-scenes/v1/train
```

5. Create EGL vendor config for GPU rendering in Singularity

Singularity containers don't have access to the host's EGL vendor configs, so we need to provide one for NVIDIA GPU rendering to work properly.

```bash
cat > $T2_DIR/10_nvidia.json << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
```

> **Note**: The `10_` prefix follows the EGL vendor library naming convention (lower numbers = higher priority). This file tells the EGL loader where to find NVIDIA's driver.





### 2. Create Singularity Overlay w/ Conda Env
See https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda for more details

1. create singularity overlay
```bash
cd $SCRATCH
cp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz .
gunzip overlay-15GB-500K.ext3.gz 
mv overlay-15GB-500K.ext3 elv.ext3
```

2. ***TIP:***
create a reusable script to launch the overlay in `RW` mode. eg, create a file `singularity_rw_elv.sh` in your `$SCRATCH` dir with the following contents:
```bash
#!/bin/bash

# path to your overlay
OVERLAY=$SCRATCH/elv.ext3

echo "Launching Singularity Overlay in rw mode: $OVERLAY" 
singularity exec --nv \
    --overlay $OVERLAY:rw \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif  /bin/bash

echo "Exited Singularity Overlay: $OVERLAY"
```

3. launch the container in `RW` mode, download conda, create an environment for `habitat`

```bash
# launch container in RW mode
bash $SCRATCH/singularity_rw_elv.sh

# download miniforge 3
wget --no-check-certificate https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3

# Initialize conda for current shell (if not already done)
source /ext3/miniforge3/etc/profile.d/conda.sh

# create habitat env
conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat

# 1. Install habitat-sim via conda (this handles the tricky dependencies including gym)
conda install habitat-sim withbullet -c conda-forge -c aihabitat -y

# 2. Install other dependencies manually
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy==1.26.4" "pillow==10.4.0" "scipy==1.13.1"

# 3. Install habitat-lab and habitat-baselines directly (from the cloned repo)
cd $T2_DIR/
pip install -e habitat-lab/habitat-lab
pip install -e habitat-lab/habitat-baselines
```

4. ***while still in the container***,
create an env wrapper script `/ext3/env.sh` inside the overlay with the following contents 
> ***TIP:*** you can open it in an editor using
> ```bash
> nano /ext3/env.sh
> ```

```bash
#!/bin/bash
echo "sourcing /ext3/env.sh"

unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh

# 1. Export base bin first (so it's available as a fallback)
export PATH=/ext3/miniforge3/bin:$PATH

# 2. Activate your environment LAST (so it prepends your env bin to the very front)
conda activate habitat
export LD_LIBRARY_PATH=/.singularity.d/libs:$LD_LIBRARY_PATH
```

> ***NOTE:** exporting the `LD_LIBRARY_PATH` will help solve some missing package issues*


5. test the environment
```bash
conda deactivate
source /ext3/env.sh

# test the environment
cd $T2_DIR
conda activate habitat
MAGNUM_LOG="verbose" python -c "
import habitat_sim
backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = 'data/scene_datasets/habitat-test-scenes/skokloster-castle.glb'
agent_cfg = habitat_sim.agent.AgentConfiguration()
cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
try:
    sim = habitat_sim.Simulator(cfg)
    print('SUCCESS: Simulator initialized with GPU!')
    print(f'GPU Device: {sim.gpu_device}')
except Exception as e:
    print(e)
"
```



### 3. Setup Jupyter Notebook Kernel

This section configures a Jupyter kernel so you can use your Singularity+Conda environment in Open OnDemand notebooks.

> See [NYU HPC: OOD with Conda/Singularity](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/open-ondemand-ood-with-condasingularity) for more details.

#### Prerequisites
Make sure `ipykernel` is installed in your conda environment (inside the Singularity container):
```bash
conda activate habitat
conda install ipykernel -y
```

#### 1. Copy the kernel template

Run the following commands **on Greene** (not inside Singularity):
```bash
mkdir -p ~/.local/share/jupyter/kernels
cd ~/.local/share/jupyter/kernels
cp -R /share/apps/mypy/src/kernel_template ./habitat
cd ./habitat
```

This creates a kernel directory with the following files:
```
kernel.json  logo-32x32.png  logo-64x64.png  python
```

#### 2. Edit the `python` wrapper script

The `python` file is a wrapper script that Jupyter uses to launch your Singularity container.

Edit `~/.local/share/jupyter/kernels/habitat/python` and locate the `singularity exec` command at the bottom. Update it to match your overlay and `.sif` file:

```bash
singularity exec $nv \
  --overlay /scratch/$USER/elv.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; $cmd $args"
```

> **IMPORTANT:** Make sure the overlay path (`/scratch/$USER/elv.ext3`) and the `.sif` image path match the ones you used in Section 2.


#### 3. Edit `kernel.json`

Edit `~/.local/share/jupyter/kernels/habitat/kernel.json` to point to your wrapper script.

**Change from:**
```json
{
  "argv": [
    "PYTHON_LOCATION",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "KERNEL_DISPLAY_NAME",
  "language": "python"
}
```

**To (replace `<NetID>` with your NetID):**
```json
{
  "argv": [
    "/home/<NetID>/.local/share/jupyter/kernels/habitat/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "habitat",
  "language": "python"
}
```

#### 4. Launch Jupyter on Open OnDemand

1. Go to [https://ood.hpc.nyu.edu](https://ood.hpc.nyu.edu)
2. Select **Interactive Apps** → **Jupyter Notebook**
3. Configure your job (select GPU resources if needed) and launch
4. Once the notebook starts, select the **habitat** kernel from the kernel dropdown
