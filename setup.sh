export CUDA_HOME="/usr/local/cuda-11.8"
pip install torch==2.0.1+cu118 torchvision xformers --extra-index-url https://download.pytorch.org/whl/cu118
pip install diffusers[torch]==0.27.2
pip install ipympl triton transformers 

# Install local dependencies in editable mode
pip install -e ./third_party/Mask2Former
pip install -e ./third_party/ODISE
pip install -e ./third_party/meshplot
pip install -e ./third_party/stablediffusion
pip install -e ./third_party/featup
pip install -e ./third_party/dift
pip install pythreejs torch-tb-profiler

# diff3f dependencies
CUDA_HOME=/usr/local/cuda-11.8 FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# DiffusionNet dependencies
pip install trimesh rtree "pyglet<2" plyfile meshio robust_laplacian potpourri3d pywavefront

# ensure some versions are compatible
pip install pytorch-lightning==1.9.5 kornia==0.7.2 pillow==9.3.0 transformers==4.27.0 matplotlib==3.9.3
pip install jupyter jupyterlab jupyter_contrib_nbextensions notebook==6.5.6 # jupyter notebook commit hook
pip install igraph==0.11.5 # future verions dont allow integer as vertex names
pip install pymeshlab==2023.12.post2
pip install numpy==1.24.1 # needs to be <2
pip install huggingface-hub==0.25.2
pip install -e .

# install pre-commit hook
cp pre-commit .git/hooks
chmod +x .git/hooks/pre-commit

