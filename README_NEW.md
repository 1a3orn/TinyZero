


# Install 

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate

conda create -n zero python=3.9
conda activate zero

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip install ray

pip install flash-attn --no-build-isolation
pip install wandb IPython matplotlib



