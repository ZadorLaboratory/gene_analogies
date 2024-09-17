## Bash script for setting up the conda environment
ENV_NAME=$1
#env_name=
env_exists=$(micromamba env list | grep -w "${ENV_NAME}")
if [ -z "$env_exists" ]; then
    # The environment does not exist, so create it
    micromamba create -y --name $ENV_NAME 
fi
micromamba install -y -n $ENV_NAME python=3.11 jupyter -c conda-forge
micromamba install -y -n $ENV_NAME numpy anndata pandas seaborn tqdm -c conda-forge
micromamba install -y -n $ENV_NAME -c pytorch -c nvidia faiss-gpu=1.8.0
micromamba activate $ENV_NAME
