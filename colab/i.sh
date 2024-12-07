!apt install -q git
!pip install -q condacolab
import condacolab
condacolab.install()
!mamba create -n min_resolution sage plotly python=3.12.5 -c conda-forge -q
!git clone https://github.com/Issyl-m/MinimalResolution.git && cd MinimalResolution

# run this block on a separate cell
%%shell
eval "$(conda shell.bash hook)" # copy conda command to shell
conda activate min_resolution
pip install jsons

./g.sh
