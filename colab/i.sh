!apt install -q git
!pip install -q condacolab
import condacolab
condacolab.install() !mamba install sage plotly -c conda-forge -q
!git clone https://github.com/Issyl-m/MinimalResolution.git && cd MinimalResolution

%%shell
eval "$(conda shell.bash hook)" # copy conda command to shell
conda activate min_resolution
pip install jsons

./g.sh
