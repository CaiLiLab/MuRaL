git clone https://github.com/cailigd/MuRaL.git

conda env create -n test -f environment.yml
#conda env update --file environment.yml --prune
conda env update --file environment.yml --prune

conda activate mural

pip install -e .

#conda deactivate
#conda env remove -n test
