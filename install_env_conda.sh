git clone git@github.com:CaiLiLab/MuRaL.git

conda env create -n mural -f environment.yml
# if the installation is interupted due to internet issues or some dependencies are updated, try running the following:
# conda env update -n mural -f environment.yml --prune

conda activate mural

#pip install -e .
#conda env remove -n test
