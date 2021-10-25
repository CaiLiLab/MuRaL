git clone git@github.com:CaiLiLab/MuRaL.git

conda env create -n mural -f environment.yml
# if the installation interupted due to network, run the following:
# conda env update -n mural -f environment.yml --prune

conda activate mural

#pip install -e .
#conda env remove -n test
