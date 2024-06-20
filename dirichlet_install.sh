# download dirchlet package 
git clone git@github.com:dirichletcal/dirichlet_python.git
cd dirichlet_python
# patch
patch -p1 < ../dirichlet.patch
# install dirchlet package
python setup.py install
cd ..

