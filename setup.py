# -*- coding: utf-8 -*-

import re

from setuptools import setup, find_packages

import subprocess
try:
    subprocess.check_call(['pip', 'install', '.'], cwd='dirichlet_python')
except subprocess.CalledProcessError as e:
    print(f"Error installing dirichlet_python: {e}")
    exit(1)

def get_version():
    try:
        f = open("MuRaL/_version.py")
    except EnvironmentError:
        return None
    for line in f.readlines():
        mo = re.match("__version__ = '([^']+)'", line)
        if mo:
            ver = mo.group(1)
            return ver
    return None

setup(
    name='MuRaL',
    version=get_version(),
	author_email='caililab@outlook.com',
    packages=find_packages(),
    description='Mutation Rate Learner with Neural Networks',
	scripts=['bin/mural_train', 'bin/mural_train_TL', 'bin/mural_predict', 'bin/gen_distal_h5', 'bin/get_best_mural_models', 'bin/calc_mu_scaling_factor', 'bin/scale_mu', 'bin/calc_regional_corr', 'bin/calc_kmer_corr'],
	include_package_data=True,
)
