<p align="center">

  <h1 align="center"><a href="http://go.epfl.ch/tencers">Tencers: Tension-Constrained Elastic Rods</a></h1>

  ![Teaser](./release/teaser.png)

  <p align="center">
    <!-- ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia), December 2024. -->
    <br />
    <a href="https://people.epfl.ch/liliane-joy.dandy"><strong>Liliane-Joy Dandy*</strong></a>
    ·
    <a href="https://people.epfl.ch/michele.vidulis"><strong>Michele Vidulis*</strong></a>
    ·
    <a href="https://samararen.github.io"><strong>Yingying Ren</strong></a>
    ·
    <a href="https://people.epfl.ch/mark.pauly?lang=en"><strong>Mark Pauly</strong></a>
    <br />
  </p>

  <p align="center">
    <a href='https://infoscience.epfl.ch/entities/publication/d78a8e72-c425-4d3a-bb28-9800226dbe4d'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat-square' alt='Paper PDF'>
    </a>
    <a href='http://go.epfl.ch/tencers' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat-square' alt='Project Page'>
    </a>
  </p>

</p>

*joint first authors.

# About

This repository contains the source code and data for the SIGGRAPH Asia 2024 paper [Tencers: Tension-Constrained Elastic Rods](https://dl.acm.org/doi/10.1145/3687967). 

A tencer is a structure made of flexible rods tensioned by a small set of cables. This framework proposes an inverse design algorithm that solves for the length and placement of cables such that the equilibrium state of the rod network best approximates a given set of input curves.

To know more about our research, please visit the project [webpage](http://go.epfl.ch/tencers).


# Getting started

## C++ code dependencies

The C++ code relies on `Boost` and `CHOLMOD/UMFPACK`, which must be installed separately. Please follow the instructions for the *C++ code dependencies* listed [here](https://github.com/EPFL-LGG/ElasticRods_fork_tencers/tree/tencers?tab=readme-ov-file#c-code-dependencies). Some parts of the code, such as inverse design optimization also depend on the commercial optimization package [`knitro`](https://www.artelys.com/solvers/knitro/); these will be omitted from the build if `knitro` is not found. A `scipy`-based version of the code is also provided as a fully open source, alternative implementation to `knitro`.

## Cloning the repository
Clone the repository *recursively* so that its submodules are also downloaded:

```
git clone git@github.com:EPFL-LGG/Tencers.git --recursive
cd Tencers
```

## Python environment
The python environment can be created as follows:

```
conda env create -f environment.yml
```

and activated with 

```
conda activate tencers
```

## Build instructions
With the python environment activated, the project can be built as follows:
```
mkdir build
cd build
cmake ..
make
```

# Running code
Launch Jupyter lab from the root directory (make sure the python environment has been activated before):

```
jupyter lab
```
Then try opening and running a notebook in the `python/notebooks` folder.

## Browse the data from the paper

The paper's data is located in the folder `python/notebooks/paper_data`. It can be displayed using the notebook `python/notebooks/display_tencers.ipynb`.

## Create a custom tencer and simulate its equilibrium state

An example is provided in the notebook `python/notebooks/compute_equilibrium.ipynb`.

## Inverse design optimization

We provide two different solvers for inverse design optimization. One is the Newton-CG trust-region method, implemented in the commercial solver [`knitro`](https://www.artelys.com/solvers/knitro/) (this is the solver that was used in the paper). The other is an L-BFGS-B solver implemented in `scipy`. The name of the solver is provided in the notebooks' names.

The following notebooks are provided:

 * Example of a tencer with a single closed rod and symmetries: 
   ```
   python/notebooks/hypotrochoid_trefoil_knitro.ipynb
   python/notebooks/hypotrochoid_trefoil_scipy.ipynb
   ```
 * Example of a tencer with multiple closed rod and symmetries: 
   ```
   python/notebooks/two_trefoils_knitro.ipynb
   python/notebooks/two_trefoils_scipy.ipynb
   ```

 * Example of a tencer made of an open rod and an external frame, without symmetries: 
   ```
   python/notebooks/helix_knitro.ipynb
   python/notebooks/helix_scipy.ipynb
   ```


# Citation

If you use this codebase in your project, please consider citing our work:
```
@article{tencers, 
	author = {Dandy, Liliane-Joy and Vidulis, Michele and Ren, Yingying and Pauly, Mark}, 
	title = {Tencers: Tension-Constrained Elastic Rods}, 
	year = {2024}, 
	issue_date = {December 2024}, 
	publisher = {Association for Computing Machinery}, 
	volume = {43}, 
	number = {6}, 
	url = {https://doi.org/10.1145/3687967}, 
	doi = {10.1145/3687967},
	journal = {ACM Trans. Graph.}, 
	articleno = {214}, 
}
```