# FORMED_ML
Machine learning models for the FORMED database and downstream tasks, and cross coupling tool.

All the raw data associated with this project can be found in the corresponding [Materials Cloud record](https://doi.org/10.24435/materialscloud:j6-e2), including interactive visualization. Notably, all labels and xyz files containing molecular 3D structure are available there.

# Installation

We provide a conda environment file `environment.yaml` to install all requirements with conda into a conda environnment called `FORMED`. The `FORMED` environment can be used to run all provided scripts and notebooks in this repository. This approach has been tested in several recent releases of Ubuntu (18-22) with python versions 3.7-3.9.

Use the environment file by running `conda env create -f environment.yaml` and activate the environment with `conda activate FORMED`. 

# Content
1. `crosscoupler` contains the source code and example of the cross-coupling tool, which can find suitable unique sp2 carbons in molecules and generate coupling products. The code is given as a jupyer notebook. To run the jupyter notebook you need to provide the conda environment `FORMED` (vide supra) by running `python -m ipykernel install --user --name=FORMED`. After that, you should be able to run the jupyter notebook normally by selecting the `FORMED` environment as kernel. Example inputs are provided and pre-filled.

2. `cv` contains 10-fold cross-validation scripts for the XGBoost ML models, as well as the outputs of the scripts. To run, please execute `generate_slatm.py` adequately by pointing to the xyz files (vide infra) to generate the `repr.npy` file containing the SLATM representations.

3. `data` contains raw data as `numpy` arrays, as extracted from the TD-DFT computations. It also contains the script to generate the SLATM representation from xyz files available in the [Materials Cloud record](https://doi.org/10.24435/materialscloud:j6-e2) and saving it as `repr.npy`. The same data is also available in the record. We also provide the *exact* definition of the SMARTS keys used for substructure search.

4. `models` contains the trained XGBoost models and learning curves. To re-run, please execute `generate_slatm.py` adequately by pointing to the xyz files (vide supdra) to generate the `repr.npy` file containing the SLATM representations.

5. `predict` contains the scripts for inference using the trained models. The SLATM representations of the dimer data can be generated with the given script from the xyz files available in the [Materials Cloud record](https://doi.org/10.24435/materialscloud:j6-e2). The output of the predictions is also given. To run, please execute `generate_slatm.py` adequately by pointing to the xyz files of the molecules to predict to generate the `repr.npy` file containing the SLATM representations.
