# FORMED_ML
Machine learning models for the FORMED database and downstream tasks, and cross coupling tool.

All the raw data associated with this project can be found in the corresponding [Materials Cloud record](https://doi.org/10.24435/materialscloud:j6-e2), including interactive visualization.

# Content

1. `crosscoupler` contains the source code and example of the cross-coupling tool, which can find suitable unique sp2 carbons in molecules and generate coupling products.
2. `cv` contains 10-fold cross-validation scripts for the XGBoost ML models, as well as the outputs of the scripts.
3. `data` contains raw data as `numpy` arrays. It also contains the script to generate the SLATM representation from xyz files available in the [Materials Cloud record](https://doi.org/10.24435/materialscloud:j6-e2). The same data is also available in the record.
4. `moodels` contains the trained XGBoost models and learning curves.
5. `predict` contains the scripts for inference using the trained models. The SLATM representations of the dimer data can be generated with the given script from the xyz files available in the [Materials Cloud record](https://doi.org/10.24435/materialscloud:j6-e2). The output of the predictions is also given.
