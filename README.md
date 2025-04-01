# Person-data-analysis
This repository contains a Jupyter notebook and related helper scripts to analyze authority data of the MetaBelgica partner institutions. It is related to deliverable Deliverable 1.2.2 - Quality requirements and categories of Belgian entities. MetaBelgica. https://doi.org/10.5281/zenodo.14974410  

## Run the notebook from the command line

```bash

# Create a new Python virtual environment
python3 -m venv py-env-person-analysis

# Activate the virtual environment
source py-env-person-analysis/bin/activate

# install ipykernel
pip install ipykernel

# add kernel based on virtual environment
python -m ipykernel install --user --name=py-env-person-analysis

# install dependencies
pip install -r requirements.txt

# run jupyter notebook
jupyter notebook

# select the correct virtual environment in the jupyter UI

```

## License

- **Code**: Licensed under the [MIT License](LICENSE).
- **Figures & Text**: Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0).

When citing this notebook, please use the provided DOI: todo, add after DOI is retrieved after first release
