.PHONY: create-env activate-env deactivate-env update-env

all:
	@echo "make-env: Create a new conda environment"
	@echo "activate-env: Activate the conda environment"
	@echo "deactivate-env: Deactivate the conda environment"
	@echo "update-env: Update the conda environment"

create-env:
	conda env create -f pvi-user.yml

activate-env:
	conda activate pvi_user

deactivate-env:
	conda deactivate

update-env:
	conda env update -f pvi-user.yml
