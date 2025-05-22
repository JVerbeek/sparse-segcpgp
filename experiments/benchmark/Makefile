SHELL := /bin/zsh
ts := $$(date +%Y%m%d%H%M%S)
# Directories to create
SUBDIRS := mean var trend per
DATA := data
BASE := benchmark_$(ts)
# Default target

# Rule to create parent directory and subdirectories
setup_data: 
	@mkdir -p $(DATA)
	@for d in $(SUBDIRS); do \
		mkdir -p $(DATA)/$$d; \
	done
	@touch $(DATA)/.done
	
# Clean up
check_clean_data:
	 @echo -n "Are you sure you want to clean the data directory? [y/N] " && read ans && [ $${ans:-N} = y ]

clean_data: check_clean_data
	rm -rf $(DATA)
	@echo "Removed $(DATA) directory"

data: setup_data
	echo $$(pwd)
	python generate_dense_datasets.py -d $(DATA) -p "data-parameters.yaml"

setup_experiment:
	@mkdir -p results
	@mkdir -p logs
	
experiment: data setup_experiment
	sbatch run_benchmark.sh -d $(DATA) -p experiment-parameters.yaml -r $(BASE)

