# Define phony targets for clarity
.PHONY: all createfig1 createfig2 createfig3 createfig4 createfig5

# Default target
all: createfig1 createfig2 createfig3 createfig4 createfig5 clean

# Create Fig 1 (nn architecture plot)
createfig1:
	mkdir -p output/figures/basic_nn
	cd code/python/basic_nn/plotting && python nn_architecture_plots.py

# Create Fig 2 (SFFN vs TSIR)
createfig2:
	#get birth data
	mkdir -p data/births
	cd code/r/get_data && Rscript get_birth_data.R
	#fit tsir
	mkdir -p output/data/tsir/uk/raw
	cd code/r/tsir && Rscript tsir_uk_run.R
	mkdir -p output/data/tsir/uk/processed
	cd code/r/tsir && Rscript tsir_uk_process.R
	mkdir -p output/data/tsir_susceptibles
	cd code/r/tsir && Rscript tsir_susceptibles_gen.R
	#fit neural nets
	mkdir -p output/models/basic_nn_yearcutoff
	cd code/python/basic_nn && ./full_basic_yearcutoff.sh
	mkdir -p output/data/basic_nn_yearcutoff
	cd code/r/basic_nn && Rscript yearcutoff_basic_nn_process.R
	mkdir -p output/data/basic_nn
	cd code/r/basic_nn && Rscript cases_process.R
	#make plots
	cd code/r/basic_nn && Rscript yearcutoff_compare_plots.R

# Create Fig 3 (SHAP plot)
createfig3:
	mkdir -p output/models/basic_nn_yearcutoff
	cd code/python/basic_nn/explain && ./data_process_explain.sh
	mkdir -p output/data/basic_nn_yearcutoff/explain
	cd code/python/basic_nn/explain && ./basic_nn_explain.sh

# Create Fig 4 (PINN plot) and Table 1 (PINN table)
createfig4:
	mkdir -p output/data/train_test_k
	cd code/python/data_processing && prevac_measles_data_loader.sh
	mkdir -p output/models/pinn_experiments/final_london_pinn_yearcutoff
	cd code/python/pinn_experiments/final_london_pinn_yearcutoff && ./run.sh
	mkdir -p output/tables
	cd code/r/pinn_experiments && Rscript pinn_london_yearcutoff_plots_tables.R



# Create Fig 5 (spatial + temporal plot) 
createfig5:
	Rscript code/r/basic_nn/map_plots.R

# Create Fig 6 (London BIF)
createfig6:
	mkdir -p output/models/basic_nn_bif
	./code/python/basic_nn/full_basic_bif.sh
	mkdir -p output/data/basic_nn_bif
	Rscript code/r/basic_nn/basic_nn_bif_process.R
	Rscript code/r/basic_nn/basic_nn_bif_plots.R

clean:
	rm -r data
	rm -r output/data
	rm -r output/models
