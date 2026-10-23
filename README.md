# Effectmodule Landbouw #

## Run workflow ##
Run from the following command from the root directory: <br />
snakemake -s ./workflows/wf_landbouw_test

## Main outputs ##
- droogerisico_new.xlsx
- droogterisico_hoofdregios_new.xlsx

## before any commit ##
Run these commands:
- pixi run pytest
- snakemake -s workflows/wf_landbouw_test.smk all -n -p

A single snakemake rule can be run with the following command from the WCF node:
- pixi run snakemake -s workflows/wf_landbouw_test.smk prepare_median_input -c 1