# %%
import pandas as pd

import utils
from Prijstool_Deltascenarios_Class_rewrite_correct import Prijstool

if "snakemake" not in globals():
    snakefile = r"C:\git_repos\effectmodule_landbouw\snakes\wf_landbouw_test_new"
    snakemake = utils.read_snakemake_rule(snakefile, name="run_prijstool")

input_fns = snakemake.input
params = snakemake.params
output_fns = snakemake.output

cropdata_split_ref = pd.read_csv(input_fns.cropdata_split_ref_fn, index_col=0)
cropdata_split_sce = pd.read_csv(input_fns.cropdata_split_sce_fn, index_col=0)

Prijstool_elasticity = pd.read_csv(input_fns.elasticity_fn, index_col=0)
Prijstool_elasticity = Prijstool_elasticity.drop('gewashoofd').astype(float)

Crop_prices = pd.read_csv(input_fns.crop_prices_fn, index_col=0)

p = Prijstool(Prijstool_elasticity, 
              Crop_prices, 
              cropdata_split_sce.copy(),
              cropdata_split_ref.copy(),
              params)

p.run()
p.save_output(output_fns)
# %%
