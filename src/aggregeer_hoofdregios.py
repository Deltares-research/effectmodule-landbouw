#%%
# Load necessary libraries
import pandas as pd
from pathlib import Path
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt

import utils

#%%
if "snakemake" not in globals():
    snakefile = r"../snakes/wf_landbouwv12_test"
    snakemake = utils.read_snakemake_rule(snakefile, name="risico_hoofdregio")

input_fns = snakemake.input
params = snakemake.params
output_fns = snakemake.output
#%%
# # Load new uitsplitsing welvaart data
regio_df = pd.read_excel(input_fns.droogterisico_fn, index_col = [0,1])
regio_df = regio_df.groupby('Area').sum()
regio_df['Scenario'] = 'REF2017BP18'
#%%
#calculate hoofdregios
subregio_gdf = gpd.read_file(input_fns.subregio_fn)
subregio_gdf['geometry'] = subregio_gdf.representative_point()
hoofdregio_gdf = gpd.read_file(input_fns.hoofdregio_fn)

#calculate which subregion belongs to which hoofdregio
regio_sjoin = gpd.sjoin(subregio_gdf, hoofdregio_gdf, how = 'left', predicate = 'within')
sub_hoofd_regio = pd.Series(regio_sjoin.regionaam.values, index = regio_sjoin.Naam.values)

regio_df['Hoofdregio'] = sub_hoofd_regio[regio_df.index].values
hoofdregio_droogterisico = regio_df.groupby(['Hoofdregio', 'Scenario']).sum()

#droogte_risico_df = hoofdregio_droogterisico[['Droogterisico']]
totaal = hoofdregio_droogterisico.groupby('Scenario').sum()
totaal['Hoofdregio'] = 'Totaal'
totaal = totaal.reset_index().set_index(['Hoofdregio', 'Scenario'])
droogte_risico_df = pd.concat([hoofdregio_droogterisico, totaal])
droogte_risico_df /= 1E6
droogte_risico_df.columns = ['Droogterisico (mln euro / jaar)', 'Droogterisico_T20 (mln euro)']
droogte_risico_df.to_excel(output_fns.droogterisico_hoofdregios_fn)

# %%
# sns.barplot(data=droogte_risico_df.reset_index(), 
#             x='Droogterisico', 
#             y='Hoofdregio',
#             hue='Scenario',
#             palette=['#D3D3D3', '#A8DBA8', '#3B753B'],  # Example colors
#             orient='h'  # Horizontal orientation
# )
# plt.xlabel('Droogterisico (miljoen euro / jaar)')
# plt.grid(axis='x')
# # %%
