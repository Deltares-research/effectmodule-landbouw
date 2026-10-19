#%%

import numpy as np
import pandas as pd

import utils
from prijstool_deltascenarios_Class import Prijstool


# %%
def get_hypothetical_median_year_data(cropdata: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the year with the median 'act_opbrengst' for each crop and area combination.

    Parameters
    ----------
    cropdata : pd.DataFrame
        DataFrame containing crop data with columns 'Crop', 'Area', 'Year',
        'act_opbrengst', and other relevant columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with the median year data for each crop and area combination.
    """
    # Drop the 'Scenario' column as it is not needed for median calculation
    cropdata = cropdata.drop('Scenario', axis=1)

    medyear_cropdata = []
    # Group by 'Crop' and 'Area' to calculate median for each group
    for (crop, area), grp in cropdata.groupby(['Crop', 'Area']):
        # Calculate the median value of 'act_opbrengst' for the group
        med_value = grp['act_opbrengst'].median()
        
        # Find the year with 'act_opbrengst' value closest to the median
        med_idx = abs(grp['act_opbrengst'] - med_value).idxmin()
        med_year = grp.loc[med_idx]

        medyear_cropdata.append(med_year)
    
    # Concatenate the list of median year data into a DataFrame
    medyear_cropdata_df = pd.concat(medyear_cropdata, axis=1).T
    # Drop the 'Year' column and convert data to float
    medyear_cropdata_df = medyear_cropdata_df.drop('Year', axis=1).astype(float)
    # Add 'derv_opbrengst' column with value 0 and 'Year' column with value 'med'
    medyear_cropdata_df['derv_opbrengst'] = 0
    medyear_cropdata_df['Year'] = 'med'
    
    return medyear_cropdata_df

def bereken_droogterisico(prijstool_output_sce: pd.DataFrame, median_data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Bereken droogterisico op basis van prijstool output en mediane data.

    Parameters
    ----------
    prijstool_output_sce : pd.DataFrame
        Output van de prijstool voor een scenario.
    median_data : pd.DataFrame
        Mediane data van veranderingen in totale welvaart.

    Returns
    -------
    mean_W_sce : pd.Series
        Gemiddelde verandering in totale welvaart per gewas en gebied.
    Droogte_Risico : pd.Series
        Berekend droogterisico per gewas en gebied.
    """
    #dit is een rare regel want er is geen andere dimensie?
    W_A_med = median_data['verandering totale welvaart'].groupby(['Crop', 'Area']).mean()
    
    W_sce = prijstool_output_sce['verandering totale welvaart']
    mean_W_sce = W_sce.groupby(['Crop', 'Area']).mean()
    Droogte_Risico = -(W_A_med - mean_W_sce)
    return mean_W_sce, Droogte_Risico

def calculate_T20(damage_series: pd.Series) -> float:
    """
    Bereken droogteschade van een T20 jaar.

    Parameters
    ----------
    damage_series : pd.Series
        Input series met de droogteschade voor de complete tijdreeks

    Returns
    -------
    float
        Droogteschade van een T20 jaar.
    """
    # Sorteer de schadewaarden van klein naar groot
    sorted_damage = damage_series.sort_values(ascending=False)
    
    # Rank de jaren (hoogste schade krijgt de laagste rank)
    ranks = sorted_damage.rank(method='first', ascending=True)
    
    # Bepaal de herhalingstijd
    n = len(damage_series)
    return_periods = (n + 1) / ranks

    #merge return periods and damage
    merged = pd.concat([damage_series, return_periods], 
                       axis=1, 
                       keys=['schade', 'herhalingstijd']).sort_values('herhalingstijd', ascending=True)
    
    #interpolate where return period is 20
    T20 = np.interp(20, merged['herhalingstijd'], merged['schade'])
    return T20

def bereken_droogterisico_T20(prijstool_output_sce: pd.DataFrame, median_data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Bereken droogterisico op basis van prijstool output en mediane data.

    Parameters
    ----------
    prijstool_output_sce : pd.DataFrame
        Output van de prijstool voor een scenario.
    median_data : pd.DataFrame
        Mediane data van veranderingen in totale welvaart.

    Returns
    -------
    T20_W_sce : pd.Series
        Gemiddelde verandering in totale welvaart per gewas en gebied voor de T20-situatie.
    Droogte_Risico : pd.Series
        Berekend droogterisico per gewas en gebied.
    """
  
    #dit is een rare regel want er is geen andere dimensie?
    W_A_med = median_data['verandering totale welvaart'].groupby(['Crop', 'Area']).mean()
    
    W_sce = prijstool_output_sce['verandering totale welvaart']
    T20_W_sce = W_sce.groupby(['Crop', 'Area']).apply(calculate_T20)
    Droogte_Risico = -(W_A_med - T20_W_sce)
    return T20_W_sce, Droogte_Risico

#%%
if "snakemake" not in globals():
    snakefile = r"C:\git_repos\effectmodule_landbouw\snakes\wf_landbouw_test_new"
    snakemake = utils.read_snakemake_rule(snakefile, name="bereken_droogterisico")

input_fns = snakemake.input
params = snakemake.params
output_fns = snakemake.output

Prijstool_elasticity = pd.read_csv(input_fns.elasticity_fn, index_col=0)
Prijstool_elasticity = Prijstool_elasticity.drop('gewashoofd').astype(float)

Crop_prices = pd.read_csv(input_fns.crop_prices_fn, index_col=0)

prijstool_output_sce = pd.read_csv(input_fns.sce_output_fn, index_col = [0,1,2], sep=';', decimal=',')

cropdata_split_sce = pd.read_csv(input_fns.cropdata_split_sce_fn, index_col = 0)
cropdata_split_ref = pd.read_csv(input_fns.cropdata_split_ref_fn, index_col = 0)

#%%
#calculate droogterisico voor scenario
#select the year with the median act_opbrengst for each area and crop
medyear_cropdata_sce = get_hypothetical_median_year_data(cropdata_split_sce)

p_A_med = Prijstool(Prijstool_elasticity, 
            Crop_prices, 
            medyear_cropdata_sce.copy(),
            cropdata_split_ref.copy(),
            params)
p_A_med.run()

mean_W_sce, droogte_risico_sce = bereken_droogterisico(prijstool_output_sce, p_A_med.comb1_df)
mean_W_sce, droogte_risico_sce_T20 = bereken_droogterisico_T20(prijstool_output_sce, p_A_med.comb1_df)

# %%
#print(WE_totaal, WE_niet_droogte, toename_droogterisico, WE_landgebruik, WE_klimaat_beregening, WE_klimaat_T_CO2)
# Voeg resultaten toe aan een DataFrame
resultaten_df = pd.DataFrame({
    'Droogterisico': droogte_risico_sce,
    'Droogterisico_T20': droogte_risico_sce_T20
})

# Opslaan naar een CSV-bestand
resultaten_df.to_excel(output_fns.droogterisico_fn)
