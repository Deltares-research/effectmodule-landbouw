# %% -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 09:48:43 2018
Prijstool
converter: https://bit.ly/2r2JILl
Format the pandas file as GAMS scv inputs
- Possibility to split
- Possibility to read idf file, note: library raster_func from Wiebe Borren required!
@author: hermawan

Version histrory
v8   259219 Teun van Woerkom translated script to xarray dataformat and numpy for name-dependent calculation instead of index-dependent
v7   250128 Marnix van der Vat modified option 'optdervAct_Actmed' to calculate economic impact relative to the minimum of potental and median production
1.04 191030 Marnix van der Vat reads areaal.asc and multiplies input values with areaal/10000 to convert input from unit/ha to unit instead of multiplying with 6.25
1.05 191119 Marnix van der Vat doubling of variable irrigation costs
1.06 1911281128 Marnix van der Vat correction of multiplication of energy labour costs with the area in ha per grid cel
1.061 191204 Marnix van der Vat remoeplaced Pandas from_csv with read_csv to comply with newer Pandas versions
"""

import pandas as pd
import numpy as np
import xarray as xr
import imod
from pathlib import Path
import sys
import geopandas as gpd

sys.path.append(r"C:\git_repos\effectmodule_landbouw\src")
import utils

# %%################################
########## FUNCTIES #############
#################################
# using idf as input files
# MvdV 191030 added reading of areaal.asc and multiplication with areaal / 10000
# MvdV 191128 added switch perha to muliply with ha per cell or not
# JP 200619 replacing raster-func with imod-python to open grids if AGRICOM output is in idfs.
# the asc file is loaded as 2 dimensional array

def NCPrijs(data: xr.DataArray,
            landgebruik: xr.DataArray,
            regio: xr.DataArray,
            areagrid: xr.DataArray,
            perha: bool):
    """
    Returns a summarized DataFrame of the input data aggregated by land use and region.

    Parameters
    ----------
    data : xr.DataArray
        The input data array to be aggregated.
    landgebruik : xr.DataArray
        The land use array corresponding to the input data.
    regio : xr.DataArray
        The region array corresponding to the input data.
    areagrid : xr.DataArray
        The area in m2 per grid cell.
    perha : bool
        Boolean indication wether "data" is expressed per hectare.
        If True, the data will be multiplied by the area in hectares per grid cell.

    Returns
    -------
    sum_df : pd.DataFrame
        The summarized DataFrame with data aggregated by land use and region.
    """


    mask_area = regio.notnull()
    data = data * mask_area
    regio = regio * mask_area

    # Convert areagrid from m2 to ha
    areagrid = areagrid * 0.0001
    if perha:
        data = data * areagrid

    regio.name = "regio"
    data.name = "data"

    database = xr.merge([data, landgebruik, regio, areagrid], compat="override")
    database = database.where(database.landgebruik != -9999)
    # database['year'] = database.data.time.dt.year

    database = database.compute()
    crop_regio_sum_da = database.groupby(["landgebruik", "regio"]).sum()

    #calculate yearlyd data from smaller timesteps
    if "time" in crop_regio_sum_da.dims:
        crop_regio_sum_da["yearly_data"] = (
            crop_regio_sum_da["data"].groupby(crop_regio_sum_da.time.dt.year).sum()
        )
        sum_df = (
            crop_regio_sum_da[["yearly_data", "areaal"]]
            .to_dataframe()
            .drop(["dx", "dy"], axis=1)
        )
        sum_df["data"] = sum_df["yearly_data"] / sum_df["areaal"]
        sum_df = sum_df.drop("yearly_data", axis=1)
    else:
        sum_df = (
            crop_regio_sum_da[["data", "areaal"]]
            .to_dataframe()
            .drop(["dx", "dy"], axis=1)
        )
        sum_df["data"] = sum_df["data"] / sum_df["areaal"]
    if 'spatial_ref' in sum_df.columns:
        sum_df = sum_df.drop('spatial_ref', axis=1)
    return sum_df


# SPLIT===============================================================================================


def databasesplit(split_area, data, attribute, header):
    #    split_area=split_df
    #    data=cropdata
    # The relevant crop that will be split are filtered from the data in xml file
    n = split_area.Crop.unique()
    split_crop = data.loc[data.Crop.isin(n)]

    split_crop["lookup"] = (
        split_crop[attribute[2]].map(str) + "_" + split_crop[attribute[3]].map(str)
    )
    split_area["lookup"] = (
        split_area[attribute[2]].map(str) + "_" + split_area[attribute[3]].map(str)
    )

    # number of fractions for each crop
    fraction = (
        split_area.drop(["Crop", "Area", "lookup"], axis=1).columns
    ).values.tolist()

    # New columns (number depends on the number of fractions) are added to the database.
    # The columns are filled based on the specified fraction defined for each crop and area
    split_crop = pd.merge(
        split_crop,
        split_area.iloc[:, 2:],
        how="left",
        left_on="lookup",
        right_on="lookup",
    )
    split_crop = split_crop.dropna(how="any")

    # Fraction of areas are multiplied by area
    split_crop[fraction] = split_crop[fraction].multiply(
        split_crop["areaal"], axis="rows"
    )
    split_crop = split_crop.drop(columns=["lookup"])

    # The new areas (split) column are melt as row in the dataframe
    split_crop_melt = pd.melt(
        split_crop, id_vars=header, var_name="CropSplit", value_name="AreaSplit"
    )

    # The new crop is renamed as new number = old number*100 + fraction.
    # For example potato (crop number 3) that have 4 fractions will have such new names: 301, 302, 303, 304
    split_crop_melt["CropSplit"] = split_crop_melt["CropSplit"].apply(pd.to_numeric)
    split_crop_melt["Crop"] = (
        split_crop_melt["Crop"] * 100 + split_crop_melt["CropSplit"]
    )

    # The column with old area values are replaced with new area (with fraction) values.
    split_crop_melt = split_crop_melt.drop(columns=["CropSplit", "areaal"])
    split_crop_melt = split_crop_melt.rename(columns={"AreaSplit": "areaal"})

    # The rows in data from xml file which contains the value before splitting are removed
    data["lookup"] = data["Crop"].map(str) + "_" + data["Area"].map(str)
    #    data=data[~data['lookup'].isin(split)]
    data = data.drop(columns=["lookup"])

    # The new rows which contains the value after splitting are added
    data = pd.concat(
        [data, split_crop_melt]
    )  # data.append(split_crop_melt, ignore_index=False)
    data = data.sort_values(by=attribute, axis=0, ascending=True)
    data = data[header]

    return data


# %%
# paths
if "snakemake" not in globals():
    snakefile = r"C:\git_repos\effectmodule_landbouw\snakes\wf_landbouw_test_new"
    snakemake = utils.read_snakemake_rule_2(
        snakefile, name="run_prijstool_preprocessing"
    )

input_fns = snakemake.input
params = snakemake.params
output_fns = snakemake.output
print(params)

Sub_crops = pd.read_csv(input_fns.sub_crops_fn, index_col=0)
# %%
for fn in [output_fns.cropdata_fn]:  # , output_fns.outnam1_fn]:
    Path(fn).parent.mkdir(exist_ok=True, parents=True)

# open all files
Yldactdry_med_combi = imod.rasterio.open(input_fns.Yldactdry_med_combi_fn)
Ylderv_med_combi = imod.rasterio.open(input_fns.Dervactdry_med_combi_fn)
potASC_combi = xr.open_dataarray(input_fns.potASC_combi_fn)
act_dry_ASC_combi = xr.open_dataarray(input_fns.act_dry_ASC_combi_fn)


landgebruik_ref = imod.rasterio.open(input_fns.landgebruik_ref_fn)
areaalasc_ref = imod.rasterio.open(input_fns.areaalasc_ref_fn)

landgebruik_sce = xr.open_dataarray(input_fns.landgebruik_fn).rename(
    "landgebruik"
)  # imod.rasterio.open
areaalasc_sce = imod.rasterio.open(input_fns.areaalasc_fn).rename("areaal")

arbdASC_sce = xr.open_dataarray(input_fns.arbdASC_fn)
enrgASC_sce = xr.open_dataarray(input_fns.enrgASC_fn)

regios = gpd.read_file(input_fns.risico_subregio_fn)
regios['Nummer'] = regios['Nummer'].astype(float)
regios_ar = imod.prepare.rasterize(geodataframe=regios,
                                   like = areaalasc_sce, 
                                   column = "Nummer", 
                                   fill=np.nan)

years = np.unique(potASC_combi.time.dt.year)
continu_years = np.arange(years.min(), years.max() + 1)
missing_years = continu_years[~np.isin(continu_years, years)]

assert (
    missing_years.size == 0
), f"{', '.join(missing_years.astype(str))} are not present, between {years.min()} and {years.max()}"


# %%################################
### MAKE PRIJSTOOL INPUT #########
#################################

# Load Yield
# MvdV 191128 added switch perha to muliply with ha per cell or not Tru for yields, Slase for costs
#data, landgebruik, regio, areagrid, perha = potASC_combi[potASC_combi.time.dt.year.isin(years)],landgebruik_sce,regios_ar,areaalasc_sce, True

pot_opbrengst = NCPrijs(
    potASC_combi[potASC_combi.time.dt.year.isin(years)],
    landgebruik_sce,
    regios_ar,
    areaalasc_sce,
    perha=True,
).rename(columns={"data": "pot_opbrengst", "Areagrid": "areaal"})

act_opbrengst = (
    NCPrijs(
        act_dry_ASC_combi[act_dry_ASC_combi.time.dt.year.isin(years)],
        landgebruik_sce,
        regios_ar,
        areaalasc_sce,
        perha=True,
    )
    .drop(columns=["areaal"])
    .rename(columns={"data": "act_opbrengst"})
)

ref_med_derving = (
    NCPrijs(
        Ylderv_med_combi, landgebruik_ref, regios_ar, areaalasc_ref, perha=True
    )
    .drop(columns=["areaal"])
    .rename(columns={"data": "ref_med_derv"})
)

# Load Labor costs
# MvdV 191128 added switch perha to muliply with ha per cell or not Tru for yields, Slase for costs
arbeidskosten = (
    NCPrijs(
        arbdASC_sce[arbdASC_sce.time.dt.year.isin(years)],
        landgebruik_sce,
        regios_ar,
        areaalasc_sce,
        perha=False,
    )
    .drop(columns=["areaal"])
    .rename(columns={"data": "arbeidskosten"})
)

# Load Energy costs
energiekosten = (
    NCPrijs(
        enrgASC_sce[enrgASC_sce.time.dt.year.isin(years)],
        landgebruik_sce,
        regios_ar,
        areaalasc_sce,
        perha=False,
    )
    .drop(columns=["areaal"])
    .rename(columns={"data": "energiekosten"})
)

# Merge Potential and Actual Yield and make Crop Database
cropdata = act_opbrengst.merge(pot_opbrengst, left_index=True, right_index=True)
cropdata = cropdata.merge(ref_med_derving, left_index=True, right_index=True)
cropdata = cropdata.merge(arbeidskosten, left_index=True, right_index=True)
cropdata = cropdata.merge(energiekosten, left_index=True, right_index=True)

#%%
#calculate derving given certain calculation method specified in params.calc
if params.calc == "optdervActPot":
    cropdata["derv_opbrengst"] = cropdata["pot_opbrengst"] - cropdata["act_opbrengst"]

elif params.calc == "optdervAct_Actgem":
    cropdata["derv_opbrengst"] = cropdata["gem_opbrengst"] - cropdata["act_opbrengst"]
    cropdata["derv_opbrengst"] = np.where(
        cropdata["derv_opbrengst"] < 0, 0, cropdata["derv_opbrengst"]
    )

# MvdV 250128 optie aangepast om derving te bepalen ten opzicht van het minimum van potentiele en mediane opbrengst
elif params.calc == "optdervAct_Actmed":
    cropdata["derv_opbrengst"] = cropdata["med_opbrengst"] - cropdata["act_opbrengst"]
    cropdata["derv_opbrengst"] = np.where(
        cropdata["derv_opbrengst"] < 0, 0, cropdata["derv_opbrengst"]
    )

elif params.calc == "optdervAct_ActMedPot":
    cropdata["derv_opbrengst"] = np.where(
        cropdata["med_opbrengst"] < cropdata["pot_opbrengst"],
        cropdata["med_opbrengst"] - cropdata["act_opbrengst"],
        cropdata["pot_opbrengst"] - cropdata["act_opbrengst"],
    )
    cropdata["derv_opbrengst"] = np.where(
        cropdata["derv_opbrengst"] < 0, 0, cropdata["derv_opbrengst"]
    )
elif params.calc == "optdervderv_ActMed_new":
    pot_act = cropdata["pot_opbrengst"] - cropdata["act_opbrengst"]
    cropdata["derv_opbrengst"] = pot_act - cropdata["ref_med_derv"]
    cropdata["derv_opbrengst"] = np.where(
        cropdata["derv_opbrengst"] < 0, 0, cropdata["derv_opbrengst"]
    )

    # #test same as old
    # med_act_opbrengst = cropdata.groupby(['regio', 'landgebruik'])["act_opbrengst"].median()
    # cropdata['act_opbrengst'] = [med_act_opbrengst.loc[regio, landgebruik] - derv for regio, landgebruik, derv in zip(cropdata.index.get_level_values('regio'), cropdata.index.get_level_values('landgebruik'), cropdata['derv_opbrengst'])]
    # cropdata['schadeberekening_opbrengst'] = cropdata['act_opbrengst'] 

    #real
    #med_act_opbrengst = cropdata.groupby(['regio', 'landgebruik'])["act_opbrengst"].median()
    #cropdata['damage_calculation_yield'] = [med_act_opbrengst[r, l] - c for r, l, c in zip(cropdata.index.get_level_values('regio'), cropdata.index.get_level_values('landgebruik'), cropdata['derv_opbrengst'])]
else:
    print("no calculation method given")


#finalize cropdata data
cropdata["Scenario"] = params.scenario
cropdata.reset_index(inplace=True)

cropdata = cropdata[
    [
        "regio",
        "landgebruik",
        "act_opbrengst",
        "pot_opbrengst",
        "areaal",
        "damage_calculation_yield",
        "derv_opbrengst",
        "ref_med_derv",
        "arbeidskosten",
        "energiekosten",
        "year",
        "Scenario",
    ]
]
cropdata = cropdata.rename(columns={"year": "Year"})
cropdata.to_csv(output_fns.cropdata_fn)

# cropdata = pd.read_csv(r"p:\11205271-kpp-dp-zoetwater\Effectmodules_NWM\Landbouw\Prijstool_git\prijstool\data\3-input\LHM4.2.0_FAO_NWM22\LHM4.2.0_FAO_NWM22_agricom.csv")
# ==================================================================================================
# Het gewas 21 (fruitkwekerijen) omzetten in gewas 9 (fruit).
cropdata = cropdata.rename({"landgebruik": "Crop", "regio": "Area"}, axis=1)
cropdata.loc[cropdata["Crop"] == 21, "Crop"] = 9
header = list(cropdata.columns.values)

# ==================================================================================================
# SPLITTING based on the Sub crops
split_df = Sub_crops.copy()
split_df.reset_index(level=0, inplace=True)
split_df = (split_df.rename(columns={"Split_Crop": "Crop"})).astype(float)
split_df1 = split_df.copy()
if split_df.empty:
    print("No splitting!")
else:
    attribute = ["Scenario", "Year", "Crop", "Area"]
    cropdata = databasesplit(split_df, cropdata, attribute, header)

for crop in split_df.Crop.unique():
    cropdata.loc[cropdata.Crop == crop, ["areaal"]] = 0

cropdata = cropdata[cropdata.areaal != 0]
cropdata.to_csv(output_fns.cropdata_split_fn)
# %%
