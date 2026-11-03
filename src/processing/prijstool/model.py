# %%
# import all needed packages
import numpy as np
import pandas as pd
import xarray as xr

from processing.prijstool.costs import calculate_variable_irrigation_costs
from processing.prijstool.cropdata import create_cropdata_xr
from processing.prijstool.demand import (
    calculate_constant_elasticity_price_and_consumer_surplus,
    calculate_linear_price_and_consumer_surplus,
    calibrate_constant_elasticity_demand,
    calibrate_linear_demand,
)
from processing.prijstool.elasticity import get_price_effect_masks, prepare_elasticity
from processing.prijstool.prices import prepare_base_prices
from processing.prijstool.production import (
    aggregate_production_by_crop,
    calculate_average_standard_production,
    calculate_production_and_costs,
    calculate_scenario_yields,
    reconstruct_local_yield,
)
from processing.prijstool.results import create_results_dataframe
from processing.prijstool.welfare import calculate_welfare_changes
from processing.prijstool.yields import get_standard_yield


# import utils
# %%
class Prijstool:
    def __init__(
        self,
        prijstool_elasticity,
        crop_prices,
        cropdata_split_sce,
        cropdata_split_ref,
        params,
        optFixedPrice=False,
        optConstElas=False,
        multpotyield=1.0,
        multredyield=1.0,
        perc_prijspeil_beregening=19.4,  # 3.7#19.4   # De prijzen staan nu in prijspeil 2019. We kunnen het updaten naar prijspeil 2023 door 19.4% erbij op te tellen.
    ):

        self.optFixedPrice = optFixedPrice
        self.optConstElas = optConstElas
        self.prijstool_elasticity = prijstool_elasticity
        self.crop_prices = crop_prices
        self.params = params
        self.multpotyield = multpotyield
        self.multredyield = multredyield
        self.perc_prijspeil_beregening = perc_prijspeil_beregening

        self.cropdata_split_sce = create_cropdata_xr(
            cropdata_split_sce, crop_prices, prijstool_elasticity
        )
        self.cropdata_split_ref = create_cropdata_xr(
            cropdata_split_ref, crop_prices, prijstool_elasticity
        )

    def run(self):
        # step 0 read input files
        cropnum = np.unique(self.cropdata_split_sce["Cropnum"].dropna("Area"))
        cropnam_bool = self.prijstool_elasticity.loc["nummer"].isin(cropnum)
        cropnam = cropnam_bool[cropnam_bool].index.tolist()

        area_name_num = pd.Series(
            self.crop_prices.regio_naam.values,
            index=self.crop_prices.regio_nummer.values,
        )

        # set the calculation arrays
        k6_sce = get_standard_yield(self.cropdata_split_sce, self.params.calc)
        k6_ref = get_standard_yield(self.cropdata_split_ref, self.params.calc)

        # step 1 calculate the production and costs
        (
            areaal_sce,
            standard_opbrengst_kg_ref,
            standard_opbrengst_kg_sce,
            droogtederving_kg_sce,
            arbeidskosten_sce,
            energiekosten_sce,
        ) = calculate_production_and_costs(
            cropdata_sce=self.cropdata_split_sce,
            cropdata_ref=self.cropdata_split_ref,
            standard_yield_sce=k6_sce,
            standard_yield_ref=k6_ref,
            multpotyield=self.multpotyield,
            multredyield=self.multredyield,
        )

        # step 2 determines the annual price based on the actual production and calculate the welfare impact
        # calibrate the price function
        p_elas = prepare_elasticity(self.prijstool_elasticity, cropnam)

        # set masks for the different price effects
        (
            fixed_price_effect,
            local_market_price_effect,
            world_market_price_effect,
        ) = get_price_effect_masks(p_elas)

        (
            standard_opbrengst_kg_ref_per_crop_world,
            standard_opbrengst_kg_sce_per_crop_world,
            droogtederving_kg_sce_per_crop,
        ) = aggregate_production_by_crop(
            standard_production_ref=standard_opbrengst_kg_ref,
            standard_production_sce=standard_opbrengst_kg_sce,
            yield_loss_sce=droogtederving_kg_sce,
            elasticity=p_elas,
        )

        base_price = prepare_base_prices(
            crop_prices=self.crop_prices,
            crop_names=cropnam,
        )

        ################
        # DIT IS YREF

        # MvdV 181130 added constant elasticity demand function

        # Calibrate demand using reference production
        if self.optConstElas:
            demand_scale_factor = calibrate_constant_elasticity_demand(
                standard_production=standard_opbrengst_kg_ref,
                standard_production_world=(standard_opbrengst_kg_ref_per_crop_world),
                base_price=base_price,
                elasticity=p_elas,
                local_market_price_effect=local_market_price_effect,
                world_market_price_effect=world_market_price_effect,
            )

        else:
            demand_slope, demand_intercept = calibrate_linear_demand(
                standard_production=standard_opbrengst_kg_ref,
                standard_production_world=(standard_opbrengst_kg_ref_per_crop_world),
                base_price=base_price,
                elasticity=p_elas,
                local_market_price_effect=local_market_price_effect,
                world_market_price_effect=world_market_price_effect,
            )

        # TOT HIER YREF
        #######################
        # HIER BEGIMT YSCE

        ## new production - global for sel == 2 to calculate new price, local to calculate chps

        ############################
        # standard_opbrengst_kg_sce is hier dus de mediaan van de actuele opbrengst van sce
        # dat is raar maar was eerst ook zo?
        ######################################

        # Calculate scenario production
        (
            new_yield_local,
            new_yield_world_per_crop_year,
            new_yield_local_from_world_per_crop_year,
        ) = calculate_scenario_yields(
            standard_production_sce=standard_opbrengst_kg_sce,
            standard_production_sce_world=(standard_opbrengst_kg_sce_per_crop_world),
            yield_loss_sce=droogtederving_kg_sce,
            yield_loss_sce_per_crop=(droogtederving_kg_sce_per_crop),
            elasticity=p_elas,
        )

        # Calculate average scenario production
        (
            average_standard_yield_local,
            average_standard_yield_world,
        ) = calculate_average_standard_production(
            standard_production_local=standard_opbrengst_kg_sce,
            standard_production_world=(standard_opbrengst_kg_sce_per_crop_world),
        )

        # Recalibrate linear demand using average scenario production
        if self.optFixedPrice:
            demand_slope, demand_intercept = calibrate_linear_demand(
                standard_production_local=average_standard_yield_local,
                standard_production_world=average_standard_yield_world,
                base_price=base_price,
                elasticity=p_elas,
                local_market_price_effect=local_market_price_effect,
                world_market_price_effect=world_market_price_effect,
            )

        # MvdV 190108 corrected error in calculation nyls: it contained total actual demand in NL, but needs to contain actual demand per area
        # fraction per area and crop of total average potential production in NL
        # hier wordt de regionale productie teruggerekend uit de fractie
        # en de globale productie. Dit wordt gebruikt om de
        # nieuwe prijs te berekenen.
        new_yield_local_reconstructed = reconstruct_local_yield(
            average_standard_yield_local=average_standard_yield_local,
            new_yield_local_from_world=(new_yield_local_from_world_per_crop_year),
        )

        # new price and change in consumer surplies
        # MvdV 181130 added constant elasticity demand function
        if self.optConstElas:  # constant elasticity y=a*p^b with b=elasd
            (
                new_price,
                change_consumer_surplus,
            ) = calculate_constant_elasticity_price_and_consumer_surplus(
                new_yield_local=new_yield_local,
                new_yield_world_per_crop_year=(new_yield_world_per_crop_year),
                demand_scale_factor=demand_scale_factor,
                base_price=base_price,
                p_elas=p_elas,
                fixed_price_effect=fixed_price_effect,
                local_market_price_effect=local_market_price_effect,
                world_market_price_effect=world_market_price_effect,
            )

        else:  # linear demand function y=demand_slope*p+demand_intercept

            (
                new_price,
                change_consumer_surplus,
            ) = calculate_linear_price_and_consumer_surplus(
                new_yield_local=new_yield_local,
                new_yield_world=new_yield_world_per_crop_year,
                reconstructed_local_yield=(new_yield_local_reconstructed),
                standard_production_sce=(standard_opbrengst_kg_sce),
                demand_slope=demand_slope,
                demand_intercept=demand_intercept,
                base_price=base_price,
                local_market_price_effect=(local_market_price_effect),
                world_market_price_effect=(world_market_price_effect),
            )  # ysce

        # change in producer surplus, consumer surplus and total welfare impact
        # variable_irrigation_costs = arbeidskosten + energiekosten
        # MvdV 191119 doubling of variable irrigation costs (for explanation, see p:\11211541-003-kentalleneffecten\Landbouw\011a Effectmodule Landbouw\src\workflow\Omschrijvingen\beregening_factor2)
        variable_irrigation_costs_def = calculate_variable_irrigation_costs(
            arbeidskosten_sce=arbeidskosten_sce,
            energiekosten_sce=energiekosten_sce,
            perc_prijspeil_beregening=self.perc_prijspeil_beregening,
        )

        (
            change_producer_surplus,
            change_total_welfare,
        ) = calculate_welfare_changes(
            variable_irrigation_costs_def=variable_irrigation_costs_def,
            new_yield_local=new_yield_local,
            new_price=new_price,
            standard_opbrengst_kg_sce=standard_opbrengst_kg_sce,
            base_price=base_price,
            droogtederving_kg_sce=droogtederving_kg_sce,
            change_consumer_surplus=change_consumer_surplus,
            p_elas=p_elas,
        )

        self.comb1_df = create_results_dataframe(
            standard_opbrengst_kg_sce=standard_opbrengst_kg_sce,
            droogtederving_kg_sce=droogtederving_kg_sce,
            change_producer_surplus=change_producer_surplus,
            change_consumer_surplus=change_consumer_surplus,
            change_total_welfare=change_total_welfare,
            variable_irrigation_costs_def=variable_irrigation_costs_def,
            areaal_sce=areaal_sce,
            base_price=base_price,
            new_price=new_price,
            area_name_num=area_name_num,
        )

    def save_output(self, output_fns):
        self.comb1_df.to_csv(output_fns.outnam1_fn, sep=";", decimal=",")

        # per area and year
        twpay_df = (
            self.comb1_df["verandering totale welvaart"]
            .unstack("Area")
            .groupby("Year")
            .sum()
        )
        twpay_df.to_csv(output_fns.twpaynam_fn, sep=";", decimal=",")

        # per crop and year
        slpcy_df = (
            self.comb1_df["verandering totale welvaart"]
            .unstack("Crop")
            .groupby("Year")
            .sum()
        )
        slpcy_df.to_csv(output_fns.twpcyfil_fn, sep=";", decimal=",")

        # per crop and area, average over years
        twpca_df = (
            self.comb1_df["verandering totale welvaart"]
            .unstack("Crop")
            .groupby("Area")
            .mean()
        )
        twpca_df.to_csv(output_fns.twpcafil_fn, sep=";", decimal=",")


# %%
