#%% 
#import all needed packages
import pandas as pd
import numpy as np
import xarray as xr

#import utils
#%%
class Prijstool:
    def __init__(self, 
                 Prijstool_elasticity, 
                 Crop_prices, 
                 cropdata_split_sce,
                 cropdata_split_ref,
                 params,
                 optFixedPrice = False,
                 optConstElas = False,
                 multpotyield = 1.0,
                 multredyield = 1.0,
                 Perc_prijspeil_beregening = 19.4, #3.7#19.4   # De prijzen staan nu in prijspeil 2019. We kunnen het updaten naar prijspeil 2023 door 19.4% erbij op te tellen. 
                 Perc_prijspeil_voeding = 36.3#0 #36.3     # De prijzen staan nu in prijspeil 2016. We kunnen het updaten naar prijspeil 2023 door 36.3% erbij op te tellen.
                ):       
                        
        self.optFixedPrice = optFixedPrice
        self.optConstElas = optConstElas
        self.Prijstool_elasticity = Prijstool_elasticity
        self.params = params
        self.multpotyield = multpotyield
        self.multredyield = multredyield
        self.Perc_prijspeil_beregening = Perc_prijspeil_beregening

        self.cropdata_split_sce = self.create_cropdata_xr(cropdata_split_sce)
        self.cropdata_split_ref = self.create_cropdata_xr(cropdata_split_ref)

        self.update_crop_prices(Perc_prijspeil_voeding, Crop_prices)

    def create_cropdata_xr(self, cropdata):
        crop_name_num = pd.Series(self.Prijstool_elasticity.columns, index = self.Prijstool_elasticity.loc['nummer'])

        #preprocess cropdata to xarray from csv
        cropdata['Cropnum'] = cropdata['Crop']
        cropdata['Crop'] = cropdata['Crop'].map(crop_name_num)
        cropdata = cropdata.set_index(['Crop', 'Area', 'Year'])

        # Add regio_13 rows with zeros
        regio_13 = pd.MultiIndex.from_product([cropdata.index.get_level_values('Crop').unique(), 
                                            [13.0], 
                                            cropdata.index.get_level_values('Year').unique()],
                                            names=cropdata.index.names)
        regio_13_df = pd.DataFrame(0.0, index=regio_13, columns=cropdata.columns)

        cropdata = pd.concat([cropdata, regio_13_df]).sort_index().to_xarray().fillna(0)
        return cropdata

    def update_crop_prices(self, Perc_prijspeil_voeding, Crop_prices):
        # Update crop_prices
        # Columns to multiply by Perc_prijspeil_voeding
        columns_to_multiply = ['Grasland', 'Mais', 'Aardappelen', 'Bieten', 'Granen', 'Overig', 
                            'Boomteelt', 'Fruit', 'Bollen', 'Consumptieaardappelen', 
                            'Fabrieksaardappel', 'Pootaardappel', 'Overig.1', 'Uien', 
                            'Gladiool', 'Hyacint', 'Narcis', 'Tulp', 'Rest']
        
        self.updated_crop_prices = Crop_prices.copy()
        # Multiply each specified column by 1.5
        self.updated_crop_prices[columns_to_multiply] = self.updated_crop_prices[columns_to_multiply] * (1+Perc_prijspeil_voeding/100)
    
    def run(self):
        #step 0 read input files 
        cropnum=list()
        cropnum = np.unique(self.cropdata_split_sce['Cropnum'].dropna('Area'))
        cropnam_bool = self.Prijstool_elasticity.loc['nummer'].isin(cropnum)
        cropnam = cropnam_bool[cropnam_bool].index.tolist()

        area_name_num = pd.Series(self.updated_crop_prices.regio_naam.values, index = self.updated_crop_prices.regio_nummer.values)

        # Load data for 'single' crops : Grasland, mais, bieten, granen, boomteelt, fruit
        Prijstool_elasticity_single_crops = self.Prijstool_elasticity[cropnam]

        #set the calculation arrays
        if self.params.calc == 'optdervActPot':
            k6_sce = self.cropdata_split_sce['pot_opbrengst']
            k6_ref = self.cropdata_split_ref['pot_opbrengst']
        elif self.params.calc == 'optdervAct_Actgem':
            k6_sce = self.cropdata_split_sce['gem_opbrengst']
            k6_ref = self.cropdata_split_ref['gem_opbrengst']
        elif self.params.calc in ['optdervAct_Actmed', 'optdervderv_ActMed', 'optdervderv_ActMed_new']:
            k6_sce = self.cropdata_split_sce['act_opbrengst'].median('Year')
            k6_ref = self.cropdata_split_ref['act_opbrengst'].median('Year')

        #step 1 calculate the production and costs
        areaal_sce = self.cropdata_split_sce['areaal'].mean('Year')
        areaal_ref = self.cropdata_split_ref['areaal'].mean('Year')

        ############
        #AREAAL BIJ Y IS NU AREAAL VAN SCENARIO EN NIET VAN DE REF
        #EN MOET DE K6 DUS VAN DE REF BLIJVEN OF PER SCENARIO?
        #DUS DIT ,MOET IN DE BASIS OOK AL EEN YREF EN YSCE ZIJN
        #INCLUSIEF AREAAL VOOR YREF EN YSCE

        standard_opbrengst_kg_ref = areaal_ref*k6_ref*self.multpotyield
        standard_opbrengst_kg_sce = areaal_sce*k6_sce*self.multpotyield
        act_yield_kg_sce = areaal_sce*self.cropdata_split_sce['act_opbrengst']*self.multpotyield
        droogtederving_kg_sce = areaal_sce*self.cropdata_split_sce['derv_opbrengst']*self.multredyield
        damage_calculation_yield = areaal_sce*self.cropdata_split_sce['damage_calculation_yield']*self.multpotyield
        #######################


        arbeidskosten_sce = areaal_sce*self.cropdata_split_sce['arbeidskosten']
        energiekosten_sce = areaal_sce*self.cropdata_split_sce['energiekosten']
            
        #step 2 determines the annual price based on the actual production and calculate the welfare impact
        #calibrate the price function
        Prijstool_elasticity_single_crops.columns.name = 'Crop'
        p_elas = Prijstool_elasticity_single_crops.T.sort_index().to_xarray()
        p_elas[['NL aandeel wereldaanbod', 'NL aandeel wereldvraag']] = xr.where(p_elas[['NL aandeel wereldaanbod', 'NL aandeel wereldvraag']]<=0, 1, p_elas[['NL aandeel wereldaanbod', 'NL aandeel wereldvraag']])

        #set masks for the different price effects
        fixed_price_effect = p_elas['prijseffect']==0
        local_market_price_effect = p_elas['prijseffect']==1
        world_market_price_effect = p_elas['prijseffect']==2

        #do sums over area
        #DIT MOETEN EEN YREF EN EEN YSCE VARIANT WORDEN
        #############################

        standard_opbrengst_kg_ref_per_crop = standard_opbrengst_kg_ref.sum('Area')
        standard_opbrengst_kg_sce_per_crop = standard_opbrengst_kg_sce.sum('Area')
        act_yield_kg_sce_per_crop = act_yield_kg_sce.sum('Area')
        damage_calculation_yield_per_crop = damage_calculation_yield.sum('Area')

        standard_opbrengst_kg_ref_per_crop_world = standard_opbrengst_kg_ref_per_crop / p_elas['NL aandeel wereldaanbod']
        standard_opbrengst_kg_sce_per_crop_world = standard_opbrengst_kg_sce_per_crop / p_elas['NL aandeel wereldaanbod']
        act_yield_kg_sce_per_crop_world = act_yield_kg_sce_per_crop / p_elas['NL aandeel wereldaanbod']
        damage_calculation_yield_per_crop_world = damage_calculation_yield_per_crop / p_elas['NL aandeel wereldaanbod']

        base_price = self.updated_crop_prices.set_index('regio_nummer')
        base_price = base_price[cropnam]
        base_price = base_price.stack(0)
        base_price.index.names = ['Area', 'Crop']
        base_price = base_price.swaplevel(0, 1).sort_index().to_xarray()

        ################
        #DIT IS YREF

        #MvdV 181130 added constant elasticity demand function
        if self.optConstElas: #constant elasticity y=a*p^b with b=elasd
        #MvdV 181203 make a copy of the elasticities to modify them later
            demand_scale_factor = xr.where(local_market_price_effect, standard_opbrengst_kg_ref / (base_price ** p_elas['elasticity']), 0) #yref
            demand_scale_factor = xr.where(world_market_price_effect, standard_opbrengst_kg_ref_per_crop_world / (base_price ** p_elas['elasticity']), demand_scale_factor) #yref
            demand_scale_factor = xr.where(demand_scale_factor==0, 1, demand_scale_factor)

        else: #linear demand function y=demand_slope*p+demand_intercept
            #demand_slope slope of the demand curve
            #demand_intercept intercept of the demand curve
            demand_slope = xr.where(local_market_price_effect, p_elas['elasticity'] * standard_opbrengst_kg_ref / base_price, 1) #yref
            demand_slope = xr.where(world_market_price_effect, p_elas['elasticity'] * standard_opbrengst_kg_ref_per_crop_world / base_price, demand_slope) #yref   
            demand_slope = xr.where(demand_slope == 0, 1, demand_slope)
        
            demand_intercept  = xr.where(local_market_price_effect, standard_opbrengst_kg_ref - demand_slope * base_price, 0) #yref
            demand_intercept = xr.where(world_market_price_effect, standard_opbrengst_kg_ref_per_crop_world - demand_slope * base_price, demand_intercept)

        #TOT HIER YREF
        #######################
        #HIER BEGIMT YSCE

        ## new production - global for sel == 2 to calculate new price, local to calculate chps
        
        ############################
        #standard_opbrengst_kg_sce is hier dus de mediaan van de actuele opbrengst van sce
        #dat is raar maar was eerst ook zo?
        ######################################
        damage_calculation_yield_from_world_per_crop_year = p_elas['NL aandeel wereldvraag'] * damage_calculation_yield_per_crop_world #new yield local .... per crop and year

        act_yield_local = act_yield_kg_sce.copy()
        act_yield_world_per_crop_year = act_yield_kg_sce_per_crop_world.copy()

        if 'Year' in standard_opbrengst_kg_sce.dims:
            average_standard_yield_local = standard_opbrengst_kg_sce.mean('Year') 
            average_standard_yield_world = standard_opbrengst_kg_sce_per_crop_world.mean('Year')
        else:
            average_standard_yield_local = standard_opbrengst_kg_sce
            average_standard_yield_world = standard_opbrengst_kg_sce_per_crop_world

        if self.optFixedPrice:
            #calibrate the pricefunction on the average potential yield
            demand_slope = xr.where(local_market_price_effect, p_elas['elasticity'] * average_standard_yield_local / base_price, 1)
            demand_slope = xr.where(world_market_price_effect, p_elas['elasticity'] * average_standard_yield_world / base_price, demand_slope)
            demand_slope = xr.where(demand_slope == 0, 1, demand_slope)

            demand_intercept = xr.where(local_market_price_effect, average_standard_yield_local - demand_slope * base_price, 0)
            demand_intercept = xr.where(world_market_price_effect, average_standard_yield_world - demand_slope * base_price, demand_intercept)

        #MvdV 190108 corrected error in calculation nyls: it contained total actual demand in NL, but needs to contain actual demand per area
        total_average_standard_yield_area = average_standard_yield_local.sum('Area')
        total_average_standard_yield_area[total_average_standard_yield_area==0] = 1

        fraction_average_standard_yield = average_standard_yield_local / total_average_standard_yield_area #fraction per area and crop of total average potential production in NL
        
        #hier wordt de regionale productie teruggerekend uit de fractie
        #en de globale productie. Dit wordt gebruikt om de 
        #nieuwe prijs te berekenen.
        new_yield_local_reconstructed = fraction_average_standard_yield * damage_calculation_yield_from_world_per_crop_year

        #new price and change in consumer surplies
        #MvdV 181130 added constant elasticity demand function
        new_price = base_price.copy()

        if self.optConstElas: #constant elasticity y=a*p^b with b=elasd
            act_yield_local = xr.where(act_yield_local==0, 1, act_yield_local)
            act_yield_world_per_crop_year = xr.where(act_yield_world_per_crop_year==0, 1, act_yield_world_per_crop_year)

            new_price = xr.where(local_market_price_effect, (act_yield_local / demand_scale_factor) ** (1/p_elas['elasticity']), new_price)
            new_price = xr.where(world_market_price_effect, (act_yield_world_per_crop_year / demand_scale_factor) ** (1/p_elas['elasticity']), new_price)

            #MvdV 181203 prevent division by zero when b == 1
            alt_elasd = xr.where(p_elas['elasticity'] == -1, -9, p_elas['elasticity'])

            change_consumer_surplus = xr.where(p_elas['elasticity'] == -1, 
                            demand_scale_factor * np.log(base_price) - demand_scale_factor * np.log(new_price),  
                            (demand_scale_factor/(alt_elasd+1)* base_price ** (alt_elasd+1)) - (demand_scale_factor/(alt_elasd+1)* new_price ** (alt_elasd+1)))
            change_consumer_surplus = xr.where(world_market_price_effect, p_elas['NL aandeel wereldvraag'] * change_consumer_surplus, change_consumer_surplus)
            change_consumer_surplus.loc[fixed_price_effect] = 0

        else: #linear demand function y=demand_slope*p+demand_intercept
            new_price = xr.where(local_market_price_effect, (act_yield_local - demand_intercept) / demand_slope, new_price)
            new_price = xr.where(world_market_price_effect, (act_yield_world_per_crop_year - demand_intercept) / demand_slope, new_price)

            #change_consumer_surplus: change in consumer surplus
            change_consumer_surplus = xr.where(local_market_price_effect, -((new_price-base_price)* damage_calculation_yield + 0.5 * (new_price - base_price) * droogtederving_kg_sce), 0) #ysce
            change_consumer_surplus = xr.where(world_market_price_effect, -((new_price-base_price)* new_yield_local_reconstructed + 0.5 * (new_price - base_price) * (standard_opbrengst_kg_sce - new_yield_local_reconstructed)), change_consumer_surplus) #ysce
        
        #change in producer surplus, consumer surplus and total welfare impact
        #variable_irrigation_costs = arbeidskosten + energiekosten
        #MvdV 191119 doubling of variable irrigation costs (for explanation, see p:\11211541-003-kentalleneffecten\Landbouw\011a Effectmodule Landbouw\src\workflow\Omschrijvingen\beregening_factor2)
        variable_irrigation_costs = (1 + (self.Perc_prijspeil_beregening/100)) * 2 * (arbeidskosten_sce + energiekosten_sce)
        #JP 4-9-2025 pak alleen beregeningskosten als deze hoger zijn dan de mediaan (vanwege nieuwe schadedefinitie)
        variable_irrigation_costs_median = variable_irrigation_costs.median(dim='Year')
        variable_irrigation_costs_median_broadcasted = xr.concat([variable_irrigation_costs_median.expand_dims(Year=[year]) for year in variable_irrigation_costs.coords['Year'].values], dim='Year')
        variable_irrigation_costs_def = (variable_irrigation_costs - variable_irrigation_costs_median_broadcasted).where(variable_irrigation_costs >= variable_irrigation_costs_median_broadcasted, 0)

        change_producer_surplus = -variable_irrigation_costs_def + xr.where(p_elas['prijseffect']>0, ((damage_calculation_yield * new_price) - (standard_opbrengst_kg_sce * base_price)), -(droogtederving_kg_sce * base_price))
        change_total_welfare = change_producer_surplus + change_consumer_surplus

        #merge data for saving
        merge_data = [standard_opbrengst_kg_sce, droogtederving_kg_sce, change_producer_surplus, change_consumer_surplus, change_total_welfare, variable_irrigation_costs_def, areaal_sce, base_price.broadcast_like(new_price), new_price]
        merge_columns = ['productie', 'productiederving', 'veranderingen producentensurplus', 'verandering consumenten surplus', 'verandering totale welvaart', 'variabele beregeningskosten', 'areaal', 'prijs', 'nieuwe prijs']
        comb1 = xr.merge([x.rename(n) for x, n in zip(merge_data, merge_columns)])

        # Use cropnamenum to replace index level crops by numbers
        area_names = area_name_num[comb1.Area.values].values

        #make final columns atc
        comb1 = comb1.assign_coords(Area=area_names)
        self.comb1_df = comb1.to_dataframe().sort_index()
        #TOT EIND
        ##############################
        
    def save_output(self, output_fns):
        self.comb1_df.to_csv(output_fns.outnam1_fn, sep = ';', decimal = ',')

        #per area and year
        twpay_df = self.comb1_df['verandering totale welvaart'].unstack('Area').groupby('Year').sum()
        twpay_df.to_csv(output_fns.twpaynam_fn, sep = ';', decimal = ',')

        #per crop and year
        slpcy_df = self.comb1_df['verandering totale welvaart'].unstack('Crop').groupby('Year').sum()
        slpcy_df.to_csv(output_fns.twpcyfil_fn, sep = ';', decimal = ',')

        #per crop and area, average over years
        twpca_df = self.comb1_df['verandering totale welvaart'].unstack('Crop').groupby('Area').mean()
        twpca_df.to_csv(output_fns.twpcafil_fn, sep = ';', decimal = ',')



 # %%
