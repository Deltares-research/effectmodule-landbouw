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

        y_ref =areaal_ref*k6_ref*self.multpotyield
        y_sce = areaal_sce*k6_sce*self.multpotyield
        #######################

        derv = areaal_sce*self.cropdata_split_sce['derv_opbrengst']*self.multredyield

        arbeidskosten_sce = areaal_sce*self.cropdata_split_sce['arbeidskosten']
        energiekosten_sce = areaal_sce*self.cropdata_split_sce['energiekosten']
            
        #step 2 determines the annual price based on the actual production and calculate the welfare impact
        #calibrate the price function
        Prijstool_elasticity_single_crops.columns.name = 'Crop'
        p_elas = Prijstool_elasticity_single_crops.T.sort_index().to_xarray()
        p_elas[['NL aandeel wereldaanbod', 'NL aandeel wereldvraag']] = xr.where(p_elas[['NL aandeel wereldaanbod', 'NL aandeel wereldvraag']]<=0, 1, p_elas[['NL aandeel wereldaanbod', 'NL aandeel wereldvraag']])

        #set masks for the different price effects
        pe1 = p_elas['prijseffect']==1
        pe2 = p_elas['prijseffect']==2

        #do sums over area
        #DIT MOETEN EEN YREF EN EEN YSCE VARIANT WORDEN
        #############################

        ypcy_ref = y_ref.sum('Area')
        ypcy_sce = y_sce.sum('Area')

        ygpcy_ref = ypcy_ref / p_elas['NL aandeel wereldaanbod']
        ygpcy_sce = ypcy_sce / p_elas['NL aandeel wereldaanbod']


        dervpcy = derv.sum('Area')

        pri = self.updated_crop_prices.set_index('regio_nummer')
        pri = pri[cropnam]
        pri = pri.stack(0)
        pri.index.names = ['Area', 'Crop']
        pri = pri.swaplevel(0, 1).sort_index().to_xarray()

        ################
        #DIT IS YREF

        #MvdV 181130 added constant elasticity demand function
        if self.optConstElas: #constant elasticity y=a*p^b with b=elasd
        #MvdV 181203 make a copy of the elasticities to modify them later
            a = xr.where(pe1, y_ref / ( pri ** p_elas['elasticity']), 0) #yref
            a = xr.where(pe2, ygpcy_ref / ( pri ** p_elas['elasticity']), a) #yref
            a = xr.where(a==0, 1, a)

        else: #linear demand function y=sd*p+id
            #sd slope of the demand curve
            #id intercept of the demand curve
            sd = xr.where(pe1, p_elas['elasticity'] * y_ref / pri, 1) #yref
            sd = xr.where(pe2, p_elas['elasticity'] * ygpcy_ref / pri, sd) #yref   
            sd = xr.where(sd == 0, 1, sd)
        
            id  = xr.where(pe1, y_ref - sd * pri, 0) #yref
            id = xr.where(pe2, ygpcy_ref - sd * pri, id)

        #TOT HIER YREF
        #######################
        #HIER BEGIMT YSCE

        ## new production - global for sel == 2 to calculate new price, local to calculate chps
        
        ############################
        #y_sce is hier dus de mediaan van de actuele opbrengst van sce
        #dat is raar maar was eerst ook zo?
        ######################################

        nylp = y_sce - derv 
        nygpcy = ygpcy_sce - dervpcy #new yield global per crop and year
        nylspcy = p_elas['NL aandeel wereldvraag'] * nygpcy #new yield local .... per crop and year

        if 'Year' in y_sce.dims:
            ypca = y_sce.mean('Year') 
            ygpc = ygpcy_sce.mean('Year')	
        else:
            ypca = y_sce
            ygpc = ygpcy_sce

        if self.optFixedPrice:
            #calibrate the pricefunction on the average potential yield
            sd = xr.where(pe1, p_elas['elasticity'] * ypca / pri, 1)
            sd = xr.where(pe2, p_elas['elasticity'] * ygpc / pri, sd)
            sd = xr.where(sd == 0, 1, sd)

            id = xr.where(pe1, ypca - sd * pri, 0)
            id = xr.where(pe2, ygpc - sd * pri, id)

        #MvdV 190108 corrected error in calculation nyls: it contained total actual demand in NL, but needs to contain actual demand per area
        ypc = ypca.sum('Area')
        ypc[ypc==0] = 1

        frypca = ypca / ypc #fraction per area and crop of total average potential production in NL
        
        #hier wordt de regionale productie teruggerekend uit de fractie
        #en de globale productie. Dit wordt gebruikt om de 
        #nieuwe prijs te berekenen.
        nyls = frypca * nylspcy

        #new price and change in consumer surplies
        #MvdV 181130 added constant elasticity demand function
        npri = y_sce.copy()*0+pri

        if self.optConstElas: #constant elasticity y=a*p^b with b=elasd
            nylp = xr.where(nylp==0, 1, nylp)
            nygpcy = xr.where(nygpcy==0, 1, nygpcy)

            npri = xr.where(pe1, (nylp / a) ** (1/p_elas['elasticity']), npri)
            npri = xr.where(pe2, (nygpcy / a) ** (1/p_elas['elasticity']), npri)

            #MvdV 181203 prevent division by zero when b == 1
            alt_elasd = xr.where(p_elas['elasticity'] == -1, -9, p_elas['elasticity'])

            chcs = xr.where(p_elas['elasticity'] == -1, 
                            a * np.log(pri) - a * np.log(npri),  
                            (a/(alt_elasd+1)* pri ** (alt_elasd+1)) - (a/(alt_elasd+1)* npri ** (alt_elasd+1)))
            chcs = xr.where(pe2, p_elas['NL aandeel wereldvraag'] * chcs, chcs)
            chcs.loc[p_elas['prijseffect'] == 0] = 0

        else: #linear demand function y=sd*p+id
            npri = xr.where(pe1, (nylp - id) / sd, npri)
            npri = xr.where(pe2, (nygpcy - id) / sd, npri)

            #chcs: change in consumer surplus
            chcs = xr.where(pe1, -((npri-pri)* nylp + 0.5 * (npri - pri) * (y_sce - nylp)), 0) #ysce
            chcs = xr.where(pe2, -((npri-pri)* nyls + 0.5 * (npri - pri) * (y_sce - nyls)), chcs) #ysce
            
        #change in producer surplus, consumer surplus and total welfare impact
        #varirricosts = arbeidskosten + energiekosten
        #MvdV 191119 doubling of variable irrigation costs (for explanation, see p:\11211541-003-kentalleneffecten\Landbouw\011a Effectmodule Landbouw\src\workflow\Omschrijvingen\beregening_factor2)
        varirricosts = (1 + (self.Perc_prijspeil_beregening/100)) * 2 * (arbeidskosten_sce + energiekosten_sce)
        #JP 4-9-2025 pak alleen beregeningskosten als deze hoger zijn dan de mediaan (vanwege nieuwe schadedefinitie)
        varirricosts_median = varirricosts.median(dim='Year')
        varirricosts_median_broadcasted = xr.concat([varirricosts_median.expand_dims(Year=[year]) for year in varirricosts.coords['Year'].values], dim='Year')
        varirricosts_def = (varirricosts - varirricosts_median_broadcasted).where(varirricosts >= varirricosts_median_broadcasted, 0)

        chps = -varirricosts_def + xr.where(p_elas['prijseffect']>0, ((nylp * npri) - (y_sce * pri)), -(derv * pri))
        chtw = chps + chcs

        #merge data for saving
        merge_data = [y_sce, derv, chps, chcs, chtw, varirricosts_def, areaal_sce, pri.broadcast_like(npri), npri]
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
