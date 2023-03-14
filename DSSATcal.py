# Imports
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from DSSATTools import *
import h3
from BayDSSAT import CSWconnect
import duckdb
from skopt import gp_minimize
from multiprocessing import Pool

# Extract
def getFields(crop: str, regn: str) -> pd.DataFrame:
   
    query_hss = f"""
    SELECT 
      FIELD_NAME, commercialName, createdBrand, createdMG, protocolNumber, FIELD_Country, FIELD_field_latitude,
      FIELD_field_longitude, FIELD_plantingDate, FIELD_harvestDate, OBS_observationRefCd, OBS_numValue,  
      plot_id, QC_Flag, field_id, FIELD_mac, FIELD_mic, FIELD_uniqueName
    FROM 
      latam_datasets.hss_{regn}_current_{crop} 
    WHERE 
      OBS_observationRefCd in ('YLD','MAT')
    UNION ALL
    SELECT 
      FIELD_NAME, commercialName, createdBrand, createdMG, protocolNumber, FIELD_Country, FIELD_field_latitude,
      FIELD_field_longitude, FIELD_plantingDate, FIELD_harvestDate, OBS_observationRefCd, OBS_numValue,  
      plot_id, QC_Flag, field_id, FIELD_mac, FIELD_mic, FIELD_uniqueName
    FROM 
      latam_datasets.hss_{regn}_historical_{crop} 
    WHERE 
      OBS_observationRefCd in ('YLD','MAT')
    """

    return CSWconnect('bcs-market-dev-lake').load(query_hss)

# Transform
def transFields(df_hss: pd.DataFrame) -> pd.DataFrame:

    df1_field = df_hss[df_hss['createdMG'].notnull()]
    df2_field = df1_field[df1_field['QC_Flag'].isnull()]

    df_field_MAT = df2_field[df2_field['OBS_observationRefCd'] == 'MAT']
    df_field_YLD = df2_field[df2_field['OBS_observationRefCd'] == 'YLD']

    # QC maturity date
    df_field_MAT = df_field_MAT[df_field_MAT['OBS_numValue'] > 90]

    df_field_MAT.set_index('plot_id', drop=False, inplace=True)
    df_field_YLD.set_index('plot_id', drop=False, inplace=True)

    df = pd.merge(df_field_MAT,
                  df_field_YLD['OBS_numValue'],
                  left_index=True,
                  right_index=True)

    df.set_index('FIELD_uniqueName', inplace=True)

    df = df.iloc[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 17, 12]]

    df.rename(
        columns={"OBS_numValue_x": "MAT", "OBS_numValue_y": "YLD"},
        inplace=True,
    )

    df['createdMG'] = df['createdMG'].astype('float')
    df['MG'] = df['createdMG'].apply(np.round_).astype('int')
    df['MAT'] = df['MAT'].astype('int')

    df['HARV'] = (df['FIELD_harvestDate'] -
                  df['FIELD_plantingDate']).astype('timedelta64[D]')

    # Correcting when MAT > Harvest
    df['HARV'][df['MAT'] > df['HARV']] = df['MAT']

    # Removing duplicated values, YM cases
    df = df.loc[-df['plot_id'].duplicated()]

    df['R1_DSSAT'] = np.nan
    df['R2_DSSAT'] = np.nan
    df['R3_DSSAT'] = np.nan
    df['R5_DSSAT'] = np.nan
    df['R7_DSSAT'] = np.nan
    df['R8_DSSAT'] = np.nan
    df['YLD_DSSAT'] = np.nan
    df['pars_DSSAT'] = np.nan
    df['AE_defPars'] = np.nan
    df['AE_calPars'] = np.nan
    df['pars_DSSAT'] = df.pars_DSSAT.astype('object')

    return df

# DSSAT Calibration
def DSSAT_CalPlots(fld):
    try:

        # Field crop level
        print('Processing Field unique name: ' + fld)
        df_field = df[df.index == fld]

        pd = df_field['FIELD_plantingDate'][0].replace(tzinfo=None)
        hd = df_field['FIELD_harvestDate'][0].replace(tzinfo=None)
        lon, lat = df_field['FIELD_field_longitude'][0], df_field['FIELD_field_latitude'][0]
        st = (pd - timedelta(days=30)).date()
        
        # Weather retrieval
        
        query_TWC = f"""
        SELECT 
            *
        FROM 
            read_parquet('s3://s3-latam-gmd-coe/WEATHER/TWC_brazil_soybeans.parquet')
        WHERE
            lat = '{lat}' AND lon = '{lon}' AND date >= '{st.strftime('%Y-%m-%d')}' AND date <= '{hd.strftime('%Y-%m-%d')}' 

        """
        
        df_wth = con.execute(query_TWC).df()      
        df_wth['DOY'] = df_wth['date'].apply(lambda x: int(x.strftime('%j')))

        # Incident solar radiation (Rs_in)
        rad = np.pi/180  # Radians to degrees
        gra = 180/np.pi  # Degrees to radians

        Ko = 37.63*(1+(0.033*(np.cos(rad*((360*df_wth['DOY'])/365)))))
        ds = 23.45*np.sin(rad*(360*(df_wth['DOY']-80)/365))
        hn = (np.arccos(-np.tan(rad*df_wth['lat'][0])*np.tan(rad*ds)))*gra
        Qo = Ko*(rad*hn*np.sin(rad*df_wth['lat'][0])*np.sin(rad*ds) +
                 np.cos(rad*df_wth['lat'][0])*np.cos(rad*ds)*np.sin(rad*hn))

        df_wth['Rs_in'] = 0.16*Qo * \
            ((df_wth['max_temperature'])-(df_wth['min_temperature']))**0.5

        WTH_columns = [
            'date',
            'min_temperature',
            'max_temperature',
            'total_precipitation',
            'Rs_in',
            'avg_relative_humidity',
        ]
        # Create a WeatherData instance
        WTH_DATA = WeatherData(df_wth.loc[:, WTH_columns],
                               variables={
                                   'min_temperature': 'TMIN',
                                   'max_temperature': 'TMAX',
                                   'total_precipitation': 'RAIN',
                                   'Rs_in': 'SRAD',
                                   'avg_relative_humidity': 'RHUM',
        })

        query_elev =f"""
        SELECT 
            elevation 
        FROM 
            read_parquet('s3://s3-latam-gmd-coe/FIELDS/FIELD_brazil_soybeans.parquet')
        WHERE
            FIELD_field_latitude = {lat} AND FIELD_field_longitude = {lon}
        """

        elev = con.execute(query_elev).df().squeeze()

        # Create a WheaterStation instance
        wth = WeatherStation(WTH_DATA, {
            'ELEV': elev,
            'LAT': lat,
            'LON': lon,
            'INSI': 'dpoes'
        })

        print('Weather for Field ' + fld + ' was obtained')
        
        # Soil retrieval
        query_soil = f"""
        SELECT 
            *
        FROM 
            read_parquet('s3://s3-latam-gmd-coe/SOIL/ISRIC_brazil_soybeans.parquet')
        WHERE
            h3_index_10 = '{h3.geo_to_h3(lat=lat, lng=lon, resolution=10)}'
        """

        df_soil = con.execute(query_soil).df()
        # Creating a soil profile instance
        soilprofile = SoilProfile(
            pars={
                'SALB': 0.16,  # Albedo
                'SLU1': 6,  # Stage 1 Evaporation (mm)
                'SLPF': 0.8,  # Soil fertility factor
                'lon': lon,
                'lat': lat,
            })

        layers = [
            SoilLayer(
                0, {
                    'SLCL': df_soil['clyppt_depth_0cm'][0],
                    'SLSI': df_soil['sltppt_depth_0cm'][0]
                }),
            SoilLayer(
                5, {
                    'SLCL': df_soil['clyppt_depth_5cm'][0],
                    'SLSI': df_soil['sltppt_depth_5cm'][0]
                }),
            SoilLayer(
                15, {
                    'SLCL': df_soil['clyppt_depth_15cm'][0],
                    'SLSI': df_soil['sltppt_depth_15cm'][0]
                }),
            SoilLayer(
                30, {
                    'SLCL': df_soil['clyppt_depth_30cm'][0],
                    'SLSI': df_soil['sltppt_depth_30cm'][0]
                }),
            SoilLayer(
                60, {
                    'SLCL': df_soil['clyppt_depth_60cm'][0],
                    'SLSI': df_soil['sltppt_depth_60cm'][0]
                }),
            SoilLayer(
                100, {
                    'SLCL': df_soil['clyppt_depth_100cm'][0],
                    'SLSI': df_soil['sltppt_depth_100cm'][0]
                })
        ]

        for layer in layers:
            soilprofile.add_layer(layer)

        print('Soil profile for Field ' + fld + ' was obtained')
        # Plot cultivar level
        df_field['DSSATcultivar'] = df_field['MG'].apply(
            lambda x: '9900' + f'{x:02d}')
        df_field['CSDL'] = -0.321 * df_field['createdMG'] + 14.51

        pars_name = ['EM-FL', 'FL-SH', 'FL-SD', 'SD-PM']

        for i in range(len(df_field)):

            try:
                print('Running DSSAT for cultivar ' +
                      df_field['DSSATcultivar'][i])
                # DSSAT simulation
                pars = [
                    crop.cultivar[df_field['DSSATcultivar'][i]].get(par)
                    for par in pars_name
                ]
                crop.cultivar[df_field['DSSATcultivar']
                              [i]]['CSDL'] = df_field['CSDL'][i]

                man = Management(cultivar=df_field['DSSATcultivar'][i],
                                 planting_date=pd,
                                 sim_start=WTH_DATA.index[0],
                                 harvest='R',
                                 irrigation='N')

                man.simulation_controls['SYMBI'] = 'Y'
                man.simulation_controls['SMODEL'] = 'CRGRO'
                man.simulation_controls['NITRO'] = 'Y'
                man.harvest_details['table']['HDATE'][0] = hd.strftime('%y%j')

                obs = df_field['MAT'][i]
                # DSSAT calibration
                print('Calibrating DSSAT for cultivar ' +
                      df_field['DSSATcultivar'][i])

                # DSSAT run
                dssat = DSSAT()
                dssat.setup()

                dssat.run(
                    soil=soilprofile,
                    weather=wth,
                    crop=crop,
                    management=man,
                )

                df_out = dssat.output['PlantGro']
                YLD_e = df_out['GWAD'].iloc[-1] / 100  # qq/ha

                # Fixing the error of 'None' value when R8 is not computed
                MAT_day = df_out[df_out.GSTD == 8].first_valid_index()
                MAT_day = df_out[df_out.GSTD == 3].first_valid_index(
                ) if MAT_day is None else MAT_day

                MAT_e = (MAT_day - pd).days

                AE_def = abs(obs - MAT_e)

                print('Absolute error defaut parameters = ' + str(AE_def))

                # Objective function
                def DSSAT_obj(pars):

                    crop.cultivar[df_field['DSSATcultivar']
                                  [i]]['EM-FL'] = pars[0]
                    crop.cultivar[df_field['DSSATcultivar']
                                  [i]]['FL-SH'] = pars[1]
                    crop.cultivar[df_field['DSSATcultivar']
                                  [i]]['FL-SD'] = pars[2]
                    crop.cultivar[df_field['DSSATcultivar']
                                  [i]]['SD-PM'] = pars[3]

                    dssat.run(
                        soil=soilprofile,
                        weather=wth,
                        crop=crop,
                        management=man,
                    )

                    df_out = dssat.output['PlantGro']
                    YLD_e = df_out['GWAD'].iloc[-1] / 100  # qq/ha

                    MAT_day = df_out[df_out.GSTD == 8].first_valid_index()
                    MAT_day = df_out[df_out.GSTD == 3].first_valid_index(
                    ) if MAT_day is None else MAT_day

                    MAT_e = (MAT_day - pd).days

                    print('Absolute error = ' + str(abs(obs - MAT_e)) +
                          '\nField unique name = ' + str(fld) + '\nPlot ID = ' +
                          str(df_field['plot_id'][i]))

                    return abs(obs - MAT_e)

                DSSAT_cal = gp_minimize(
                    func=DSSAT_obj,
                    dimensions=bounds,
                    acq_func='EI',
                    xi=3,
                    initial_point_generator='lhs',
                    acq_optimizer='sampling'
                )

                idx_func = np.where(
                    DSSAT_cal['func_vals'] == DSSAT_cal['func_vals'].min())[0].tolist()
                AE_cal = DSSAT_cal['func_vals'][idx_func[0]]
                pars_cal = [DSSAT_cal['x_iters'][index] for index in idx_func]

                # round parameters
                pars_cal = [[np.round(float(i), 2) for i in nested]
                            for nested in pars_cal]
                print(pars_cal)

                df_field.iat[i, df_field.columns.get_loc(
                    'pars_DSSAT')] = pars_cal
                df_field['AE_defPars'][i] = AE_def
                df_field['AE_calPars'][i] = AE_cal

                # Updating with the calibrated parameters
                crop.cultivar[df_field['DSSATcultivar']
                              [i]]['EM-FL'] = pars_cal[0][0]
                crop.cultivar[df_field['DSSATcultivar']
                              [i]]['FL-SH'] = pars_cal[0][1]
                crop.cultivar[df_field['DSSATcultivar']
                              [i]]['FL-SD'] = pars_cal[0][2]
                crop.cultivar[df_field['DSSATcultivar']
                              [i]]['SD-PM'] = pars_cal[0][3]

                # DSSAT re-run with calibrated parameters
                dssat.run(
                    soil=soilprofile,
                    weather=wth,
                    crop=crop,
                    management=man,
                )

                df_out = dssat.output['PlantGro']

                R1 = (df_out[df_out.GSTD == 1].first_valid_index() - pd).days
                R2 = (df_out[df_out.GSTD == 2].first_valid_index() - pd).days
                R3 = (df_out[df_out.GSTD == 3].first_valid_index() - pd).days
                R5 = (df_out[df_out.GSTD == 5].first_valid_index() - pd).days
                R7 = (df_out[df_out.GSTD == 7].first_valid_index() - pd).days
                R8 = (df_out[df_out.GSTD == 8].first_valid_index() - pd).days
                YLD_e = df_out['GWAD'].iloc[-1] / 100  # qq/ha

                print('R1 estimated = ' + str(R1) + ' dap\n' +
                      'R2 estimated = ' + str(R2) + ' dap\n' +
                      'R3 estimated = ' + str(R3) + ' dap\n' +
                      'R5 estimated = ' + str(R5) + ' dap\n' +
                      'R7 estimated = ' + str(R7) + ' dap\n' +
                      'R8 estimated = ' + str(R8) + ' dap' + '\nR8 observed = ' +
                      str(df_field['MAT'][i]) + ' dap' + '\nYLD estimated = ' +
                      str(YLD_e) + ' qq/ha' + '\nYLD observed = ' +
                      str(df_field['YLD'][i]) + ' qq/ha')

                df_field['R1_DSSAT'][i] = R1
                df_field['R2_DSSAT'][i] = R2
                df_field['R3_DSSAT'][i] = R3
                df_field['R5_DSSAT'][i] = R5
                df_field['R7_DSSAT'][i] = R7
                df_field['R8_DSSAT'][i] = R8
                df_field['YLD_DSSAT'][i] = YLD_e

                dssat.close()

            except Exception as e:
                print(e)
                continue

    except Exception as e:
        print(e)
    
    return df_field

# Load
def loadDSSATpars(calibration: list, crop: str, regn: str) -> None: 
    
    df_cal = pd.concat(calibration)
    df_cal = df_cal[df_cal.R8_DSSAT.notna()]

    df_cal.drop(['MG', 'YLD_DSSAT', 'AE_defPars',
                    'AE_calPars', 'CSDL'], axis=1, inplace=True)

    df_cal = df_cal.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 12, 19, 20]]
    df_cal = df_cal.reset_index()

    df_cal[['FIELD_uniqueName', 'FIELD_NAME', 'commercialName',
                       'protocolNumber', 'plot_id', 'pars_DSSAT', 'DSSATcultivar']] = \
    df_cal[['FIELD_uniqueName', 'FIELD_NAME', 'commercialName',
                           'protocolNumber', 'plot_id', 'pars_DSSAT', 'DSSATcultivar']].astype('string')

    CSWconnect('bcs-market-dev-lake').save(df_cal, f'latam_datasets.dssat_{regn}_{crop}', append=True)

if __name__ == '__main__':
    
    # N computers/hardware tiers
    job = sys.argv[1] 
    n_hards = sys.argv[2] 
    
    # Creating bounds
    crop = Crop('Soybean')

    # Creating bounds
    bounds = [
        (crop.cultivar['999991']['EM-FL'], crop.cultivar['999992']['EM-FL']),
        (crop.cultivar['999991']['FL-SH'], crop.cultivar['999992']['FL-SH']),
        (crop.cultivar['999991']['FL-SD'], crop.cultivar['999992']['FL-SD']),
        (crop.cultivar['999991']['SD-PM'], crop.cultivar['999992']['SD-PM']),
    ]

    # Some configs
    aws_access_key_id=os.environ['aws_key']
    aws_secret_access_key=os.environ['aws_secret']
    aws_region='us-east-2'

    # Connect duckdb
    con = duckdb.connect()
    con.execute(f"""
            INSTALL httpfs;
            LOAD httpfs;
            SET s3_region='us-east-2';
            SET s3_access_key_id='{aws_access_key_id}';
            SET s3_secret_access_key='{aws_secret_access_key}';
            SET threads TO 20;
            """)

    # HSS 
    df_hss = getFields(crop = 'soybeans', regn = 'brazil')

    # Dataframe transformed
    df = transFields(df_hss)

    # Defining the fields
    fields = df.index.unique()
       
    # Number of cores in one machine
    n_cores = 25

    # Number of Fields
    # n_fields=len(fields)

    # Number of jobs/machines started in Domino
    # n_hards = np.ceil(n_fields/n_cores)
        
    # Horizontal scaling for n hardwares
    fields_hard = np.array_split(fields, int(n_hards))
   
    # Parallel processing for n_cores
    with Pool(n_cores) as pool:
        calibration = pool.map(DSSAT_CalPlots, fields_hard[int(job)])
       

    loadDSSATpars(calibration, 'soybeans', 'brazil')    
