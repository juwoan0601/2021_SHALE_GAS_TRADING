TRAIN_DATASET_PATH          = "D:/POSTECH/대외활동/2021 제1회 데이터사이언스경진대회/data/trainSet.csv"
TEST_DATASET_PATH           = "D:/POSTECH/대외활동/2021 제1회 데이터사이언스경진대회/data/examSet.csv"
TARGET_COLUMN_LAST_6_NAME   = 'Last 6 mo. Avg. GAS (Mcf)'
TARGET_COLUMN_FIRST_6_NAME  = 'First 6 mo. Avg. GAS (Mcf)'
PREDICT_COLUMN_NAME         = 'Pred 6 mo. Avg. GAS (Mcf)'
PRICE_COLUMN_NAME           = 'PRICE ($)'
COST_COLUMN_NAME            = 'Per Month Operation Cost ($)'

ALL_COLUMNS = [
    "No",
    "CPA Pretty Well ID",
    "Reference (KB) Elev. (ft)",
    "Ground Elevation (ft)",
    "MD (All Wells) (ft)",
    "TVD (ft)",
    "Bot-Hole direction (N/S)/(E/W)",
    "Bot-Hole Easting (NAD83)",
    "Bot-Hole Northing (NAD83)",
    "On Prod YYYY/MM/DD",
    "First Prod YYYY/MM",
    "Last Prod. YYYY/MM",
    "Stimulation Fluid",
    "Total Proppant Placed (tonne)",
    "Avg Proppant Placed per Stage (tonne)",
    "Total Fluid Pumped (m3)",
    "Avg Fluid Pumped per Stage (m3)",
    "Stages Actual",
    "Completed Length (m)",
    "Avg Frac Spacing (m)",
    "Load Fluid Rec (m3)",
    "Load Fluid (m3)",
    'Avg Fluid Pumped / Meter (m3)',
    'Avg Proppant Placed / Meter (tonne)',
    'Proppant Composition',
    'Proppant Name 1',
    'Proppant Size 1',
    'Avg Proppant 1 Placed (tonne)',
    'Total Proppant 1 Placed (tonne)',
    'Total Ceramic Proppant Placed (tonne)',
    'Total Sand Proppant Placed (tonne)',
    'First 6 mo. Avg. GAS (Mcf)',
    'GAS_MONTH_1',
    'GAS_MONTH_2',
    'GAS_MONTH_3',
    'GAS_MONTH_4',
    'GAS_MONTH_5',
    'GAS_MONTH_6',
    'GAS_MONTH_7',
    'GAS_MONTH_8',
    'GAS_MONTH_9',
    'GAS_MONTH_10',
    'GAS_MONTH_11',
    'GAS_MONTH_12',
    'GAS_MONTH_13',
    'GAS_MONTH_14',
    'GAS_MONTH_15',
    'GAS_MONTH_16',
    'GAS_MONTH_17',
    'GAS_MONTH_18',
    'GAS_MONTH_19',
    'GAS_MONTH_20',
    'GAS_MONTH_21',
    'GAS_MONTH_22',
    'GAS_MONTH_23',
    'GAS_MONTH_24',
    'GAS_MONTH_25',
    'GAS_MONTH_26',
    'GAS_MONTH_27',
    'GAS_MONTH_28',
    'GAS_MONTH_29',
    'GAS_MONTH_30',
    'GAS_MONTH_31',
    'GAS_MONTH_32',
    'GAS_MONTH_33',
    'GAS_MONTH_34',
    'GAS_MONTH_35',
    'GAS_MONTH_36',
    'Last 6 mo. Avg. GAS (Mcf)',
    'CND_MONTH_1',
    'CND_MONTH_2',
    'CND_MONTH_3',
    'CND_MONTH_4',
    'CND_MONTH_5',
    'CND_MONTH_6',
    'CND_MONTH_7',
    'CND_MONTH_8',
    'CND_MONTH_9',
    'CND_MONTH_10',
    'CND_MONTH_11',
    'CND_MONTH_12',
    'CND_MONTH_13',
    'CND_MONTH_14',
    'CND_MONTH_15',
    'CND_MONTH_16',
    'CND_MONTH_17',
    'CND_MONTH_18',
    'CND_MONTH_19',
    'CND_MONTH_20',
    'CND_MONTH_21',
    'CND_MONTH_22',
    'CND_MONTH_23',
    'CND_MONTH_24',
    'CND_MONTH_25',
    'CND_MONTH_26',
    'CND_MONTH_27',
    'CND_MONTH_28',
    'CND_MONTH_29',
    'CND_MONTH_30',
    'CND_MONTH_31',
    'CND_MONTH_32',
    'CND_MONTH_33',
    'CND_MONTH_34',
    'CND_MONTH_35',
    'CND_MONTH_36',
    'HRS_MONTH_1',
    'HRS_MONTH_2',
    'HRS_MONTH_3',
    'HRS_MONTH_4',
    'HRS_MONTH_5',
    'HRS_MONTH_6',
    'HRS_MONTH_7',
    'HRS_MONTH_8',
    'HRS_MONTH_9',
    'HRS_MONTH_10',
    'HRS_MONTH_11',
    'HRS_MONTH_12',
    'HRS_MONTH_13',
    'HRS_MONTH_14',
    'HRS_MONTH_15',
    'HRS_MONTH_16',
    'HRS_MONTH_17',
    'HRS_MONTH_18',
    'HRS_MONTH_19',
    'HRS_MONTH_20',
    'HRS_MONTH_21',
    'HRS_MONTH_22',
    'HRS_MONTH_23',
    'HRS_MONTH_24',
    'HRS_MONTH_25',
    'HRS_MONTH_26',
    'HRS_MONTH_27',
    'HRS_MONTH_28',
    'HRS_MONTH_29',
    'HRS_MONTH_30',
    'HRS_MONTH_31',
    'HRS_MONTH_32',
    'HRS_MONTH_33',
    'HRS_MONTH_34',
    'HRS_MONTH_35',
    'HRS_MONTH_36',
]
STATIC_COLUMNS = [
    "No",
    "CPA Pretty Well ID",
    "Reference (KB) Elev. (ft)",
    "Ground Elevation (ft)",
    "MD (All Wells) (ft)",
    "TVD (ft)",
    "Bot-Hole direction (N/S)/(E/W)",
    "Bot-Hole Easting (NAD83)",
    "Bot-Hole Northing (NAD83)",
    "On Prod YYYY/MM/DD",
    "First Prod YYYY/MM",
    "Last Prod. YYYY/MM",
    "Stimulation Fluid",
    "Total Proppant Placed (tonne)",
    "Avg Proppant Placed per Stage (tonne)",
    "Total Fluid Pumped (m3)",
    "Avg Fluid Pumped per Stage (m3)",
    "Stages Actual",
    "Completed Length (m)",
    "Avg Frac Spacing (m)",
    "Load Fluid Rec (m3)",
    "Load Fluid (m3)",
    'Avg Fluid Pumped / Meter (m3)',
    'Avg Proppant Placed / Meter (tonne)',
    'Proppant Composition',
    'Proppant Name 1',
    'Proppant Size 1',
    'Avg Proppant 1 Placed (tonne)',
    'Total Proppant 1 Placed (tonne)',
    'Total Ceramic Proppant Placed (tonne)',
    'Total Sand Proppant Placed (tonne)',
]
STATIC_COLUMNS_WITHOUT_NAN = [
    "No",
    "CPA Pretty Well ID",
    "Reference (KB) Elev. (ft)",
    "Ground Elevation (ft)",
    "MD (All Wells) (ft)",
    "TVD (ft)",
    "Bot-Hole direction (N/S)/(E/W)",
    "Bot-Hole Easting (NAD83)",
    "Bot-Hole Northing (NAD83)",
    "Total Proppant Placed (tonne)",
    "Avg Proppant Placed per Stage (tonne)",
    "Total Fluid Pumped (m3)",
    "Avg Fluid Pumped per Stage (m3)",
    "Stages Actual",
    "Completed Length (m)",
    "Avg Frac Spacing (m)",
    "Load Fluid Rec (m3)",
    "Load Fluid (m3)",
    'Avg Fluid Pumped / Meter (m3)',
    'Avg Proppant Placed / Meter (tonne)',
    'Avg Proppant 1 Placed (tonne)',
    'Total Proppant 1 Placed (tonne)',
    'Total Ceramic Proppant Placed (tonne)',
    'Total Sand Proppant Placed (tonne)',
]
NAN_COLUMNS = [
    'On Prod YYYY/MM/DD', 
    'First Prod YYYY/MM',
    'Last Prod. YYYY/MM',
    'Stimulation Fluid',
    'Proppant Composition',
    'Proppant Name 1',
    'Proppant Size 1',
]
SERIES_COLUMNS_GAS = [
    'GAS_MONTH_1',
    'GAS_MONTH_2',
    'GAS_MONTH_3',
    'GAS_MONTH_4',
    'GAS_MONTH_5',
    'GAS_MONTH_6',
    'GAS_MONTH_7',
    'GAS_MONTH_8',
    'GAS_MONTH_9',
    'GAS_MONTH_10',
    'GAS_MONTH_11',
    'GAS_MONTH_12',
    'GAS_MONTH_13',
    'GAS_MONTH_14',
    'GAS_MONTH_15',
    'GAS_MONTH_16',
    'GAS_MONTH_17',
    'GAS_MONTH_18',
    'GAS_MONTH_19',
    'GAS_MONTH_20',
    'GAS_MONTH_21',
    'GAS_MONTH_22',
    'GAS_MONTH_23',
    'GAS_MONTH_24',
    'GAS_MONTH_25',
    'GAS_MONTH_26',
    'GAS_MONTH_27',
    'GAS_MONTH_28',
    'GAS_MONTH_29',
    'GAS_MONTH_30',
    'GAS_MONTH_31',
    'GAS_MONTH_32',
    'GAS_MONTH_33',
    'GAS_MONTH_34',
    'GAS_MONTH_35',
    'GAS_MONTH_36',
]
SERIES_COLUMNS_CND = [
    'CND_MONTH_1',
    'CND_MONTH_2',
    'CND_MONTH_3',
    'CND_MONTH_4',
    'CND_MONTH_5',
    'CND_MONTH_6',
    'CND_MONTH_7',
    'CND_MONTH_8',
    'CND_MONTH_9',
    'CND_MONTH_10',
    'CND_MONTH_11',
    'CND_MONTH_12',
    'CND_MONTH_13',
    'CND_MONTH_14',
    'CND_MONTH_15',
    'CND_MONTH_16',
    'CND_MONTH_17',
    'CND_MONTH_18',
    'CND_MONTH_19',
    'CND_MONTH_20',
    'CND_MONTH_21',
    'CND_MONTH_22',
    'CND_MONTH_23',
    'CND_MONTH_24',
    'CND_MONTH_25',
    'CND_MONTH_26',
    'CND_MONTH_27',
    'CND_MONTH_28',
    'CND_MONTH_29',
    'CND_MONTH_30',
    'CND_MONTH_31',
    'CND_MONTH_32',
    'CND_MONTH_33',
    'CND_MONTH_34',
    'CND_MONTH_35',
    'CND_MONTH_36',
]
SERIES_COLUMNS_HRS = [
    'HRS_MONTH_1',
    'HRS_MONTH_2',
    'HRS_MONTH_3',
    'HRS_MONTH_4',
    'HRS_MONTH_5',
    'HRS_MONTH_6',
    'HRS_MONTH_7',
    'HRS_MONTH_8',
    'HRS_MONTH_9',
    'HRS_MONTH_10',
    'HRS_MONTH_11',
    'HRS_MONTH_12',
    'HRS_MONTH_13',
    'HRS_MONTH_14',
    'HRS_MONTH_15',
    'HRS_MONTH_16',
    'HRS_MONTH_17',
    'HRS_MONTH_18',
    'HRS_MONTH_19',
    'HRS_MONTH_20',
    'HRS_MONTH_21',
    'HRS_MONTH_22',
    'HRS_MONTH_23',
    'HRS_MONTH_24',
    'HRS_MONTH_25',
    'HRS_MONTH_26',
    'HRS_MONTH_27',
    'HRS_MONTH_28',
    'HRS_MONTH_29',
    'HRS_MONTH_30',
    'HRS_MONTH_31',
    'HRS_MONTH_32',
    'HRS_MONTH_33',
    'HRS_MONTH_34',
    'HRS_MONTH_35',
    'HRS_MONTH_36',
]
ID_COLUMNS = [
    "No",
    "CPA Pretty Well ID"
]
STATIC_COLUMNS_WITHOUT_NAN_ID = [
    "Reference (KB) Elev. (ft)",
    "Ground Elevation (ft)",
    "MD (All Wells) (ft)",
    "TVD (ft)",
    "Bot-Hole direction (N/S)/(E/W)",
    "Bot-Hole Easting (NAD83)",
    "Bot-Hole Northing (NAD83)",
    "Total Proppant Placed (tonne)",
    "Avg Proppant Placed per Stage (tonne)",
    "Total Fluid Pumped (m3)",
    "Avg Fluid Pumped per Stage (m3)",
    "Stages Actual",
    "Completed Length (m)",
    "Avg Frac Spacing (m)",
    "Load Fluid Rec (m3)",
    "Load Fluid (m3)",
    'Avg Fluid Pumped / Meter (m3)',
    'Avg Proppant Placed / Meter (tonne)',
    'Avg Proppant 1 Placed (tonne)',
    'Total Proppant 1 Placed (tonne)',
    'Total Ceramic Proppant Placed (tonne)',
    'Total Sand Proppant Placed (tonne)',
]
