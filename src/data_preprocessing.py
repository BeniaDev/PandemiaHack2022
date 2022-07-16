import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import logging

fillna_zero_cols_list = ['ivl_per_100k',
                         'ivl_number',
                         'ekmo_per_100k',
                         'ekmo_number',
                         'epirank_avia',
                         'epirank_bus',
                         'epirank_train',
                         'epirank_avia_cat',
                         'epirank_bus_cat',
                         'epirank_train_cat']

fillna_median_cols_list = ['num_patients_tubercul_1992',
                           'num_patients_tubercul_1993',
                           'num_patients_tubercul_1994',
                           'num_patients_tubercul_1995',
                           'num_patients_tubercul_1996',
                           'num_patients_tubercul_1997',
                           'num_patients_tubercul_1998',
                           'num_patients_tubercul_1999',
                           'num_patients_tubercul_2000',
                           'num_patients_tubercul_2001',
                           'num_patients_tubercul_2002',
                           'num_patients_tubercul_2003',
                           'num_patients_tubercul_2004',
                           'num_patients_tubercul_2005',
                           'num_patients_tubercul_2006',
                           'num_patients_tubercul_2007',
                           'num_patients_tubercul_2008',
                           'num_patients_tubercul_2009',
                           'num_patients_tubercul_2010',
                           'num_patients_tubercul_2011',
                           'num_patients_tubercul_2012',
                           'num_patients_tubercul_2013',
                           'num_patients_tubercul_2014',
                           'num_patients_tubercul_2015',
                           'num_patients_tubercul_2016',
                           'num_patients_tubercul_2017',
                           'cleanness',
                           'public_services',
                           'neighbourhood',
                           'children_places',
                           'sport_and_outdoor',
                           'shops_and_malls',
                           'public_transport',
                           'security',
                           'life_costs',
                           'life_quality_place_rating',
                           'ecology']

trash_cols_list = ["subject", "has_metro"]

TARGET = ['inf_rate'] # (+)

# Float features
LAT_LNG = ['lat', 'lng'] # (+)

URBAN_RURAL = ['urban', 'rural'] # количество городских и сельских жителей (+)

HAS_METRO = ['has_metro'] # (+)

POPULATION = ['population', 'whole_population']
#whole_population относится к subject, а population - это данные населения города (относится к name)

DENSITY = ['density'] # Плотность населения

RATINGS = ['cleanness', 'public_services', 'neighbourhood', 'children_places', 'sport_and_outdoor',
           'shops_and_malls', 'public_transport', 'security', 'life_costs', 'life_quality_place_rating',
           'ecology'] # Рейтинги

VENTILATOR = ['ivl_per_100k', 'ivl_number', 'ekmo_per_100k', 'ekmo_number'] # ИВЛ
#количество аппаратов искусственной вентиляции легких в абсолютном выражении,
# на 100 тыс. человек населения, количество оборудования для ЭКМО — в абсолютном выражении, на 100 тыс. человек населения.
#

WEATHER = [ 'avg_temp_min', 'avg_temp_max', 'avg_temp_std', 'avg_temp_median', 'humidity_min',
           'humidity_max', 'humidity_std', 'humidity_median', 'pressure_min', 'pressure_max',
           'pressure_std', 'pressure_median', 'wind_speed_ms_min', 'wind_speed_ms_max',
           'wind_speed_ms_std', 'wind_speed_ms_median'] # Погода за март 2020 года

RESIDENTS_AGE = ['urban_50-54_years', 'urban_55-59_years', 'urban_60-64_years', 'urban_65-69_years',
                 'urban_70-74_years', 'urban_75-79_years', 'urban_80-84_years', 'urban_85-89_years',
                 'urban_90-94_years', 'rural_50-54_years', 'rural_55-59_years', 'rural_60-64_years',
                 'rural_65-69_years', 'rural_70-74_years', 'rural_75-79_years', 'rural_80-84_years',
                 'rural_85-89_years', 'rural_90-94_years'] # количество жителей по возрастным группам и районам проживания (город/село)

POPULATION_WORK = ['work_ratio_15-72_years', 'work_ratio_55-64_years', 'work_ratio_15-24_years',
                   'work_ratio_15-64_years', 'work_ratio_25-54_years',] # Занятость населения по возрастам

HISTORICAL_TUBERCUL = [ 'num_patients_tubercul_1992', 'num_patients_tubercul_1993', 'num_patients_tubercul_1994',
                       'num_patients_tubercul_1995', 'num_patients_tubercul_1996', 'num_patients_tubercul_1997',
                       'num_patients_tubercul_1998', 'num_patients_tubercul_1999', 'num_patients_tubercul_2000',
                       'num_patients_tubercul_2001', 'num_patients_tubercul_2002', 'num_patients_tubercul_2003',
                       'num_patients_tubercul_2004', 'num_patients_tubercul_2005', 'num_patients_tubercul_2006',
                       'num_patients_tubercul_2007', 'num_patients_tubercul_2008', 'num_patients_tubercul_2009',
                       'num_patients_tubercul_2010', 'num_patients_tubercul_2011', 'num_patients_tubercul_2012',
                       'num_patients_tubercul_2013', 'num_patients_tubercul_2014', 'num_patients_tubercul_2015',
                       'num_patients_tubercul_2016', 'num_patients_tubercul_2017']  # количество больных туберкулезом
                       # в населенных пунктах по годам

ECONOMIC_VALUE = ['volume_serv_household_2017', 'volume_serv_chargeable_2017', 'volume_serv_transport_2017',
                  'volume_serv_post_2017', 'volume_serv_accommodation_2017', 'volume_serv_telecom_2017',
                  'volume_serv_others_2017', 'volume_serv_veterinary_2017', 'volume_serv_housing_2017',
                  'volume_serv_education_2017', 'volume_serv_medicine_2017', 'volume_serv_disabled_2017',
                  'volume_serv_culture_2017', 'volume_serv_sport_2017', 'volume_serv_hotels_2017',
                  'volume_serv_tourism_2017', 'volume_serv_sanatorium_2017']  # Экономическая активность

NUM_PHONES = ['num_phones_rural_2018', 'num_phones_urban_2019'] # количество телефонов в разбивке по городским и сельским районам

BAS_TRAVEL = ['bus_march_travel_18', 'bus_april_travel_18'] # пассажирооборот автобусов по маршрутам регулярных перевозок (тысяча пассажиро-километров)

ENVIROMENTAL_SAFETY = ['epirank_avia', 'epirank_bus', 'epirank_train', 'epirank_avia_cat',
                       'epirank_bus_cat', 'epirank_train_cat'] # рейтинг экологической безопасности, индекс epirank

# Cat features
DISTRICT = ['district'] # Округ
SUBJECT = ['subject'] # Регион
TOWN = ['name'] # Город
REGION = ['region_x'] # по сути тоже самое что и SUBJECT

NUMERIC_FEATURES = LAT_LNG + URBAN_RURAL + POPULATION + DENSITY \
                 + RATINGS + VENTILATOR + WEATHER + RESIDENTS_AGE + POPULATION_WORK \
                 + HISTORICAL_TUBERCUL + ECONOMIC_VALUE + NUM_PHONES + BAS_TRAVEL \
                 + ENVIROMENTAL_SAFETY

def train_drop(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['inf_rate'].notnull()]
    df.drop([362, 363, 364], inplace=True)
    df.drop([40, 39, 38], inplace=True)
    df.drop([372, 371, 370], inplace=True)
    df.drop([283, 282, 281], inplace=True)
    df.drop([120, 119, 118], inplace=True)
    df.drop([243, 242, 241], inplace=True)
    df.drop([68], inplace=True)
    df.drop([230], inplace=True)

    return df

def clean_df(df: pd.DataFrame, fillna_zero_cols: list, fillna_median_cols: list, trash_cols: list) -> pd.DataFrame:
    for col in fillna_zero_cols:
        df[col] = df[col].fillna(0)

    for col in fillna_median_cols:
        df[col] = df[col].fillna(df[col].median())

    # убираем trash_cols
    df = df.drop(columns=trash_cols)

    return df

def normalize_numeric(df: pd.DataFrame, numeric_features: list) -> pd.DataFrame:
    # load the model from disk
    loaded_MinMaxScalers = pickle.load(open("./models/" + "MinMaxScalers.sav", 'rb'))
    numeric_normalizers = {}
    for col in numeric_features:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[col].values.reshape((-1, 1)))
        numeric_normalizers[col] = scaler

    # saving pickle of dict with MinMaxScaler for every col in df
    model_filename = "MinMaxScalers.sav"
    saved_model = pickle.dump(numeric_normalizers, open(model_filename, 'wb'))
    logging.info('Model is saved into to disk successfully Using Pickle')

    return df


if __name__ == "__main__":
    df = pd.read_csv("./data/" + "covid_data_train.csv")
    df = clean_df(df, fillna_zero_cols_list, fillna_median_cols_list, trash_cols_list)
    df = normalize_numeric(df, NUMERIC_FEATURES)
    print(df)









