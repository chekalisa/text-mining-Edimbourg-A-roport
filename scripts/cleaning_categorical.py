import pandas as pd
import numpy as np


def map_agree_disagree(df, columns):
    '''
    This function maps agree/disagree options in specified columns to 0/1 values

    Arguments:
    df (DataFrame): the input Dataframe
    columns (list): a list of column names to map

    Returns: 
    a DataFrame with mapped values
    '''
    mapping = {'Strongly disagree': 0, 'Disagree': 0, 'Somewhat disagree': 0,
               'Somewhat agree': 1, 'Agree': 1, 'Strongly agree': 1}
    for col in columns:
        df[col] = df[col].map(mapping)
    return df

def map_likely_unlikely(df, columns):
    '''
    This function maps likely/unlikely options in specified columns to 0/1 values

    Arguments:
    df (DataFrame): the input Dataframe
    columns (list): a list of column names to map

    Returns:
    a DataFrame: DataFrame with mapped values
    '''
    mapping = {'Very unlikely': 0, 'Unlikely': 0, 'Somewhat likely': 1,
               'Likely': 1, 'Very likely': 1}
    for col in columns:
        df[col] = df[col].map(mapping)
    return df

def cleaning(df):
    '''
    This function cleans and preprocesses the input DataFrame

    Arguments:
    df (DataFrame): the input dataframe

    Returns:
    DataFrame: the cleaned DataFrame
    '''
    df.drop_duplicates(keep='first', inplace=True)
    df = df.drop(columns=['#', 'Start Date (UTC)', 'Submit Date (UTC)', 'What country were you travelling to? ', 
                          'What country do you live in? ', 'Do you have any other feedback or suggestions?',
                          'Are there any other premium services that you would like to see introduced at Edinburgh?'])
    
    columns_selected = [col for col in df.columns if df[col].isna().sum() / len(df) * 100 < 10]
    df = df[columns_selected].copy()

    column_mapping = {'Overall, were you satisfied with your most recent experience at Edinburgh Airport?  ': 'satisfaction_binary',
                      'Did you arrive into, or depart from Edinburgh Airport?': 'arrival_departure_both',
                      'What age group do you fall into?  ': 'age',
                      'When do you think you will fly next from Edinburgh Airport?': 'next_flight_from_airport',
                      'How did you get to and/or from the airport?': 'transport_to_from_airport',
                      'Including yourself, how many people were in your travel party?': 'travellers_number'}
    df.rename(columns=column_mapping, inplace=True)
    df.dropna(subset=['travellers_number'], inplace=True)

    mapping_y = {'Yes': 1, 'No': 0}
    df['satisfaction_binary'] = df['satisfaction_binary'].map(mapping_y)

    agree_disagree_columns = ['"It was easy to get to / from the airport"', 
                              '"It was easy to find information about my flight"',
                              '"It was easy to find my way around the airport"',
                              '"I was satisfied with the staff service I received"',
                              '"I was satisfied with the cleanliness of the airport"',
                              '"It was easy to find a seat at the airport"',
                              '"It was easy to recycle at Edinburgh Airport"']
    df = map_agree_disagree(df, agree_disagree_columns)

    likely_unlikely_columns = ['Meet and greet / airport concierge service', 'Luggage porterage',
                               'Private terminal with airside vehicle collection/pick-up',
                               'Home collection baggage services', 'Priority queuing']
    df = map_likely_unlikely(df, likely_unlikely_columns)

    age_dummies = pd.get_dummies(df['age'], prefix="age", drop_first=True)
    next_flight_dummies = pd.get_dummies(df["next_flight_from_airport"], prefix="next_flight", drop_first=True)
    travellers_dummies = pd.get_dummies(df['travellers_number'], prefix="num_trav", drop_first=True)
    transport_dummies = pd.get_dummies(df['transport_to_from_airport'], prefix="transport", drop_first=True)
    arrivaldeparture_dummies = pd.get_dummies(df["arrival_departure_both"], prefix="arrdep", drop_first=True)
    age_dummies = age_dummies.astype(int)
    next_flight_dummies = next_flight_dummies.astype(int)
    travellers_dummies = travellers_dummies.astype(int)
    transport_dummies = transport_dummies.astype(int)
    arrivaldeparture_dummies = arrivaldeparture_dummies.astype(int)

    df = pd.concat([df.drop(columns=["age", "next_flight_from_airport", "travellers_number",
                                     "arrival_departure_both", "transport_to_from_airport"]), 
                    age_dummies, next_flight_dummies, travellers_dummies,
                    arrivaldeparture_dummies, transport_dummies], axis=1)
    return df

