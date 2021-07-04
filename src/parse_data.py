import logging
import pathlib
from collections import namedtuple

import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras


import database as db


Region = namedtuple('Region', ('code', 'name'))
Topic = namedtuple('Topic', ('name', 'main_topic'))
Country = namedtuple('Country', ('code', 'region_code', 'currency', 'short_name', 'long_name', 'table_name'))
Indicator = namedtuple('Indicator', ('code', 'name', 'topic', 'description'))
Value = namedtuple('Value', ('year', 'country_code', 'indicator_code', 'value'))


def get_regions(df_country):
    region_names = [s for s in set(df_country['Region']) if isinstance(s, str)]
    regions = []
    for r in region_names:
        r_idx = np.where(df_country['Long Name'] ==  r)[0]
        if len(r_idx) == 1:
            region_code = df_country['Country Code'][r_idx[0]]
        elif len(r_idx) == 0:
            logging.warn(f'No `region_code` found for `{r}`. Skipping!')
            break
        else:
            logging.warn(f'`region_code` is not unambiguously for `{r}`. Skipping!')
            break
        regions.append(Region(code=region_code, name=r))
    return regions


def get_currencies(df_country):
    return [s for s in set(df_country['Currency Unit']) if isinstance(s, str)]


def get_countries(df_country, regions, currencies):
    countries = []
    region_lookup = {r.name: r.code for r in regions}
    currencies_set = set(currencies)
    for _, row in df_country.iterrows():
        if row['Region'] == '' or not isinstance(row['Region'], str) or row['Currency Unit'] not in currencies_set:
            continue
        countries.append(Country(code=row['Country Code'],
                                 region_code=region_lookup[row['Region']],
                                 currency=row['Currency Unit'],
                                 short_name=row['Short Name'],
                                 table_name=row['Table Name'],
                                 long_name=row['Long Name']))
    return countries
        

def get_topics(df_series):
    main_topics = set()
    topics = []
    for s in set(df_series['Topic']):
        main_topic = s.split(': ')[0].strip()
        topics.append(Topic(name=s, main_topic=main_topic))
        main_topics.add(main_topic)
    return topics, list(main_topics)


def get_indicators(df_series, topics):
    indicators = []
    topics_set = set(t.name for t in topics)
    for _, row in df_series.iterrows():
        if row['Topic'] not in topics_set:
            continue
        indicators.append(Indicator(code=row['Series Code'],
                                    name=row['Indicator Name'],
                                    topic=row['Topic'],
                                    description=row['Long definition']))
    return indicators


def values_generator(df_data, country_codes, start_year=1960, stop_year=2020):
    for i, row in df_data.iterrows():
        country_code = row['Country Code']
        if country_code not in country_codes:
            continue
        indicator_code = row['Indicator Code']
        for y in range(start_year, stop_year+1):
            value = row[str(y)]
            if not np.isnan(value):
                value = Value(year=y,
                              country_code=country_code,
                              indicator_code=indicator_code,
                              value=value)
                yield value


def parse_wdi_excel(db_cursor, xlsx_file):
    logging.info('\tParsing WDI excel')
    xls = pd.ExcelFile(xlsx_file)#, engine='openpyxl')
    df_country = pd.read_excel(xls, 'Country')
    df_series = pd.read_excel(xls, 'Series')

    # insert regions
    logging.info('\t\tInserting `regions`')
    regions = get_regions(df_country)
    sql = f"INSERT INTO {db.TABLE_REGION_NAME} (region_code, name) VALUES %s"
    values = ((r.code, r.name) for r in regions)
    psycopg2.extras.execute_values(db_cursor, sql, values)

    # insert currencies
    logging.info('\t\tInserting `currencies`')
    currencies = get_currencies(df_country)
    sql = f"INSERT INTO {db.TABLE_CURRENCY_NAME} (name) VALUES %s RETURNING currency_id;"
    currency_ids = psycopg2.extras.execute_values(db_cursor, sql, [(c, ) for c in currencies], fetch=True)
    currency_lookup = {c: i[0] for c, i in zip(currencies, currency_ids)}

    # insert countries
    logging.info('\t\tInserting `countries`')
    countries = get_countries(df_country, regions, currencies)
    sql = f"INSERT INTO {db.TABLE_COUNTRY_NAME} (country_code, region_code, currency_id, short_name, long_name, table_name) VALUES %s;"
    values = [(c.code, c.region_code, currency_lookup[c.currency], c.short_name, c.long_name, c.table_name) for c in countries]
    psycopg2.extras.execute_values(db_cursor, sql, values)

    # insert main_topics, topics
    logging.info('\t\tInserting `main_topics` and `topics`')
    topics, main_topics = get_topics(df_series)
    sql = f"INSERT INTO {db.TABLE_MAIN_TOPIC_NAME} (name) VALUES %s RETURNING main_topic_id;"
    main_topic_ids = psycopg2.extras.execute_values(db_cursor, sql, [(t, ) for t in main_topics], fetch=True)
    main_topic_lookup = {n: i[0] for n, i in zip(main_topics, main_topic_ids)}
    sql = f"INSERT INTO {db.TABLE_TOPIC_NAME}(name, main_topic_id) VALUES %s RETURNING topic_id;"
    values = [(t.name, main_topic_lookup[t.main_topic]) for t in topics]
    topic_ids = psycopg2.extras.execute_values(db_cursor, sql, values, fetch=True)
    topic_lookup = {n.name: i[0] for n, i in zip(topics, topic_ids)}

    # insert indicators
    logging.info('\t\tInserting `indicators`')
    indicators = get_indicators(df_series, topics)
    sql = f"INSERT INTO {db.TABLE_INDICATOR_NAME} (indicator_code, topic_id, name, description) VALUES %s;"
    values = ((i.code, topic_lookup[i.topic], i.name, i.description) for i in indicators)
    psycopg2.extras.execute_values(db_cursor, sql, values)

    del df_country, df_series

    # insert values
    logging.info('\t\tInserting `values`')
    df_data = pd.read_excel(xls, 'Data')
    values = []
    n_values = 0
    sql = f"INSERT INTO {db.TABLE_VALUE_NAME} (year, country_code, indicator_code, value) VALUES %s;"
    for v in values_generator(df_data, set([c.code for c in countries])):
        values.append((v.year, v.country_code, v.indicator_code, v.value))
        if len(values) >= 1000:
            n_values += len(values)
            logging.info(f'\t\t\tAdding values {n_values}/???')
            psycopg2.extras.execute_values(db_cursor, sql, values)
            values = []
    if len(values) > 0:
        n_values += len(values)
        logging.info(f'\t\t\tAdding values {n_values}/{n_values}')
        psycopg2.extras.execute_values(db_cursor, sql, values)


def parse_csv_files(db_cursor, config):
    logging.info('\tParsing CSV files')
    sql = f"INSERT INTO {db.TABLE_MAIN_TOPIC_NAME} (name) VALUES (%s) RETURNING main_topic_id;"
    main_topic_name = 'Whiteboard CSVs'
    db_cursor.execute(sql, (main_topic_name,))
    main_topic_id = db_cursor.fetchone()[0]
    sql = f"INSERT INTO {db.TABLE_TOPIC_NAME} (name, main_topic_id) VALUES (%s, %s) RETURNING topic_id;"
    db_cursor.execute(sql, ('General', main_topic_id))
    topic_id = db_cursor.fetchone()[0]

    #add co2 emission csv
    logging.info('\t\tAdding CO2 values')
    indicator_code = 'KVV.CSV.CO2'
    sql = f"INSERT INTO {db.TABLE_INDICATOR_NAME} (indicator_code, topic_id, name, description) VALUES (%s, %s, %s, %s);"
    db_cursor.execute(sql, (indicator_code,
                            topic_id,
                            'Annual CO2 emissions (tonnes)',
                            'CO2 emissions per contury in tonnes. Data taken from the `co2_emission.csv`'))
    df_co2 = pd.read_csv(pathlib.Path(config['data']['data_folder']) / 'co2_emission.csv')
    sql = f"INSERT INTO {db.TABLE_VALUE_NAME} (year, country_code, indicator_code, value) SELECT %s, %s, %s, %s WHERE EXISTS (SELECT country_code FROM country WHERE country_code = %s);"

    values = []
    df_co2['Code'] = df_co2['Code'].apply(lambda s: str(s))
    for _, row in df_co2.iterrows():
        if len(row['Code']) != 3:
            continue
        db_cursor.execute(sql, (row['Year'], row['Code'], indicator_code, row[-1], row['Code']))


    logging.info('\t\tAdding total pop values')
    indicator_code = 'KVV.CSV.POP'
    sql = f"INSERT INTO {db.TABLE_INDICATOR_NAME} (indicator_code, topic_id, name, description) VALUES (%s, %s, %s, %s);"
    db_cursor.execute(sql, (indicator_code,
                            topic_id,
                            'Total Population',
                            'Total Population. Data taken from the `population_total.csv`'))

    df_pop = pd.read_csv(pathlib.Path(config['data']['data_folder']) / 'population_total.csv')
    sql = f"""INSERT INTO {db.TABLE_VALUE_NAME} (year, country_code, indicator_code, value)
                SELECT %s, country_code, %s, %s
                FROM country
                WHERE table_name = %s;"""
    values = []
    for _, row in df_pop.iterrows():
        db_cursor.execute(sql, (row['Year'], indicator_code, row['Count'], row['Country Name']))

def fill_tables(connection, config):
    logging.info('Add values to tables')
    connection.autocommit = True
    cursor = connection.cursor()
    parse_wdi_excel(cursor, pathlib.Path(config['data']['data_folder']) / 'WDIEXCEL.xlsx')
    parse_csv_files(cursor, config)

