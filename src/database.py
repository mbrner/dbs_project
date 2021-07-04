from functools import partial
import logging
from collections import namedtuple
import socket
import time
import datetime

import psycopg2
import psycopg2.extras

from parse_data import fill_tables


Table = namedtuple('table', ['name', 'create_cmd'])

TABLE_REGION_NAME = "region"
TABLE_REGION = f"""CREATE TABLE IF NOT EXISTS {TABLE_REGION_NAME} (
    region_code char(3) NOT NULL,
    name varchar(128),
    PRIMARY KEY (region_code)
    );"""


TABLE_COUNTRY_NAME = "country"
TABLE_COUNTRY = f"""CREATE TABLE IF NOT EXISTS {TABLE_COUNTRY_NAME} (
    country_code char(3) NOT NULL,
    region_code char(3) REFERENCES region(region_code),
    currency_id integer REFERENCES currency(currency_id),
    short_name varchar(255),
    table_name varchar(255),
    long_name varchar(255),
    PRIMARY KEY (country_code),
    FOREIGN KEY (region_code) REFERENCES region(region_code),
    FOREIGN KEY (currency_id) REFERENCES currency(currency_id)
    );"""


TABLE_CURRENCY_NAME = "currency"
TABLE_CURRENCY = f"""CREATE TABLE IF NOT EXISTS {TABLE_CURRENCY_NAME} (
    currency_id SERIAL,
    name varchar(128),
    PRIMARY KEY (currency_id)
    );"""


TABLE_INDICATOR_NAME = "indicator"
TABLE_INDICATOR = f"""CREATE TABLE IF NOT EXISTS {TABLE_INDICATOR_NAME} (
    indicator_code varchar(128),
    topic_id INT REFERENCES topic(topic_id),
    name varchar(256),
    description text,
    PRIMARY KEY (indicator_code),
    FOREIGN KEY (topic_id) REFERENCES topic(topic_id)
    );"""


TABLE_MAIN_TOPIC_NAME = "main_topic"
TABLE_MAIN_TOPIC = f"""CREATE TABLE IF NOT EXISTS {TABLE_MAIN_TOPIC_NAME} (
    main_topic_id SERIAL,
    name varchar(128),
    PRIMARY KEY (main_topic_id)
    );"""

TABLE_TOPIC_NAME = "topic"
TABLE_TOPIC = f"""CREATE TABLE IF NOT EXISTS {TABLE_TOPIC_NAME} (
    topic_id SERIAL,
    name varchar(128),
    main_topic_id INT REFERENCES topic(topic_id),
    PRIMARY KEY (topic_id),
    FOREIGN KEY (main_topic_id) REFERENCES main_topic(main_topic_id)
    );"""


TABLE_VALUE_NAME = "value"
TABLE_VALUE = f"""CREATE TABLE IF NOT EXISTS {TABLE_VALUE_NAME} (
    value_id SERIAL,
    year INT NOT NULL,
    country_code char(3) NOT NULL,
    indicator_code varchar(128) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (value_id),
    FOREIGN KEY (country_code) REFERENCES country(country_code),
    FOREIGN KEY (indicator_code) REFERENCES indicator(indicator_code)
    );"""


TABLES = [Table(TABLE_REGION_NAME, TABLE_REGION),
          Table(TABLE_CURRENCY_NAME, TABLE_CURRENCY),
          Table(TABLE_MAIN_TOPIC_NAME, TABLE_MAIN_TOPIC),
          Table(TABLE_TOPIC_NAME, TABLE_TOPIC),
          Table(TABLE_COUNTRY_NAME, TABLE_COUNTRY),
          Table(TABLE_INDICATOR_NAME, TABLE_INDICATOR),
          Table(TABLE_VALUE_NAME, TABLE_VALUE)]

def connect_to_database(database_name,
                        user,
                        password,
                        host,
                        port,
                        master_db='postgres',
                        recreate_db=False):

    def connect_wait_retry(database, host, user, password, port, retries=5, wait=5):
        for i in range(retries):
            try:
                return psycopg2.connect(database=database, host=host, user=user, password=password, port=port)
            except psycopg2.OperationalError as err:
                if 'database system is starting up' in str(err):
                    time.sleep(wait)
                else:
                    raise err
        raise psycopg2.OperationalError(f'No connection to database after {retries} retries')


    connect_to_db = partial(connect_wait_retry, host=host, user=user, password=password, port=port)
    if recreate_db:
        logging.info(f'Recreating database {database_name}')
        con_master = connect_to_db(database=master_db)
        con_master.autocommit = True
        cursor = con_master.cursor()
        cursor.execute(f'DROP DATABASE IF EXISTS {database_name}')
        con_master.close()
    try:
        con = connect_to_db(database=database_name)
        created_database = False
    except psycopg2.OperationalError:
        logging.info(f'Creating database {database_name}')
        con_master = connect_to_db(database=master_db)
        con_master.autocommit = True
        cursor = con_master.cursor()
        cursor.execute(f"CREATE database {database_name}")
        con_master.close()
        con = connect_to_db(database=database_name)
        created_database = True
    return con, created_database


def check_tables(cursor, create=False):
    missing_tables = False
    for table in TABLES:
        cursor.execute("select * from information_schema.tables where table_name=%s", (table.name,))
        if not cursor.fetchone():
            missing_tables = True
            if create:
                logging.info(f'Creating table {table.name}')
                cursor.execute(table.create_cmd)
            else:
                break
    return missing_tables


def wait_for_db(host, port, timeout=60):
    logging.info('Waiting for database')
    wait_time = datetime.timedelta(seconds=timeout)
    start = datetime.datetime.now()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    got_connection = False
    while datetime.datetime.now() < start + wait_time:
        try:
            s.connect((host, port))
            s.close()
            got_connection = True
            break
        except socket.error as ex:
            time.sleep(0.1)
    return got_connection


def check_database(config):
    db_rdy = wait_for_db(config['postgres']['host'], int(config['postgres']['port']))
    if not db_rdy:
        raise RuntimeError('Database not ready!')
    else:
        logging.info('Database ready!')
    connection, created_database = connect_to_database(**config['postgres'], recreate_db=config['cli_options']['recreate_db'])
    connection.autocommit = True
    cursor = connection.cursor()
    missing_tables = check_tables(cursor, create=False)
    if missing_tables and not created_database:
        connection.close()
        connection, created_database = connect_to_database(**config['postgres'], recreate_db=True)
    if created_database:
        created_tables = check_tables(cursor, create=True)
    else:
        created_tables = False
    if created_tables:
        fill_tables(connection, config)
    return connection, TABLES
