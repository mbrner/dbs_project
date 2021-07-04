import logging

import click
import toml
from app import build_app

from database import check_database


@click.command()
@click.argument('config', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--recreate-db/--not-recreate-db', 'recreate_db', default=False)
def run(config, recreate_db):
    config = toml.load(config)
    config['cli_options'] = {
        'recreate_db': recreate_db
    }
    connection, tables = check_database(config)
    app = build_app(connection)
    app.run_server(host="0.0.0.0", debug=True, dev_tools_hot_reload=False)
    connection.close()

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    run()

