import logging
from math import isfinite

import dash
import dash_table
import time
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
from kats.tsfeatures.tsfeatures import TsFeatures
from kats.consts import TimeSeriesData
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


TS_FEATURES = ["length",
               "mean",
               "var",
               "entropy",
               "lumpiness",
               "stability",
               "flat_spots",
               "hurst",
               "std1st_der",
               "crossing_points",
               "binarize_mean",
               "unitroot_kpss",
               "heterogeneity",
               "histogram_mode",
               "linearity",
               "cusum_num",
               "cusum_conf",
               "cusum_cp_index",
               "cusum_delta",
               "cusum_llr",
               "cusum_regression_detected",
               "cusum_stable_changepoint",
               "cusum_p_value"]


def fetch_data(db_cursor, countries, indicator, years):
    logging.info('Fetching Data for line plot!')
    db_cursor.execute(f"SELECT year, value, country_code FROM value WHERE indicator_code=%s and country_code in %s and year >= %s and year <= %s ORDER BY country_code, year;", (indicator, tuple(countries), *years))
    df = pd.DataFrame(db_cursor.fetchall(), columns=['year', 'value', 'country_code'])
    return df


def fetch_data_corr(db_cursor, countries, indicator_x, indicator_y, years):
    logging.info('Fetching Data for scatter plot!')
    db_cursor.execute(f"SELECT year, value, country_code, indicator_code FROM value WHERE indicator_code in %s and country_code in %s and year >= %s and year <= %s ORDER BY country_code, year, indicator_code;", ((indicator_x, indicator_y), tuple(countries), *years))
    df = pd.DataFrame(db_cursor.fetchall(), columns=['year', 'value', 'country_code', 'indicator_code'])
    return df


def fetch_data_agg(db_cursor, countries, indicator, years):
    logging.info('Fetching Data for scatter plot!')
    db_cursor.execute(f"SELECT year, value, country_code FROM value WHERE indicator_code=%s and year >= %s and year <= %s ORDER BY country_code, year;", (indicator, *years))
    df = pd.DataFrame(db_cursor.fetchall(), columns=['year', 'value', 'country_code'])
    rows = []
    logging.info('Calculating Timeseries features!')
    model = TsFeatures(selected_features=TS_FEATURES)
    for country, group in df.groupby('country_code'):
        try:
            ts = TimeSeriesData(time=group.year, value=group.value)
            output_features = model.transform(ts)
        except:
            logging.info(f'Calculation TimeSeries features failed for `{country}`. Please remove this country and try again.')
            continue
        output_features['country_code'] = country
        rows.append(output_features)
    df_agg = pd.DataFrame(rows)

    X = StandardScaler().fit_transform(df_agg[TS_FEATURES].values)
    X = X[:, np.where(np.sum(np.isfinite(X), axis=0) == X.shape[0])[0]]
    try:
        if X.shape[1] < 2:
            raise ValueError('Too many statistical indicators were NaN!')
        x_2d = PCA(n_components=2).fit_transform(X=X)
    except Exception as err:
        raise ValueError(f'PCA calculation failed!\n{err}')
    df_agg['pca_component_1'] = x_2d[:,0]
    df_agg['pca_component_2'] = x_2d[:,1]
    return df_agg


def build_app(connection):
    with connection.cursor() as cursor:
        cursor.execute("SELECT country_code, short_name FROM country")
        countries = []
        for code, name in cursor.fetchall():
            countries.append({'label': name, 'value': code})
        cursor.execute("SELECT indicator_code, name FROM indicator")
        indicators = []
        for code, name in cursor.fetchall():
            indicators.append({'label': name, 'value': code})

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    app.title = 'WDI Data Browser'
    tab_1 = html.Div(children=[
        dcc.Graph(id='line-graph',
                  figure={}),
        html.Button('Create Scatter Plot', id='plot-button', n_clicks=0),
        dcc.Loading(
            id="loading-scatter",
            type="default",
            children=html.Div(id='scatter_plot', children=[])
        )])
    tab_2 = html.Div(children=[
        dcc.Dropdown(id='dropdown-indicator-corr',
                     options=indicators,
                     value='KVV.CSV.POP'),
        dcc.Graph(id='corr-graph',
                  figure={}),
        dash_table.DataTable(id='corr-table',
                             columns=[{'name': c, 'id': c} for c in ['Country', 'Correlation Coefficient']],
                             data={},)])
    app.layout = html.Div([
        html.H1(children='WDI Data Browser'),
        dcc.Dropdown(id='dropdown-indicator',
                     options=indicators,
                     value='KVV.CSV.CO2'),
        dcc.Dropdown(id='dropdown-countries',
                     options=countries,
                     value=['DEU', 'USA', 'FRA'],
                     multi=True),
        dcc.RangeSlider(
            id='year-slider',
            min=1950,
            max=2019,
            step=1,
            value=[1950, 2020],
            marks={i: str(i) for i in range(1950, 2025, 5)}
        ),
        html.Div(id='selected-years'),
        dcc.Tabs(id='tabs', value='tab-1', children=[
            dcc.Tab(label='Single Indicator', children=[tab_1]),
            dcc.Tab(label='Correlation', children=[tab_2])])])
    create_callbacks(app, connection)
    
    return app


def create_callbacks(app, connection):

    @app.callback(Output('selected-years', 'children'),
                  Input('year-slider', 'value'))
    def show_selected_years(years):
        start_year, end_year = years
        return f'You have selected the years between {start_year} and {end_year}!'


    @app.callback(Output('line-graph', "figure"),
                  Output('scatter_plot', "children"),
                  Input('plot-button', 'n_clicks'),
                  Input('dropdown-indicator', 'value'),
                  Input('dropdown-countries', 'value'),
                  Input('year-slider', 'value'),
                  State('line-graph', 'figure'))
    def update_main_tab(n_clicks, indicator, countries, years, line_figure):
        with connection.cursor() as db_cursor:
            ctx = dash.callback_context
            triggered = set(c['prop_id'].split('.')[0] for c in ctx.triggered)
            if 'plot-button' in triggered:
                return line_figure, create_scatter_plot(db_cursor, indicator, countries, years)
            else:
                return create_line_plot(db_cursor, indicator, countries, years), []



    @app.callback(Output('corr-graph', "figure"),
                  Output('corr-table', "data"),
                  Input('dropdown-indicator-corr', 'value'),
                  Input('dropdown-indicator', 'value'),
                  Input('dropdown-countries', 'value'),
                  Input('year-slider', 'value'))
    def update_correlation_tab(indicator_y, indicator_x, countries, years):
        with connection.cursor() as db_cursor:
            df = fetch_data_corr(db_cursor, countries, indicator_x, indicator_y, years)
            data = []
            tab_data = []
            if len(countries) > 0:
                color_cycle = px.colors.qualitative.Plotly
                for i, (country, group) in enumerate(df.groupby("country_code")):
                    dfs = {n: g for n, g in group.groupby("indicator_code")}
                    df_c = dfs[indicator_x].merge(dfs[indicator_y], on='year')
                    data.append(go.Scatter(x=df_c['value_x'],
                                           y=df_c['value_y'],
                                           mode="markers+lines",
                                           hovertext=df_c["year"],
                                           hovertemplate = '%{hovertext}<extra></extra>',
                                           marker=dict(color=color_cycle[i]),
                                           line=dict(color=color_cycle[i]),
                                           name=country))
                    tab_data.append({'Country': country, 'Correlation Coefficient': f"{np.corrcoef(x=df_c['value_x'], y=df_c['value_y'])[0, 1]:.2f}"})

            db_cursor.execute(f"SELECT name FROM indicator WHERE indicator_code=%s;", (indicator_x, ))
            title_x = db_cursor.fetchone()[0]
            db_cursor.execute(f"SELECT name FROM indicator WHERE indicator_code=%s;", (indicator_y, ))
            title_y = db_cursor.fetchone()[0]
            data_elements =  {'data': data,
                              'layout': go.Layout(title = f'{title_x} vs. {title_y}',
                                                  xaxis = {'title': indicator_x},
                                                  yaxis = {'title': indicator_y},
                                                  hovermode='closest')}
            return data_elements, tab_data




    def create_line_plot(db_cursor, indicator, countries, years):
        df = fetch_data(db_cursor, countries, indicator, years)
        data = []
        if len(countries) > 0:
            color_cycle = px.colors.qualitative.Plotly
            for i, (country, group) in enumerate(df.groupby("country_code")):
                data.append(go.Scatter(x=group['year'],
                                       y=group['value'],
                                       mode="markers+lines",
                                       marker=dict(color=color_cycle[i]),
                                       line=dict(color=color_cycle[i]),
                                       name=country))
        db_cursor.execute(f"SELECT name FROM indicator WHERE indicator_code=%s;", (indicator, ))
        title = db_cursor.fetchone()[0]
        data_elements =  {'data': data,
                          'layout': go.Layout(title = title,
                                              xaxis = {'title': 'Year'},
                                              yaxis = {'title': indicator})}
        return data_elements

    def create_scatter_plot(db_cursor, indicator, countries, years, min_years=15):
        logging.info('Creating scatter plot!')
        if years[1] - years[0] < min_years:
            return ['Please selecte a range of years longer than {min_years}!']
        try:
            df = fetch_data_agg(db_cursor, countries, indicator, years)
        except ValueError as err:
            return ['Calculations of statistical indicators failed. This is probably due to not enough data points. Increase year range or different indicator\nDetails: {err}']
        else:
            fig = go.Figure()
            color_cycle = px.colors.qualitative.Plotly
            df_i = df.loc[df['country_code'].apply(lambda s: s not in countries)]
            fig.add_trace(go.Scatter(x=df_i["pca_component_1"],
                                     y=df_i["pca_component_2"],
                                     mode="markers",
                                     hovertext=df_i["country_code"],
                                     hovertemplate = '%{hovertext}<extra></extra>',
                                     marker=dict(color='#AAAAAA', size=4)))
            for i, c in enumerate(sorted(countries)):
                df_i = df.loc[df['country_code'] == c]
                fig.add_trace(go.Scatter(x=df_i["pca_component_1"],
                                         y=df_i["pca_component_2"],
                                         mode="markers",
                                         hovertext=df_i["country_code"],
                                         hovertemplate = '%{hovertext}<extra></extra>',
                                         marker=dict(color=color_cycle[i], size=10)))
            fig.update_layout(showlegend=False)
            fig.layout.plot_bgcolor = '#fff'
            fig.update_xaxes(zerolinecolor='#666666', zerolinewidth=1, showline=False, linecolor='black', showgrid=False, showticklabels=False)
            fig.update_yaxes(zerolinecolor='#666666', zerolinewidth=1, showline=False, linecolor='black', showgrid=False, showticklabels=False)
            fig = dcc.Graph(id='scatter-graph',
                            figure=fig)
            return fig












