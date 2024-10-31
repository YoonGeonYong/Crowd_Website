from influxdb_client_3 import InfluxDBClient3


# app에 db 추가
def init_db(app):
    host = app.config['INFLUXDB_HOST']
    org = app.config['INFLUXDB_ORG']
    database = app.config['INFLUXDB_DATABASE']
    token = app.config['INFLUXDB_TOKEN']

    app.config['db'] = InfluxDBClient3(host=host, token=token, org=org, database=database)
    print('load database connection...')

def get_db(app):
    return app.config['db']