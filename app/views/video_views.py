from flask import Blueprint, render_template, current_app, jsonify
import pandas as pd

from utils import utc_to_kst
 

bp = Blueprint('video', __name__, url_prefix='/video')

@bp.route('/stream')
def show_stream():
    return render_template('video/stream.html')

@bp.route('/list')
def show_list():
    db = current_app.config['db']
    
    # query = """SELECT DISTINCT id
    #             FROM "crowd_density"
    #             WHERE time >= now() - interval '1 hour'"""
    
    query = """SELECT id, lat, lon
                FROM "crowd_density"
                WHERE time = (
                    SELECT MAX(time) 
                    FROM "crowd_density" 
                    WHERE time >= now() - interval '1 hour' 
                    GROUP BY id
                )
                ORDER BY id ASC"""

    table = db.query(query=query, language='sql')
    table = table.to_pydict()   # pyarrow.Table -> dict

    id_list = table['id']
    latitude_list = table['lat']
    longitude_list = table['lon']

    return render_template('video/list.html', id_list=id_list, latitude_list=latitude_list, longitude_list=longitude_list)

# @bp.route('/<string:id>/cam')
# def show_camera(id):
#     return render_template('video/camera.html', id=id)

@bp.route('/<string:id>/stat')
def statistic(id):
    return render_template('video/statistics.html', id=id)

# json for plot
@bp.route('/<string:id>/data')
def get_data(id):
    db = current_app.config['db']
    
    # query = f"""SELECT *                  # test graph
    #             FROM "crowd_density"
    #             WHERE "id" IN ('hall')
    #             AND time >= '2024-10-31T04:41:17'
    #             AND time <= '2024-10-31T05:25:19'
    #             ORDER BY "time" ASC"""
    query = f"""SELECT *
                FROM "crowd_density"
                WHERE "id" IN ('{id}')
                AND time >= now() - interval '1 hour'
                ORDER BY "time" ASC"""

    table = db.query(query=query, language='sql')
    table = table.to_pydict()
    
    # 데이터
    data = {
        'id': table['id'],
        'count': table['count'],
        'density': table['density'],
        'time': [utc_to_kst(t) for t in table['time']]
    }
    return jsonify(data)