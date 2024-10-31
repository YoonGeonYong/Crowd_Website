import cv2
from flask_socketio import emit
from influxdb_client_3 import Point

from utils import decode_img, encode_img, normalize_img


# socketio.on('image')
def process_img(data, model, db):
    id = data['id']
    lat = data['lat']
    lon = data['lon']

    # 모델 처리
    img = decode_img(data['image'], type='base64')

    dm = model.density_map(img)
    X, Y = model.crowd_point(dm)
    count = X.shape[0]
    den = model.crowd_density(dm)

    # 조정
    img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4), interpolation=cv2.INTER_LINEAR) # 4배 작게 (dm과 크게 맞추기)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # 빨간마커 표시하기 위함
    for x, y in zip(X, Y):
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    dm = normalize_img(dm) # for highlighting

    _dm = encode_img(dm, type='base64')
    _img = encode_img(img, type='base64')

    # 웹 소켓 send
    emit('response', {
        'densityMap' : _dm,
        'crowdPoint': _img,
        'crowdCount': count,
        'crowdDensity': den
    })

    # db 저장
    point = Point('crowd_density') \
                .tag('id', id) \
                .field('lat', lat) \
                .field('lon', lon) \
                .field('count', count) \
                .field('density', den)
    db.write(record=point)