from flask_socketio import SocketIO
from .events import process_img


# app에 socketio 추가
def init_socketio(app):
    socketio = SocketIO(app)

    # 이벤트 핸들러 등록
    @socketio.on('image')
    def handle_image(data):
        model = app.config['model']
        db = app.config['db']
        process_img(data, model, db)

    app.config['socketio'] = socketio
    print('load websocket...')

def get_socketio(app):
    return app.config.get('socketio')