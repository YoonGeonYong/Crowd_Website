from app import create_app
import db_module, ai_module, socket_module


if __name__ == '__main__':
    app = create_app()

    # config 등록
    app.config.from_pyfile('../config.py')  # 앱 기본경로를 기준으로 읽음 (/app/../config.py)
    
    # module 등록
    db_module.init_db(app)
    ai_module.init_model(app)
    socket_module.init_socketio(app)

    socketio = socket_module.get_socketio(app)
    socketio.run(app, host="0.0.0.0", port=8080, ssl_context=("app/ssl/cert.pem", "app/ssl/key.pem"))  # https
    # socketio.run(app, host='0.0.0.0', port=8080) # http