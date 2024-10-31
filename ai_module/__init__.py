from .crowded_model import Crowded


# app에 model 추가
def init_model(app):
    model = Crowded(app.config['MODEL_PATH'])
    app.config['model'] = model
    print('load model...')

def get_model(app):
    return app.config.get('model')