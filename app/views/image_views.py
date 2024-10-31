from flask import Blueprint, request, render_template, current_app

from utils import decode_img, encode_img, normalize_img


bp = Blueprint('image', __name__, url_prefix='/image')

@bp.route('/upload', methods=['GET', 'POST'])
def upload_file():
    # POST : 이미지 업로드
    if request.method == 'POST':
        img_byte = request.files['file'].read()
        if len(img_byte) == 0: 
            return render_template('image/image.html') # img가 없을경우 다시 입력화면으로
        
        # model 적용
        img = decode_img(img_byte, type='byte') # bytes -> cv img

        model = current_app.config['model']
        dm = model.density_map(img)
        X, Y = model.crowd_point(dm)
        count = len(X)
        den = model.crowd_density(dm)

        dm = normalize_img(dm) # for highliting

        _img = encode_img(img, type='base64')
        _dm = encode_img(dm, type='base64')

        return render_template('image/image.html', image=_img, density_map=_dm, crowd_count=count, crowd_density=den)

    # GET : 업로드 화면
    return render_template('image/image.html')