import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import timedelta
import base64


''' image '''
# bytes|base64 -> img
def decode_img(img_e, type='byte'):
    if type == 'base64':
        img_e = base64.b64decode(img_e)

    img_d = np.frombuffer(img_e, np.int8)
    img_d = cv2.imdecode(img_d, cv2.IMREAD_GRAYSCALE)
    return img_d

# img -> bytes|base64
def encode_img(img_d, type='byte'):
    _, img_e = cv2.imencode('.jpeg', img_d, [cv2.IMWRITE_JPEG_QUALITY, 50])
    img_e = img_e.tobytes()

    if type == 'base64':
        img_e = base64.b64encode(img_e).decode('utf-8')
    return img_e

# normalize img(densityMap)
def normalize_img(img):
    img = (img - img.min()) / (img.max() - img.min())   # 정규화
    img = (img*255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)      # colormap
    return img


''' time '''
# utc -> kst (+ 9hour)
'''
    in :    utc (e.g., 2024-10-20 15:48:47.667609197)
    out :   kst (e.g., 2024-10-20 12:48:47)
'''
def utc_to_kst(utc):
    kst = utc + timedelta(hours=9)
    kst = kst.strftime('%m/%d %H:%M:%S') # datetime -> str (+ 날짜형식 지정)
    return kst


''' testing '''
# 데이터셋 이미지 로드
def load_img(part, type, num):
    img_path = '/Users/leejuchan/workspace/projects/CrowdCounting/MCNN_svishwa/data/original/shanghaitech/part_' +part+ '_final/' +type+ '_data/images/IMG_' +str(num)+ '.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img

# 그림 출력
def plot_imgs(imgs, names):
    num = len(names)

    # 그림 하나
    if num == 1:
        plt.imshow(imgs[0])
        plt.title(names[0])
        plt.show()

    # 그림 2개 이상
    else:
        _, axe = plt.subplots(1, num, constrained_layout=True) # suplots는 row|col이 2 이상이어야함
        for i in range(num):
            print(i, num)
            axe[i].set_title(names[i])
            axe[i].imshow(imgs[i])
            axe[i].set_axis_off()
        plt.show()