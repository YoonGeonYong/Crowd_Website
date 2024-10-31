'''
    MCNN 모델을 사용해서, crowd counting을 적용하는 모듈
'''
import numpy as np
import cv2
import torch
from sklearn.neighbors import KernelDensity

from .mcnn import MCNN


# device
DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')

# model
class Crowded():
    def __init__(self, path):
        self.model = MCNN()
        self.model.to(DEVICE)
        self.model.load_state_dict(torch.load(path)) # 가중치 로드
        self.model.eval() # 추론 모드
    
    # img -> density map 변환
    '''
        in :    img         (r,c) (int 0~255)
        out :   density map (r,c) (float 0~1)
    '''
    def density_map(self, img):
        img = img_to_tensor(img)
        dm = self.model(img)    # model 처리
        dm = tensor_to_img(dm)
        return dm
    
    # density map -> density point 변환
    '''
        in :    density map     (r,c)               (float 0~1)
        out :   density point   (X=(x,..),Y=(y,..)) (int ~)
    '''
    # def crowd_point(self, dm):  
    #     dm = (dm*255).astype(np.uint8) # 0~255 정규화

    #     mask = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #     erode = cv2.erode(dm, mask)
    #     dilate = cv2.dilate(dm, mask)

    #     flat = dm - erode
    #     peak = dm - dilate

    #     _, flat_t = cv2.threshold(flat, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #     _, in_peak_t = cv2.threshold(peak, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    #     flat_and_in_peak = cv2.bitwise_and(in_peak_t, flat_t)

    #     Y, X = np.nonzero(flat_and_in_peak) # crowd point
    #     return X, Y
    
    def crowd_point(self, dm):
        maxmask = cv2.dilate(dm, np.ones((3,3)), iterations=4)
        medmask = cv2.medianBlur(dm, ksize=3)

        maxima = (dm == maxmask)
        med = (dm >= medmask + 0.025)
        result = maxima & med
        Y, X = np.nonzero(result)
        return X, Y
    

    '''
        in :    density map (r,c)   (float 0~1)
        out :   density     score   (float ~)
    '''
    def crowd_density(self, dm):
        nums = []

        for i in range(2, 7):
            kernel = np.ones((i, i))
            maxmask = cv2.dilate(dm, kernel, iterations=3)
            medmask = cv2.medianBlur(dm, ksize=3)

            maxima = (dm == maxmask)
            med = (dm >= medmask + 0.025)
            result = maxima & med
            Y, X = np.nonzero(result)

            nums.append(len(Y))
        diffs = np.diff(nums)
        den = round(abs(np.mean(diffs)), 2)

        return den
    

''' function '''
# img to tensor : (r, c) -> (1, 1, r, c)
def img_to_tensor(img):
    tensor = torch.from_numpy(img) \
                  .type(torch.FloatTensor) \
                  .to(DEVICE) \
                  .unsqueeze(0).unsqueeze(0) # batch, channel 차원 추가 (1, 1, r, c)
    return tensor

# tensor to img : (1, 1, r, c) -> (r, c)
def tensor_to_img(tensor):
    img = tensor.detach() \
                .cpu() \
                .numpy() \
                .squeeze() # batch, channel 차원 제거 (r, c)
    return img


# test
# if __name__ == '__main__':
#     model = Crowded('ai_module/trained_B.pth')
#     img = cv2.imread('app/upload/input.jpg', 0)
#     dm = model.density_map(img)

#     X, Y = model.crowd_point(dm)

#     cod = []
#     for i in range(X.shape[0]):
#         cod.append((X[i], Y[i]))
#     cod = np.array(cod)
    
#     for i in np.arange(0.1, 30, 0.1):
#         kde = KernelDensity(kernel="gaussian", bandwidth=i).fit(cod) # bandwidth 3.5 적정????
#         log_density = kde.score_samples(cod) # 음수값 포함
#         density = np.exp(log_density)

#         import matplotlib.pyplot as plt

#         plt.plot(density)
#         plt.title(i)
#         plt.ylim([0, 0.001])
#         # plt.hist(density)
#         plt.show()
#         # print(density.max())