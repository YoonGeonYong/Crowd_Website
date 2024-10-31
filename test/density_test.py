import cv2
import numpy as np
import matplotlib.pyplot as plt

from ai_module.crowded_model import Crowded
from utils import load_img

model = Crowded('ai_module/trained_A.pth')

# k에 따른 density point
def plot_kde(num):
    img = load_img('A', 'test', num)
    dm = model.density_map(img)
    den = model.density(dm)

    _, axe = plt.subplots(3, 5, constrained_layout=True)

    for i in range(2, 7):
        maxmask = cv2.dilate(dm, np.ones((i, i)), iterations=4)
        medmask = cv2.medianBlur(dm, ksize=3)

        maxima = (dm == maxmask)
        med = (dm >= medmask + 0.025)
        y, x = np.nonzero(maxima & med)

        # plot
        axe[0,i-2].set_title(f'k ={i} (num: {len(x)})')
        axe[0,i-2].set_axis_off()
        axe[0,i-2].imshow(dm)

        axe[1,i-2].set_axis_off()
        axe[1,i-2].imshow(maxmask)

        axe[2,i-2].set_axis_off()
        axe[2,i-2].imshow(dm)
        axe[2,i-2].scatter(x, y, color='r', s=5)
    
    plt.suptitle(f'density: {den}')
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.show()

# img, density map, density point 시각화
def plot_density_point(num):
    img = load_img('A', 'test', num)
    dm = model.density_map(img)
    x, y = model.crowd_point(dm)
    # den = model.density(dm)         # 밀집도가 높을 수록 커널크기에 영향을 많이 받음, 커널 크기에 따른, 인원수 변화율로 밀집도 추정 가능
    # print(f'count: {len(x)}, density: {den}')

    # 시각화
    _, axe = plt.subplots(1, 3, constrained_layout=True)
    axe[0].set_title('origin')
    axe[0].set_axis_off()
    axe[0].imshow(img)

    axe[1].set_title('density map')
    axe[1].set_axis_off()
    axe[1].imshow(dm)

    axe[2].set_title(f'density point (num : {len(y)})')
    axe[2].set_axis_off()
    axe[2].imshow(img)
    axe[2].scatter(x*4, y*4, color='r', s=5)
    plt.show()

    # 이미지 저장
    # plt.imsave('img.jpeg', img)
    # plt.imsave('density_map.jpeg', dm)
    # plt.imshow(img)
    # plt.scatter(x*4, y*4, c='r', vmax=6)
    # plt.axis('off')
    # plt.savefig(fname='density_point.jpg', bbox_inches='tight', pad_inches=0)


''' main '''
# for i in range(1, 182): # 182 316
#     plot_density_point(i)

im = load_img('A', 'test', 20)
dm = model.density_map(im)

w, h = dm.shape
x = np.arange(0, h, 1)
y = np.arange(0, w, 1)
X, Y = np.meshgrid(x, y)
Z = dm[Y, X]
# print(Z)

ax = plt.figure().add_subplot(projection='3d')
# ax.plot_wireframe(X, Y, Z)
ax.plot_surface(X, Y, Z, cmap='jet')
# plt.imshow(dm)
plt.show()




# '''내가 KDE 알고리즘을 고안했지만, 기존의 알고리즘이 훨씬 좋다..'''
# import time

# s1 = time.time()
# im = load_img('A','test',2)
# dm = model.density_map(im)
# den = model.density(dm)
# print(den)
# e1 = time.time()
# print('time1', e1-s1)


# from sklearn.neighbors import KernelDensity
# import numpy as np
# denp = model.density_point(dm)
# print(denp)

# s2 = time.time()
# kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(denp)
# log_density = kde.score_samples(denp)
# density = np.exp(log_density)
# print(density)
# e2 = time.time()
# print('time2', e2-s2)
