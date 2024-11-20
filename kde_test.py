import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from ai_module import Crowded
from utils import load_img
from kde.kde import kernel_density_estimation


''' hyper parameter '''
kernels = ['gaussian', 'epanechnikov', 'biweight', 'triangular', 'rectangular']
methods = ['product', 'radial']
normals = [True, False]
bands = range(1, 10)

"""
''' 1D '''
# data
S = np.array([-2.6, -2.1, -1.3, -0.4, 1.9, 3.1]).reshape(1, -1)     # (m,n)
S0 = np.zeros_like(S) # for scatter
x = np.arange(-10, 10, 0.1).reshape(1, -1)                          # (m,l)

# kde & plot
# kernels
fig, axes = plt.subplots(1, len(kernels))
fig.suptitle('kernel functions (1D)', fontsize=16)

for id_k, k in enumerate(kernels):
    y = kernel_density_estimation(x, S, bands[0], k, methods[0], normals[0])
    _x = x.flatten()

    axes[id_k].plot(_x, y)
    axes[id_k].scatter(S, S0, s=10, c='red')

    axes[id_k].set_title(f'{k}')
    axes[id_k].set_box_aspect(2)
    axes[id_k].set_ylim([0, 0.3])
    axes[id_k].set_xlim([-5, 5])
plt.show()

# band-widths
fig, axes = plt.subplots(1, len(bands))
fig.suptitle('bandwidths (1D)', fontsize=16)

for id_b, b in enumerate(bands):
    y = kernel_density_estimation(x, S, b, kernels[0], methods[0], normals[0])
    _x = x.flatten()

    axes[id_b].plot(_x, y)
    axes[id_b].scatter(S, S0, s=10, c='red')

    axes[id_b].set_title(f'{float(b)}')
    axes[id_b].set_box_aspect(2)    # y=2x
    axes[id_b].set_ylim([0, 0.3])
    axes[id_b].set_xlim([-5, 5])
plt.show()

# normalization
fig, axes = plt.subplots(1, len(normals))
fig.suptitle('normalization (1D)', fontsize=16)

for id_n, n in enumerate(normals):
    y = kernel_density_estimation(x, S, bands[0], kernels[0], methods[0], n)
    _x = x.flatten()

    axes[id_n].plot(_x, y)
    axes[id_n].scatter(S, S0, s=10, c='red')

    axes[id_n].set_title(f'{n}')
    axes[id_n].set_box_aspect(2)    # y=2x
    axes[id_n].set_ylim([0, 1.2])
    axes[id_n].set_xlim([-5, 5])
plt.show()

# for n in normals:
#     for m in methods:
#         fig, axes = plt.subplots(len(bands), len(kernels))
#         fig.suptitle(f'normal={n}, method={m}') # , fontsize=1, fontsize=166
#         fig.tight_layout(pad=0.5)

#         for id_k, k in enumerate(kernels):
#             for id_b, b in enumerate(bands):
#                 # print(x.shape, S.shape, b, k, m, n)

#                 y = kernel_density_estimation(x, S, b, k, m, n)
#                 _x = x.flatten()

#                 axes[id_b][id_k].plot(_x, y)
#                 axes[id_b][id_k].scatter(S, S0, s=10, c='red')

#                 # axes[id_b][id_k].set_aspect('equal')
#                 axes[id_b][id_k].set_xticks([])
#                 axes[id_b][id_k].set_yticks([])
#                 axes[id_b][id_k].set_ylim([-0.1, 1.5])
#                 axes[id_b][id_k].set_xlim([-4, 4])
#         plt.show()
"""

''' 2D '''
# hyper parameter (model, img)
model = Crowded('ai_module/trained_A.pth')

parts = ['A', 'B']
types = ['train', 'test']
nums = [[300, 182], [400, 316]]

# data
im = load_img(parts[0], types[1], nums[0][1]) # A test 24
dm = model.density_map(im)
dm_ys, dm_xs = dm.shape # (170, 356)

Xs, Ys = model.crowd_point(dm)
S = np.vstack([Xs.ravel(), Ys.ravel()]) # (m,n) (2, 375)

x = np.arange(0, dm_xs, 1)
y = np.arange(0, dm_ys, 1)
X, Y = np.meshgrid(x, y)
A = np.vstack([X.ravel(), Y.ravel()]) # (m,l) (2, 43520)

# kde & plot
# kernels
fig, axes = plt.subplots(1, len(kernels), subplot_kw={"projection" : "3d"})
fig.suptitle('kernel functions (2D)', fontsize=16)

for id_k, k in enumerate(kernels):
    Z = kernel_density_estimation(A, S, bands[7], k, methods[1], normals[1]).reshape(X.shape)

    axes[id_k].plot_wireframe(X, Y, Z, rstride=5, cstride=5)
    axes[id_k].scatter(Xs, Ys, 0, s=1, c='red')
    axes[id_k].contourf(X,Y,Z, zdir='x', offset=0)
    axes[id_k].contourf(X,Y,Z, zdir='y', offset=0)

    axes[id_k].invert_yaxis()
    axes[id_k].set_box_aspect([dm_xs, dm_ys, 300])
    # axes[id_k].set_aspect('equal')
    # axes[id_k].set_xlim([0, dm_xs])
    # axes[id_k].set_ylim([0, dm_ys])
    axes[id_k].set_zlim([0, 0.3])
    axes[id_k].set_title(f'{k}')
plt.show()

# band-widths
fig, axes = plt.subplots(1, len(bands), subplot_kw={"projection" : "3d"})
fig.suptitle('bandwidths (2D)', fontsize=16)

for id_b, b in enumerate(bands):
    Z = kernel_density_estimation(A, S, b, kernels[0], methods[1], normals[0]).reshape(X.shape)

    axes[id_b].plot_wireframe(X, Y, Z, rstride=5, cstride=5)
    axes[id_b].scatter(Xs, Ys, 0, s=1, c='red')
    axes[id_b].contourf(X,Y,Z, zdir='x', offset=0)
    axes[id_b].contourf(X,Y,Z, zdir='y', offset=0)

    axes[id_b].invert_yaxis()
    axes[id_b].set_title(f'{float(b)}')
plt.show()

# normalization
fig, axes = plt.subplots(1, len(normals), subplot_kw={"projection" : "3d"})
fig.suptitle('normalization (2D)', fontsize=16)

for id_n, n in enumerate(normals):
    Z = kernel_density_estimation(A, S, bands[7], kernels[0], methods[1], n).reshape(X.shape)

    axes[id_n].plot_wireframe(X, Y, Z, rstride=5, cstride=5)
    axes[id_n].scatter(Xs, Ys, 0, s=1, c='red')
    axes[id_n].contourf(X,Y,Z, zdir='x', offset=0)
    axes[id_n].contourf(X,Y,Z, zdir='y', offset=0)

    axes[id_n].invert_yaxis()
    axes[id_n].set_title(f'{n}')
plt.show()

# methods
fig, axes = plt.subplots(1, len(methods), subplot_kw={"projection" : "3d"})
fig.suptitle('methods (2D)', fontsize=16)

for id_m, m in enumerate(methods):
    Z = kernel_density_estimation(A, S, bands[7], kernels[0], m, normals[0]).reshape(X.shape)

    axes[id_m].plot_wireframe(X, Y, Z, rstride=5, cstride=5)
    axes[id_m].scatter(Xs, Ys, 0, s=1, c='red')
    axes[id_m].contourf(X,Y,Z, zdir='x', offset=0)
    axes[id_m].contourf(X,Y,Z, zdir='y', offset=0)

    axes[id_m].invert_yaxis()
    axes[id_m].set_title(f'{m}')
plt.show()






                   

# 2D
# Z = kernel_density_estimation(A, S, bands[7], kernels[0], methods[1], normals[0]).reshape(X.shape)

# plt.imshow(Z)
# plt.show()


# # 3d
# dl = []
# for i in range(1, 182): # ts: 182 316 / tr: 300, 400
#     im = load_img('A', 'test', i)

#     dm = model.density_map(im) # (170, 356)
#     h, w = dm.shape

#     Xs, Ys = model.crowd_point(dm)
#     S = np.vstack([Ys.ravel(), Xs.ravel()]) # (m,n) (2, 375)

#     x = np.arange(0, w, 1)
#     y = np.arange(0, h, 1)
#     Y, X = np.meshgrid(y, x)
#     A = np.vstack([Y.ravel(), X.ravel()]) # (m,l) (2, 43520)

#     Z = kernel_density_estimation(A, S, band, kernel, method, normal).reshape(X.shape)
#     den = np.max(Z)
#     # Z *= 1e+4
#     # Z = Z * 1e+1
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(1,3,1, projection='3d')
#     ax.plot_wireframe(X,Y,Z)
#     ax.contourf(X,Y,Z, zdir='y', offset=0)
#     ax.contourf(X,Y,Z, zdir='x', offset=0)
#     ax.invert_yaxis()
#     # ax.set_aspect('equal')
#     # ax.set_box_aspect([1, 1, 2])

#     ax1 = fig.add_subplot(1,3,2)
#     ax1.imshow(im)
#     ax1.scatter(Xs*4, Ys*4, c='red', s=3)
#     ax1.set_title(f'{den}')

#     ax2 = fig.add_subplot(1,3,3)
#     ax2.imshow(dm)
#     plt.show()

#     print(den)
#     dl.append(den)

# print('max: ', np.max(dl)) 
# # A_tr : 2.252356713631091
# # A_ts : 2.074839399002201

# # B_tr : 2.2166354651344213
# # B_ts : 2.0083350151007218


# '''test'''
# # S = np.array([[-2.1, 1.9, 5.1, -3.5, 3.0, 1.2, 4.3, -4.2, 2.9, -1.0],
# #               [-1.3, -0.4, 6.2, -4.5, 2.5, 0.8, 3.7, -3.8, 1.7, 4.4]]) # (m=2,n=10)
# # x = np.arange(-10,10,1)
# # y = np.arange(-10,10,1)
# # X, Y = np.meshgrid(x, y)
# # A = np.vstack([X.ravel(), Y.ravel()])
# # Z = kernel_estimate(A, S, triangular, 1.0).reshape(X.shape)

# # ax = plt.figure().add_subplot(projection='3d')
# # ax.plot_wireframe(X,Y,Z, rstride=1, cstride=1)
# # # ax.contourf(X,Y,Z, zdir='y', offset=8)
# # plt.show()