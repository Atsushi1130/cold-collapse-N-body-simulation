import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

potential = float(input("ポテンシャルソフトニングパラメータ："))
delta_t = float(input("時間積分のタイムステップ:"))
t_end = float(input("シミュレーションの終了時刻："))
t_out = float(input("データ解析/出力の時間間隔："))
N = int(input("粒子数："))
r_v = float(input("ビリアル比："))

#単位系
G = 1
M = 1
R = 1

#1粒子当たりの質量
m = M/N

# 一様球の作成
def init_xyz(N):
    rng = np.random.RandomState(123)
    i = 0
    X = []
    Y = []
    Z = []
    xyz = []
    while i <= N:
        # x = numpy.random.randn()
        # y = numpy.random.randn()
        # z = numpy.random.randn()
        theta = rng.uniform(-1, 1)
        phi = rng.uniform(0,2*np.pi)
        r = rng.uniform(0, 1)
        x = r ** (1 / 3) * (1 - theta ** 2) * np.cos(phi)
        y = r ** (1 / 3) * (1 - theta ** 2) * np.sin(phi)
        z = r ** (1 / 3) * theta
        coordinate = np.array([x,y,z])
        X.append(x)
        Y.append(y)
        Z.append(z)
        xyz.append(coordinate)
        i += 1
    return X,Y,Z,xyz
# print(init_xyz(N))
X,Y,Z,xyz = init_xyz(N)

#正規分布に従う速度分布の生成(平均0、分散1)
def speed_distribution(N):
    v_0 = []
    v_x = np.random.normal(loc=0,scale=1,size=N)
    v_y = np.random.normal(loc=0,scale=1,size=N)
    v_z = np.random.normal(loc=0,scale=1,size=N)
    return v_x,v_y,v_z
v_x,v_y,v_z = speed_distribution(N)
# print(v_x)

#速度分布のヒストグラム(ガウス分布となっているかの確認)
# plt.hist(v_x,bins=100)
# plt.xlim(-10,10)
# plt.show()

#初期速度分散
def init_v_dispersion(N):
    W = 0
    for i in range(1,N):
        for j in range(i+1,N+1):
            r_i = (xyz[i][0]**2 + xyz[i][1]**2 + xyz[i][2]**2)**(1/2)
            r_j = (xyz[j][0]**2 + xyz[j][1]**2 + xyz[j][2]**2)**(1/2)
            W -= (m**2)/((r_i*r_j + potential**2)**(1/2))
        sigma = ((2*r_v*abs(W))/(3*M))**(1/2)
    return sigma

#3次元プロット(3次元的視点から正しく分布しているかの確認)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.set_title("3d plot",size=20)
# ax.set_xlabel("x",size=14)
# ax.set_ylabel("y",size=14)
# ax.set_zlabel("z",size=14)
#
# ax.set_xticks([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
# ax.set_yticks([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
# # for i in range(len(xyz)):
# #     x = xyz[i][0]
# #     y = xyz[i][1]
# #     z = xyz[i][2]
# ax.plot(X,Y,Z,marker="o",linestyle="none",color="red")
# plt.show()

#二次元プロット（粒子分布が一様等方に得られているかの確認）
# xAxisList = []
# nCountList = []
# loopList = [i/100 for i in range(0,100,1)]
# for i in loopList:
#     n_count = 0
#     r_n = i
#     for j in range(len(xyz)):
#         r = (X[j]**2+Y[j]**2+Z[j]**2)**(1/2)
#         if r_n >= r:
#             n_count += 1
#     xAxisList.append(r_n)
#     nCountList.append(n_count)
# print(nCountList)
# plt.plot(xAxisList,nCountList)
# plt.show()

#相互重力計算
def mutual_gravity(N,m,X,Y,Z):
    aList = []
    for i in range(N):
        aList.append(np.array([0,0,0]))
    for i in range(N-1):
        for j in range(i+1,N):
            r_i = np.array([X[i],Y[i],Z[i]])
            r_j = np.array([X[j],Y[j],Z[j]])
            vec_r = r_j - r_i
            abs_r = abs((vec_r[0]**2 + vec_r[1]**2 + vec_r[2]**2)**(1/2))
            for k in range(3):
                aList[i][k] += (m*vec_r[k])/(abs_r**3)
                aList[j][k] += -(m*vec_r[k])/(abs_r**3)
    return aList
aList = mutual_gravity(N,m,X,Y,Z)

#リープフロッグ法
def leapFrog(v_x,v_y,v_z,aList,delta_t,xyz):
    x_1 = []
    x_1_x = []
    x_1_y = []
    x_1_z = []
    for n in range(N):
        v_0 = [v_x[n],v_y[n],v_z[n]]
        v_12 = v_0 + aList[n]*delta_t/2
        # x_1は3次元座標を表す
        x_1.append(xyz[n] + v_12*delta_t)
    return x_1
# print(xyz)
# print("-------------")
# print(leapFrog(v_x,v_y,v_z,aList,delta_t,xyz))