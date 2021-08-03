import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#　一様等方な球の作成
def init_pos(N):
    i = 0
    position = []
    while i < N:
        x = np.random.uniform(-1,1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(-1, 1)
        r = (x**2 + y**2 + z**2)**(1/2)
        if r <= 1:
            coordinate = [x,y,z]
            position.append(coordinate)
            pos = np.array(position)
            i += 1

        # theta = np.random.uniform(-1, 1)
        # phi = np.random.uniform(0,2*np.pi)
        # r = np.random.uniform(0, 1)
        # x = r ** (1 / 3) * (1 - theta ** 2) * np.cos(phi)
        # y = r ** (1 / 3) * (1 - theta ** 2) * np.sin(phi)
        # z = r ** (1 / 3) * theta
        # coordinate = [x,y,z]
        # position.append(coordinate)
        # pos = np.array(position)
        # i += 1
    return pos

# 正規分布に従う速度分布の生成(平均0、分散sigma)
def init_vel(N,sigma):
    velList = []
    initVelList = []
    v_x = np.random.normal(loc=0,scale=sigma,size=N)
    v_y = np.random.normal(loc=0,scale=sigma,size=N)
    v_z = np.random.normal(loc=0,scale=sigma,size=N)
    for n in range(N):
        velocity = [v_x[n],v_y[n],v_z[n]]
        initVelocity = (v_x[n]**2 + v_y[n]**2 + v_z[n]**2)**(1/2)
        velList.append(velocity)
        initVelList.append(initVelocity)
        vel = np.array(velList)
        initVel = np.array(initVelList)
    return vel,initVel

#相互重力の計算
def getAcc(pos, mass, G, softening):
    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # それぞれの粒子間の距離についての記述(転置することでN個生成可能)
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # 距離の逆三乗の計算
    inv_r3 = (dx ** 2 + dy**2 + dz**2 + softening ** 2)
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)

    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    # ax,ay,azについてまとめる
    a = np.hstack((ax, ay, az))


    return a

#運動エネルギーの計算
def physical_e(mass,vel):
    KE = 0.5 * np.sum(np.sum(mass * vel ** 2))

    return KE

#ポテンシャルの計算
def potential(pos, mass, G,softening):
    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # それぞれの粒子間の距離についての記述(転置することでN個生成可能)
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r for all particle pairwise particle separations
    inv_r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2+softening**2)
    inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]

    # ポテンシャル G(m_i)(m_j)/(|r_j - r_i|)
    PE = G * np.sum(np.sum(np.triu(-(mass * mass.T) * inv_r, 1)))  # np.triu・・・対角成分より下の成分を0とした行列を返す

    return PE

#3次元プロット
def threeDiPlot(N,pos):
    xList = []
    yList = []
    zList = []
    for i in range(N):
        xList.append(pos[i][0])
        yList.append(pos[i][1])
        zList.append(pos[i][2])
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title("3d plot", size=20)
    ax.set_xlabel("x", size=14)
    ax.set_ylabel("y", size=14)
    ax.set_zlabel("z", size=14)
    ax.plot(xList, yList, zList, marker=".", linestyle="none", color="red")
    plt.show()
    return 0

#2次元プロット(xy平面)
def twoDiPlot(N,pos):
    xList = []
    yList = []
    for i in range(N):
        xList.append(pos[i][0])
        yList.append(pos[i][1])
    plt.scatter(xList,yList)
    plt.show()
    return 0

#ビリアル比の時間変化のグラフ
def virial_graph(t_save,virialList,num):
    if num == 0:
        plt.scatter(t_save,virialList,marker=".",color="red",label="virial 0.1")
    elif num == 1:
        plt.scatter(t_save, virialList, marker=".", color="blue",label="virial 0.2")
    elif num == 2:
        plt.scatter(t_save, virialList, marker=".", color="green",label="virial 0.3")
    elif num == 3:
        plt.scatter(t_save, virialList, marker=".", color="black",label="virial 0.4")
    # plt.show()
    return 0

#速度のヒストグラム
def speed_hist(N,vel):
    vList = []
    for i in range(N):
        v = (vel[i][0]**2 + vel[i][1]**2 + vel[i][2]**2)**(1/2)
        vList.append(v)
    plt.hist(vList,bins = 100)
    plt.show()

    return 0

#粒子分布のヒストグラム
def density_hist(N,pos):
    rList = []
    for i in range(N):
        x = pos[i][0]
        y = pos[i][1]
        z = pos[i][2]
        r = (x**2 + y**2 + z**2)**(1/2)
        rList.append(r)
    # plt.hist(rList,bins = 200)
    plt.xlabel("中心からの距離",fontname="MS Gothic")
    plt.ylabel("個数",fontname="MS Gothic")
    plt.xlim(0,1)
    plt.hist(rList, bins=100,range=(0,1))
    plt.show()

def velMax_and_t(tList,maxVelList):
    plt.plot(tList, maxVelList)
    plt.xlabel("時間",fontname="MS Gothic")
    plt.ylabel("最大速度", fontname="MS Gothic")
    plt.show()

def dt_and_t(tList,dtList):
    plt.plot(tList,dtList)
    plt.xlabel("時間", fontname="MS Gothic")
    plt.ylabel("dt")
    plt.show()

def main():
    N = 1024 #粒子数
    t = 0 #開始時間
    tEnd = 10 #終了時間
    softening = 10**(-3) #ソフトニングパラメータ
    G = 1.0 #万有引力定数
    r_v = 0.1 #ビリアル比

    np.random.seed(17) #シード値の決定

    M = 1 #系の質量
    mass = M*np.ones((N,1))/N #全体の質量
    pos = init_pos(N) #3次元座標をN個生成(半径1の球内に収まるよう設定)

    #ポテンシャルの計算
    PE = potential(pos, mass, G,softening)
    # print(PE)
    # return 0

    #初期速度分散
    sigma = (2*r_v*abs(PE)/(3*M))**(1/2)

    # print(sigma)
    #
    # return 0

    vel,initVel = init_vel(N,sigma) #粒子の速度の生成
    maxInitVel = max(initVel)

    # speed_hist(N,vel)
    # return 0

    #運動エネルギーの初期値
    # KE = physical_e(mass,vel)

    #力学的エネルギーの計算
    # totalE = PE + KE

    acc = getAcc(pos, mass, G, softening) #初期位置での相互重力作用の計算

    # 何回転する必要があるかの計算
    # Nt = int(np.ceil(tEnd / dt))

    # #運動エネルギー、ポテンシャルの時間変化についてのリストの初期設定
    # KE_save = np.zeros(Nt + 1)
    # KE_save[0] = KE
    # PE_save = np.zeros(Nt + 1)
    # PE_save[0] = PE
    # #ビリアル比のリスト作成と初期値設定
    # virial = KE/abs(PE)
    # virialList = np.zeros(Nt + 1)
    # virialList[0] = virial
    # #時間変化のリスト
    # t_save = np.zeros(Nt + 1)
    # t_save[0] = 0

    # threeDiPlot(N,pos)
    # return 0

    dt = (softening/maxInitVel)

    maxVelList = []
    tList = []
    dtList = []

    flag1 = True
    flag2 = True
    flag3 = True
    flag4 = True
    flag5 = True

    #シミュレーション　メインループ
    while tEnd > t:
        print(dt)
        print(t)
        print("----------------------------")
        # (1/2) kick
        vel += acc * dt / 2.0
        #ドリフト
        pos += vel * dt
        #　粒子移動後の加速度の取得
        acc = getAcc(pos, mass, G, softening)
        #dt後のvelの計算
        vel += acc*dt/2
        maxVel = max((vel[0]**2 + vel[1]**2 + vel[2]**2)**(1/2))
        maxVelList.append(maxVel)
        tList.append(t)
        dtList.append(dt)
        # print(vel)
        # print("--------------")
        #時間変化の追加処理
        t += dt
        dt = (softening/maxVel)
        # #運動エネルギー、ポテンシャルの変化の記述
        # KE = physical_e(mass,vel)
        # PE = potential(pos, mass, G,softening)
        # E_deltaT = KE + PE
        # #運動エネルギー、ポテンシャルの時間変化についてのリストを更新
        # KE_save[i + 1] = KE
        # PE_save[i + 1] = PE
        # #ビリアル比のリストの更新
        # virial = KE/abs(PE)
        # virialList[i+1] = virial
        # #時間変化についてのリストを更新
        # t_save[i+1] = dt*(i+1)
        # print("success")

        # if flag1 and t>=1.2:
        #     density_hist(N, pos)
        #     threeDiPlot(N, pos)
        #     flag1 = False

        # if flag2 and t>=1.5:
        #     density_hist(N, pos)
        #     threeDiPlot(N, pos)
        #     flag2 = False

        # if flag3 and t>=2:
        #     density_hist(N, pos)
        #     threeDiPlot(N, pos)
        #     flag3 = False

        if flag4 and t>=3:
            density_hist(N, pos)
            threeDiPlot(N, pos)
            flag4 = False

        if flag5 and t>=4:
            dt_and_t(tList, dtList)
            velMax_and_t(tList, maxVelList)
            density_hist(N, pos)
            threeDiPlot(N, pos)
            flag5 = False




    # deltaE = E_deltaT - totalE
    # ans = deltaE/E_deltaT
    # print(ans)

    dt_and_t(tList, dtList)
    velMax_and_t(tList, maxVelList)
    # density_hist(N, pos)
    # threeDiPlot(N, pos)

    # twoDiPlot(N,pos)
    # virial_graph(t_save,virialList,num)
    # speed_hist(N,vel)

    return 0

if __name__ == "__main__":
    main()

    #ビリアル比のグラフを出すときに使用
    # for num in range(4):
    #     main(num)
    # plt.legend()
    # plt.show()