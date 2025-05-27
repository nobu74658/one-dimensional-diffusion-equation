import numpy as np
import matplotlib.pyplot as plt

def solve_diffusion_equation(dx=0.1, dt=0.002, t_max=0.4):
    """
    1次元拡散方程式の数値解を陽的差分法で求める関数
    
    Parameters:
    -----------
    dx : float
        空間刻み幅
    dt : float
        時間刻み幅
    t_max : float
        最大計算時間
    
    Returns:
    --------
    x : ndarray
        空間座標
    u_history : dict
        各時刻における解
    """
    # 安定性条件のチェック
    r = dt / (dx**2)
    if r > 0.5:
        print(f"警告: 安定性条件 dt/(dx^2) <= 0.5 を満たしていません。現在の値: {r}")
    
    # 空間グリッドの設定
    x = np.arange(0, 1 + dx, dx)
    nx = len(x)
    
    # 時間ステップ数の計算
    nt = int(t_max / dt) + 1
    
    # 初期条件の設定
    u = np.zeros(nx)
    for i in range(nx):
        if x[i] < 0.5:
            u[i] = 2 * x[i]
        else:
            u[i] = 2 * (1 - x[i])
    
    # 境界条件の適用
    u[0] = 0
    u[-1] = 0
    
    # 結果を保存する辞書
    u_history = {0: u.copy()}
    
    # 出力する時刻
    output_times = [0.05, 0.1, 0.2, 0.4]
    
    # 時間発展
    t = 0
    for n in range(1, nt):
        t = n * dt
        
        # 新しい時間ステップの解を計算
        u_new = np.zeros(nx)
        for i in range(1, nx-1):
            u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])
        
        # 境界条件の適用
        u_new[0] = 0
        u_new[-1] = 0
        
        # 解の更新
        u = u_new.copy()
        
        # 指定された時刻の解を保存
        for output_time in output_times:
            if abs(t - output_time) < dt/2:
                u_history[output_time] = u.copy()
    
    return x, u_history

def plot_results(x, u_history):
    """
    計算結果をプロットする関数
    
    Parameters:
    -----------
    x : ndarray
        空間座標
    u_history : dict
        各時刻における解
    """
    plt.figure(figsize=(10, 6))
    
    # 各時刻の解をプロット
    for t, u in sorted(u_history.items()):
        plt.plot(x, u, label=f't = {t}')
    
    # グラフの設定
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title('1次元拡散方程式の数値解')
    plt.legend()
    plt.grid(True)
    
    # グラフの保存
    plt.savefig('diffusion_equation_solution.png', dpi=300, bbox_inches='tight')
    
    # グラフの表示
    plt.show()

def main():
    # パラメータの設定
    dx = 0.1
    dt = 0.002
    
    # 拡散方程式を解く
    x, u_history = solve_diffusion_equation(dx, dt)
    
    # 結果をプロット
    plot_results(x, u_history)

if __name__ == "__main__":
    main()