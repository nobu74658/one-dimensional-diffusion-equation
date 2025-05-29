import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def analytical_solution(x, t, n_terms=100):
    """
    1次元拡散方程式の解析解を計算する関数
    
    Parameters:
    -----------
    x : ndarray
        空間座標
    t : float
        時刻
    n_terms : int
        フーリエ級数の項数
    
    Returns:
    --------
    u : ndarray
        解析解
    """
    u = np.zeros_like(x)
    
    # 初期条件: u(x, 0) = 2x (x < 0.5), 2(1-x) (x >= 0.5)
    # フーリエ級数展開を用いて解析解を計算
    for n in range(1, n_terms + 1):
        # フーリエ係数の計算
        bn = 8 / (n * np.pi)**2 * np.sin(n * np.pi / 2)
        
        # 各項の計算と加算
        u += bn * np.sin(n * np.pi * x) * np.exp(-(n * np.pi)**2 * t)
    
    return u

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
    plt.figure(figsize=(12, 8))
    
    # 各時刻の解をプロット
    for t, u in sorted(u_history.items()):
        # 数値解をプロット
        plt.plot(x, u, 'o-', label=f'Numerical Solution (t = {t})', markersize=4)
        
        # 解析解をプロット（t = 0 以外）
        if t > 0:
            # より細かい格子で解析解を計算
            x_fine = np.linspace(0, 1, 100)
            u_analytical = analytical_solution(x_fine, t)
            plt.plot(x_fine, u_analytical, '--', label=f'Analytical Solution (t = {t})')
    
    # グラフの設定
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x, t)', fontsize=12)
    plt.title('Comparison of Numerical and Analytical Solutions for 1D Diffusion Equation', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # グラフの保存
    plt.savefig('diffusion_equation_solution.png', dpi=300, bbox_inches='tight')
    
    # グラフの表示
    plt.show()

def print_numerical_results(x, u_history):
    """
    数値計算結果を表形式で出力する関数
    
    Parameters:
    -----------
    x : ndarray
        空間座標
    u_history : dict
        各時刻における解
    """
    # 結果をDataFrameに変換
    results = pd.DataFrame(index=x)
    
    # 各時刻の解をDataFrameに追加
    for t, u in sorted(u_history.items()):
        results[f'数値解 (t = {t})'] = u
        
        # 解析解も追加（t = 0 以外）
        if t > 0:
            results[f'理論解 (t = {t})'] = analytical_solution(x, t)
    
    # 結果を表示
    print("\n数値計算結果と理論解の比較:")
    print("=" * 100)
    print(results.round(6))
    print("=" * 100)
    
    # CSVファイルに保存
    results.to_csv('diffusion_equation_results.csv')
    print("\n数値結果と理論解をCSVファイルに保存しました: diffusion_equation_results.csv")

def main():
    # パラメータの設定
    dx = 0.1
    dt = 0.002
    
    # 拡散方程式を解く
    x, u_history = solve_diffusion_equation(dx, dt)
    
    # 数値結果と理論解を表示
    print_numerical_results(x, u_history)
    
    # 結果をプロット
    plot_results(x, u_history)

if __name__ == "__main__":
    main()