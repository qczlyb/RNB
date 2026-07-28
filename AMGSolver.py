import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyamg
from pyamg.multilevel import MultilevelSolver
from pyamg.relaxation.smoothing import change_smoothers

def scipy_to_csr(A_scipy):
    """将任何 Scipy 稀疏格式统一转换为 CSR，确保计算效率"""
    if A_scipy is None: return None
    return A_scipy.tocsr()

def gmres_custom_engine1(A, b, M_op, max_iter=2000, tol=1e-7, restart=20):
    """
    通用预条件 GMRES 迭代引擎。
    M_op: 必须是具有 matvec 方法的对象（如 LinearOperator 或 ml.aspreconditioner）
    """
    start_time = time.time()
    n = A.shape[0]
    x = np.zeros((n, 1))
    b = b.reshape(-1, 1)
    
    m = restart
    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))
    b0 = np.zeros(m + 1)
    
    curr_res = 1.0
    iters = 0

    for kk in range(max_iter):
        # 1. 计算初始残差并应用预条件
        r0 = b - A @ x
        # 应用预条件 M^-1 * r
        r0 = M_op.matvec(r0).reshape(-1, 1)
        
        beta = np.linalg.norm(r0)
        if beta < 1e-16: 
            break
        
        b0[0] = beta
        V[:, 0] = (r0 / beta).flatten()

        # 2. Arnoldi 过程构造 Krylov 子空间
        for j in range(m):
            # 预条件矩阵向量乘: M^-1 * A * v
            w = A @ V[:, j]
            w = M_op.matvec(w.reshape(-1, 1)).flatten()

            # Gram-Schmidt 正交化
            for i in range(j + 1):
                h = np.dot(w, V[:, i])
                w = w - h * V[:, i]
                H[i, j] = h

            H[j + 1, j] = np.linalg.norm(w)
            V[:, j + 1] = w / (H[j + 1, j] + 1e-16)

        # 3. 求解最小二乘问题并更新解向量
        y, _, _, _ = np.linalg.lstsq(H[:m, :m], b0[:m], rcond=None)
        x = x + (V[:, :m] @ y).reshape(-1, 1)
        
        # 4. 检查收敛性
        curr_res = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        iters = kk + 1
        if curr_res < tol:
            break

    solve_time = time.time() - start_time
    return x, curr_res, solve_time, iters
# =================================================================
# 核心引擎 1: 自编 GMRES 算法 (Custom Engine)
# =================================================================
def gmres_custom_engine(A, b, M_op, max_iter=2000, tol=1e-7, restart=50):
    """
    通用预条件 GMRES(m) 迭代引擎 (对齐标准库行为)
    
    参数:
    A       : 系统矩阵 (通常为 np.ndarray 或 scipy.sparse 矩阵)
    b       : 右侧向量
    M_op    : 左预条件器对象，必须具有 matvec 方法
    x0      : 初始解向量 (可选)，默认从全零向量开始
    max_iter: 最大重启次数
    tol     : 相对收敛容差
    restart : Krylov 子空间的最大维度 (即 GMRES(m) 中的 m)
    
    返回:
    x         : 求解得到的向量
    curr_res  : 最终的相对残差范数 ||b - Ax|| / ||b||
    solve_time: 求解耗时 (秒)
    total_iters: 内部矩阵向量乘法的总调用次数
    """
    start_time = time.time()
    n = A.shape[0]
    
    # 1. 初始化解向量
    
    x = np.zeros((n, 1))
    
        
    b = b.reshape(-1, 1)
    norm_b = np.linalg.norm(b)
    
    # 防止除以零
    if norm_b == 0.0:
        return x, 0.0, time.time() - start_time, 0
        
    m = restart
    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))
    
    total_iters = 0

    for kk in range(max_iter):
        # 2. 计算初始残差并应用左预条件
        r0 = b - A @ x
        r0 = M_op.matvec(r0).reshape(-1, 1)
        
        beta = np.linalg.norm(r0)
        # 计算基于原始系统 b 的真实相对残差
        curr_res = np.linalg.norm(b - A @ x) / norm_b
        
        # 检查是否已经满足收敛要求
        if curr_res < tol or beta < 1e-16: 
            break
            
        b0 = np.zeros(m + 1)
        b0[0] = beta
        V[:, 0] = r0.flatten() / beta
        
        inner_j = m # 用于记录内部循环实际走到的步数

        # 3. Arnoldi 过程构造 Krylov 子空间
        for j in range(m):
            total_iters += 1
            
            # 预条件矩阵向量乘: M^-1 * A * v
            w = A @ V[:, j].reshape(-1, 1)
            w = M_op.matvec(w).flatten()

            # 使用改进的 Gram-Schmidt (MGS) 正交化，提高数值稳定性
            # 
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w = w - H[i, j] * V[:, i]

            H[j + 1, j] = np.linalg.norm(w)
            
            # Happy Breakdown (遇到不变子空间或精确解)
            if H[j + 1, j] < 1e-14:
                inner_j = j + 1
                break
                
            V[:, j + 1] = w / H[j + 1, j]

            # 内部收敛检查：廉价求解当前的子级最小二乘问题
            # 标准 GMRES 用 Givens 旋转更新，这里使用 lstsq 保持逻辑直观
            y_temp, _, _, _ = np.linalg.lstsq(H[:j+2, :j+1], b0[:j+2], rcond=None)
            res_approx = np.linalg.norm(H[:j+2, :j+1] @ y_temp - b0[:j+2])
            
            # 估算相对残差，如果达到要求则提前退出当前的 Arnoldi 循环
            # if res_approx / beta < tol:
            #     inner_j = j + 1
            #     break

        # 4. 求解修正后的最小二乘问题 (GMRES 的关键：超定方程)
        y, _, _, _ = np.linalg.lstsq(H[:inner_j+1, :inner_j], b0[:inner_j+1], rcond=None)
        
        # 更新解向量
        x = x + (V[:, :inner_j] @ y).reshape(-1, 1)
        
        # 5. 重启前的真实残差检查
        curr_res = np.linalg.norm(b - A @ x) / norm_b
        if curr_res < tol:
            break

    solve_time = time.time() - start_time
    return x, curr_res, solve_time, total_iters

# =================================================================
# 核心引擎 2: Scipy 官方 GMRES 封装 (Spla Engine)
# =================================================================
def gmres_spla_engine(A, b, M_op, max_iter=2000, tol=1e-7, restart=50):
    """
    利用 scipy.sparse.linalg.gmres 实现的引擎，输出格式与自编引擎一致。
    """
    start_time = time.time()
    b_flat = b.flatten()
    n = A.shape[0]
    
    # 记录迭代次数
    count_data = {'iters': 0}
    def callback(pr_norm):
        count_data['iters'] += 1

    x, info = spla.gmres(
        A, b_flat, 
        rtol=tol, 
        restart=restart, 
        maxiter=max_iter, 
        M=M_op, 
        callback=callback,
        callback_type='pr_norm'
    )
    
    solve_time = time.time() - start_time
    curr_res = np.linalg.norm(A @ x - b_flat) / np.linalg.norm(b_flat)
    
    return x.reshape(-1, 1), curr_res, solve_time, count_data['iters']


def cg_spla_engine(A, b, M_op, max_iter=2000, tol=1e-7):
    """
    利用 scipy.sparse.linalg.cg 实现的引擎，输出格式与自编引擎一致。
    注意：CG 方法要求矩阵 A 必须是对称正定 (SPD) 的。
    """
    start_time = time.time()
    b_flat = b.flatten()
    n = A.shape[0]
    
    # 记录迭代次数
    count_data = {'iters': 0}
    
    # CG 的 callback 接收的是当前迭代的解向量 xk
    def callback(xk):
        count_data['iters'] += 1

    x, info = spla.cg(
        A, b_flat, 
        rtol=tol,           # 相对容差 (如果在较老的 scipy 版本中报错，可改为 tol=tol)
        maxiter=max_iter, 
        M=M_op, 
        callback=callback
    )
    
    solve_time = time.time() - start_time
    curr_res = np.linalg.norm(A @ x - b_flat) / np.linalg.norm(b_flat)
    
    return x.reshape(-1, 1), curr_res, solve_time, count_data['iters']
# =================================================================
# 顶层接口: AMG-GMRES 求解器
# =================================================================
def amg_gmres(A, b, P=None, max_iter=2000, tol=1e-7, restart=20, mode='custom'):
    """
    封装好的 AMG 预条件求解器。
    mode: 'custom' 使用自编算法, 'spla' 使用 scipy 官方算法
    """
    A = scipy_to_csr(A)
    
    # 1. 构造 AMG 预条件器 (LinearOperator)
    if P is not None:
        # 使用自定义插值算子 P 构造两层 AMG
        P = scipy_to_csr(P)
        R = P.T
        levels = []
        
        lvl0 = MultilevelSolver.Level()
        lvl0.A, lvl0.P, lvl0.R = A, P, R
        levels.append(lvl0)
        
        lvl1 = MultilevelSolver.Level()
        lvl1.A = scipy_to_csr(R @ A @ P)
        levels.append(lvl1)
        
        ml = MultilevelSolver(levels, coarse_solver='splu')
    else:
        # 自动生成平滑聚合 AMG
        ml = pyamg.aggregation.smoothed_aggregation_solver(A, max_coarse=10, coarse_solver='splu')

    # 配置平滑器
    smooth_opts = ('jacobi', {'iterations': 3, 'omega': 0.66})
    change_smoothers(ml, presmoother=smooth_opts, postsmoother=smooth_opts)
    
    # 转换为标准预条件接口
    M_op = ml.aspreconditioner(cycle='V')

    # 2. 调用选定的 GMRES 引擎
    if mode == 'custom':
        return gmres_custom_engine(A, b, M_op, max_iter, tol, restart)
    if mode == 'custom1':
        return gmres_custom_engine1(A, b, M_op, max_iter, tol, restart)
    else:
        return gmres_spla_engine(A, b, M_op, max_iter, tol, restart)