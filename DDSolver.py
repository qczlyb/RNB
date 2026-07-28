import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
# 直接从 AMGSolver 中调用 GMRES 算法引擎
from AMGSolver import gmres_custom_engine, gmres_spla_engine, cg_spla_engine

class DDSolver:
    """
    两层加性施瓦茨 (Two-level Additive Schwarz) 区域分解求解器。
    """
    def __init__(self, A, P, p, np_val=5, overlap=0.0):
        """
        参数:
            A: 系统矩阵
            P: 粗网格插值/投影矩阵
            p: 节点坐标 (N x 2)
            np_val: 网格划分点数
            overlap: 重叠比例 (float)。例如 0 表示不重叠，0.25 表示重叠区域为子区域宽度的 1/4。
        """
        self.A = A.tocsr()
        self.P = P.tocsr()
        self.R = self.P.T
        self.n = self.A.shape[0]
        
        # 1. 构造粗网格算子 (Galerkin 投影: R * A * P)
        self.Ac = self.R @ self.A @ self.P
        self.Ac = self.Ac + 1e-12 * sp.eye(self.Ac.shape[0], format='csc')
        self.Ac_inv = spla.factorized(self.Ac)

        # 2. 构造子区域局部求解器
        self.subdomain_solvers = []
        self.subdomain_indices = []
        
        edges = np.linspace(-1, 1, np_val)
        
        # --- 核心修改：与 pou_pr 保持一致的重叠计算逻辑 ---
        # 计算单个子区域的基础宽度 H_width
        H_width = 2.0 / (np_val - 1)
        # h 定义了重叠区域的半宽
        h = overlap * H_width
        
        for i in range(np_val - 1):
            for j in range(np_val - 1):
                xmin, xmax = edges[i], edges[i+1]
                ymin, ymax = edges[j], edges[j+1]
                
                if h > 0:
                    # 重叠划分逻辑：包含边界外 h 范围的点
                    mask = (p[:, 0] >= xmin - h) & (p[:, 0] <= xmax + h) & \
                           (p[:, 1] >= ymin - h) & (p[:, 1] <= ymax + h)
                else:
                    # 硬划分逻辑 (不重叠 h=0)
                    mask = (p[:, 0] >= xmin) & (p[:, 0] < xmax) & \
                           (p[:, 1] >= ymin) & (p[:, 1] < ymax)
                
                idx = np.where(mask)[0]
                if len(idx) > 0:
                    try:
                        # 提取局部子矩阵并进行 LU 分解
                        solver = spla.factorized(self.A[idx, :][:, idx].tocsc())
                        self.subdomain_solvers.append(solver)
                        self.subdomain_indices.append(idx)
                    except: 
                        continue

    def apply_preconditioner(self, r):
        """预条件步骤: M^-1 * r"""
        r_flat = r.flatten()
        delta_x = np.zeros(self.n)
        # 第一层：局部求解
        for solver, idx in zip(self.subdomain_solvers, self.subdomain_indices):
            delta_x[idx] += solver(r_flat[idx])
        # 第二层：粗空间校正
        delta_x += self.P @ self.Ac_inv(self.R @ r_flat)
        return delta_x

    def solve(self, b, tol=1e-7, max_iter=2000, restart=30, mode='custom'):
        """
        调用 AMGSolver.py 中的 GMRES 算法进行求解。
        mode: 'custom' (使用自编逻辑) 或 'spla' (使用 scipy 封装)
        """
        # 将自身的 apply_preconditioner 包装为标准 LinearOperator
        M_op = spla.LinearOperator((self.n, self.n), matvec=self.apply_preconditioner)

        if mode == 'custom':
            # 直接从 AMGSolver.py 调用自编逻辑
            return gmres_custom_engine(self.A, b, M_op, max_iter, tol, restart)
        if mode == 'cg':
            # 直接从 AMGSolver.py 调用自编逻辑
            return cg_spla_engine(self.A, b, M_op, max_iter=max_iter, tol=tol)
        else:
            # 直接从 AMGSolver.py 调用库函数封装逻辑
            return gmres_spla_engine(self.A, b, M_op, max_iter, tol, restart)