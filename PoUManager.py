import numpy as np
from scipy import sparse

class PoUManager:
    """
    负责分区重组 (PoU) 和 QR 处理。
    支持重叠区域划分及隆起函数平滑化。
    """

    @staticmethod
    def bump1d(x, a, b, h):
        """一维隆起函数"""
        x = np.atleast_1d(x)
        y = np.zeros_like(x, dtype=float)
        
        # 区域 1: 平滑上升
        idx1 = (x >= (a - h)) & (x < (a + h))
        y[idx1] = 0.5 * (1 + np.sin(np.pi * (x[idx1] - (a - h)) / (2 * h)))
        
        # 区域 2: 平台期
        idx2 = (x >= (a + h)) & (x <= (b - h))
        y[idx2] = 1.0
        
        # 区域 3: 平滑下降
        idx3 = (x > (b - h)) & (x <= (b + h))
        y[idx3] = 0.5 * (1 + np.sin(np.pi * (b + h - x[idx3]) / (2 * h)))
        
        return y

    @staticmethod
    def bump2d(x, y, ax, bx, ay, by, h):
        """二维隆起函数：通过两个一维隆起函数的张量积生成"""
        return PoUManager.bump1d(x, ax, bx, h) * PoUManager.bump1d(y, ay, by, h)

    def pou_pr(self, P, p, np_val=5, overlap=0.0):
            """
            执行分区 QR 分解生成插值算子。
            
            参数:
                P: 初始基函数矩阵 (N x M)
                p: 节点坐标 (N x 2)
                np_val: 网格划分点数
                overlap: 重叠比例 (float)。例如 0 表示不重叠，0.25 表示重叠区域为子区域宽度的 1/4。
                        默认设为 0.0 (不重叠)。
            """
            n_rows, n_cols = P.shape
            # 预分配结果矩阵 (最大可能宽度)
            PP_dense = np.zeros((n_rows, (np_val - 1)**2 * n_cols))
            
            edges = np.linspace(-1, 1, np_val)
            
            # 1. 计算单个子区域的基础宽度 H_width
            H_width = 2.0 / (np_val - 1)
            
            # 2. h 定义了重叠区域的半宽，由 overlap 比例控制
            h = overlap * H_width
            
            current_col = 0
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
                    if len(idx) == 0:
                        continue
                    
                    # 提取该区域的基函数切片
                    Pij = P[idx, :]
                    
                    if h > 0:
                        # 如果开启重叠，使用 bump2d 进行平滑加权
                        # 计算该子区域内每个点的隆起函数值
                        z = self.bump2d(p[idx, 0], p[idx, 1], xmin, xmax, ymin, ymax, h)
                        # 将 z 应用到基函数上 (按行广播相乘)
                        Pij = Pij * z[:, np.newaxis]
                    
                    Q, _ = np.linalg.qr(Pij)
                    
                    num_q_cols = Q.shape[1]
                    PP_dense[idx, current_col : current_col + num_q_cols] += Q
                    #PP_dense[idx, current_col : current_col + num_q_cols] = Q
                    current_col += num_q_cols
            
            return sparse.csr_matrix(PP_dense[:, :current_col])