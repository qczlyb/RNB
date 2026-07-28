import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline
import torch.nn.functional as F
from scipy.interpolate import RBFInterpolator

class FNN(nn.Module):
    """
    高度可配置的全连接神经网络，支持灵活切换激活函数。
    """
    def __init__(self, layer_sizes, activation='cos'):
        super(FNN, self).__init__()
        
        # 建立层列表
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        # 设置激活函数映射
        self.activation_name = activation.lower()
        activation_map = {
            'cos': torch.cos,
            'sin': torch.sin,
            'relu': F.relu,      # 修改点：指向 functional
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'elu': F.elu         # 修改点：指向 functional
        }
        if self.activation_name not in activation_map:
            raise ValueError(f"不支持的激活函数: {activation}. 请选择 {list(activation_map.keys())}")
            
        self.act_func = activation_map[self.activation_name]

    def forward(self, x):
        # 逐层传播，最后一层不加激活函数
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.act_func(x)
        
        # 最后一层输出 (通常是 64 维基特征)
        x = self.layers[-1](x)
        return x
    

class BasisGenerator:
    """
    二维基函数生成器。
    """
    @staticmethod
    def generate_random_spline_2d(p, total_num=64, num_knots=10, seed=42):
        """
        方案一：径向基样条 (Radial Basis Splines)。
        不再是 Sx * Sy，而是在平面上随机散布控制点，通过径向对称性生成波动。
        """
        N = p.shape[0]
        P = np.zeros((N, total_num))
        rng = np.random.default_rng(seed)
        
        # 定义控制点所在的网格（或者随机散布）
        # 这里在 [-1, 1]^2 区域内生成随机控制点位置
        for k in range(total_num):
            # 随机生成控制点的位置和对应的随机值
            knot_locs = rng.uniform(-1.1, 1.1, (num_knots, 2))
            knot_vals = rng.standard_normal(num_knots)
            
            # 使用径向基函数（Thin Plate Spline 类似物）进行真二维插值
            # 'thin_plate_spline' 具有良好的全局平滑性 (C2 连续)
            rbf = RBFInterpolator(knot_locs, knot_vals, kernel='thin_plate_spline', epsilon=1.0)
            P_col = rbf(p)
            
            # 归一化
            max_val = np.max(np.abs(P_col))
            P[:, k] = P_col / max_val if max_val > 1e-12 else P_col
                
        return P

    @staticmethod
    def generate_random_spline_2d2(p, total_num=64, num_knots=5, seed=42):
        """
        生成随机样条基函数。
        每个二维基函数通过两个随机生成的 1D 三次样条张量积得到。

        参数:
            p: 节点坐标矩阵 (N x 2)，假设分布在 [-1, 1]^2 区域。
            total_num: 需要生成的基函数总数。
            num_knots: 每个一维样条的控制点数量。数量越多，样条的局部波动越复杂。
            seed: 随机种子，确保结果可复现。
        """
        N = p.shape[0]
        P = np.zeros((N, total_num))
        rng = np.random.default_rng(seed)
        
        x_coords = p[:, 0]
        y_coords = p[:, 1]
        
        # 在 [-1, 1] 矩形区域内定义均匀的样条控制点位置
        knots = np.linspace(-1, 1, num_knots)
        
        for k in range(total_num):
            # 1. 为 X 方向生成随机控制值并构造 1D 三次样条
            vals_x = rng.standard_normal(num_knots)
            # 使用 natural bc_type 确保边界处二阶导数为 0，更平滑
            cs_x = CubicSpline(knots, vals_x, bc_type='natural')
            sx = cs_x(x_coords)
            
            # 2. 为 Y 方向生成随机控制值并构造 1D 三次样条
            vals_y = rng.standard_normal(num_knots)
            cs_y = CubicSpline(knots, vals_y, bc_type='natural')
            sy = cs_y(y_coords)
            
            # 3. 通过张量积组合成二维基函数 [P_k = Sx * Sy]
            P_col = sx * sy
            
            # 4. 归一化处理 (L-infinity)，确保数值稳定性
            max_val = np.max(np.abs(P_col))
            if max_val > 1e-12:
                P[:, k] = P_col / max_val
            else:
                P[:, k] = P_col
                
        return P
    @staticmethod
    def generate_random_fourier_2d(p, total_num=64, sigma=1.0, seed=42):
        """
        生成随机傅里叶基函数 (RFF)。
        
        参数:
            p: 节点坐标矩阵 (N x 2)
            total_num: 基函数总数 (对应 MATLAB 中的 D)
            sigma: 高斯核的带宽（控制基函数的震荡频率）
            seed: 随机种子
        """
        N = p.shape[0]
        d = 2  # 空间维度为 2
        rng = np.random.default_rng(seed)
        
        # 1. 采样频率向量 W (对应 MATLAB 中的 w_sample_)
        # 在 RFF 中，W 应该从高斯分布 N(0, sigma^-2 I) 中采样
        # 这里的 W 形状为 (total_num, 2)
        W = rng.standard_normal((total_num, d)) / sigma
        
        # 2. 采样相位 B (对应 MATLAB 中的 B = 2*pi*rand)
        # B 形状为 (total_num,)
        B = rng.uniform(0, 2 * np.pi, total_num)
        
        # 3. 计算特征映射 Z(x) = sqrt(2/D) * cos(W*x + B)
        # 利用广播机制：(N, 2) @ (2, total_num) -> (N, total_num)
        # p: (N, 2), W.T: (2, total_num)
        projection = p @ W.T + B
        
        # 计算基函数矩阵 P
        # 这里的系数 sqrt(2/D) 保证了 P*P^T 逼近高斯核矩阵
        P = np.sqrt(2.0 / total_num) * np.cos(projection)
        
        # 归一化处理（可选，为了保持数值稳定性）
        max_val = np.max(np.abs(P))
        if max_val > 1e-12:
            P = P / max_val
            
        return P
    @staticmethod
    def generate_random_fourier_2d2(p, total_num=6, max_freq=5, seed=42):
        """
        生成随机傅里叶基函数。
        公式: f(x, y) = cos(ax + by + phi) 或 sin(ax + by + phi)
        p: 节点坐标 (N x 2)
        total_num: 基函数总数
        max_freq: 最大频率系数（控制波动密度）
        """
        N = p.shape[0]
        P = np.zeros((N, total_num))
        rng = np.random.default_rng(seed)
        
        x_coords = p[:, 0]
        y_coords = p[:, 1]

        for k in range(total_num):
            # 随机生成 x 和 y 方向的频率系数 (波教)
            # 使用 np.pi 使频率与 [-1, 1] 区域对齐
            a = rng.uniform(-max_freq, max_freq) * np.pi
            b = rng.uniform(-max_freq, max_freq) * np.pi
            # 随机相位
            phi = rng.uniform(0, 2 * np.pi)
            
            # 生成随机平面波
            # 这种方式生成的基函数在 [-1, 1]^2 区域内是全局平滑的
            P[:, k] = np.cos(a * x_coords + b * y_coords + phi)
            
        return P
    
    @staticmethod
    def generate_nn_basis_2d(p, total_num=6, activation='cos', seed=42, layer_sizes=None):
        """
        通过随机初始化的神经网络生成基函数。
        
        Args:
            p: 网格坐标 (N, 2)
            total_num: 需要截取的基函数数量
            activation: 激活函数类型 ('cos', 'sin', 'relu', 'tanh', 'sigmoid', 'elu')
            seed: 随机种子
            layer_sizes: 神经网络各层神经元数量，默认为 [2, 256, 256, 256, 64]
        """
        # 设置默认层结构
        if layer_sizes is None:
            layer_sizes = [2, 256, 256, 256, 64]
            
        torch.manual_seed(seed)
        
        # 实例化网络，使用传入的 layer_sizes 和 activation
        model = FNN(layer_sizes, activation=activation)
        
        inputs = torch.tensor(p, dtype=torch.float32)
        with torch.no_grad():
            # 神经网络前向传播，得到形状为 (N, layer_sizes[-1]) 的输出
            outputs = model(inputs).numpy()
            
        # 截取所需数量并返回。注意：total_num 必须小于或等于 layer_sizes[-1]
        if total_num > layer_sizes[-1]:
            print(f"警告: 请求的 total_num ({total_num}) 超过了网络最后一层神经元数 ({layer_sizes[-1]})")
            return outputs
            
        return outputs[:, :total_num]