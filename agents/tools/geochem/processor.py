"""
GeochemProcessor - 地球化学数据清洗与变换

基于Carranza理论实现地球化学数据的预处理，包括：
1. 检测限数据处理
2. 中心对数比变换(CLR)
3. 异常值检测与处理
4. 数据标准化

核心功能：
- impute_censored_data(): 处理低于检测限数据
- transform_clr(): 中心对数比变换
- detect_outliers(): 异常值检测
- standardize_data(): 数据标准化
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Try to import pykrige for kriging interpolation
try:
    from pykrige.ok import OrdinaryKriging
    PYKRIGE_AVAILABLE = True
except ImportError:
    PYKRIGE_AVAILABLE = False
    print("Warning: pykrige not available. Kriging interpolation will use scipy instead.")

# Try to import scipy for interpolation
try:
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class GeochemProcessor:
    """
    地球化学数据处理器
    
    基于Carranza (2009) 第2章方法，实现地球化学数据的
    专业预处理，为后续统计分析提供高质量数据。
    
    参考文献：
    Carranza, E.J.M. (2009). Geochemical Anomaly and Mineral Prospectivity Mapping in GIS.
    """
    
    def __init__(self, detection_limits: Optional[Dict[str, float]] = None,
                 censoring_method: str = 'substitution'):
        """
        初始化数据处理器
        
        Args:
            detection_limits: 检测限字典 {元素: 检测限值}
            censoring_method: 检测限数据处理方法 ('substitution', 'ros', 'mle')
        """
        self.detection_limits = detection_limits or {}
        self.censoring_method = censoring_method
        self.scaler = None
        self.processing_log = []
        
    def impute_censored_data(self, df: pd.DataFrame,
                           elements: Optional[List[str]] = None,
                           method: Optional[Literal['substitution', 'ros', 'mle']] = None) -> pd.DataFrame:
        """
        处理低于检测限数据
        
        根据Carranza (2009) 2.3节方法，处理地球化学数据中
        常见的检测限以下值（左截断数据）。
        
        Args:
            df: 原始地球化学数据
            elements: 待处理元素列表
            method: 处理方法 ('substitution', 'ros', 'mle')
            
        Returns:
            pd.DataFrame: 处理后的数据
            
        Methods:
        - substitution: 替代法（检测限/2或检测限/√2）
        - ros: Regression on Order Statistics
        - mle: Maximum Likelihood Estimation
        
        Example:
            >>> processor = GeochemProcessor(
            ...     detection_limits={'Au': 0.1, 'As': 1.0, 'Sb': 0.5}
            ... )
            >>> processed_data = processor.impute_censored_data(
            ...     raw_data, 
            ...     elements=['Au', 'As', 'Sb'],
            ...     method='substitution'
            ... )
        """
        if method is None:
            method = self.censoring_method
            
        if elements is None:
            elements = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        
        processed_df = df.copy()
        
        for element in elements:
            if element not in self.detection_limits:
                continue
                
            detection_limit = self.detection_limits[element]
            censored_mask = df[element] < detection_limit
            censored_count = censored_mask.sum()
            
            if censored_count == 0:
                continue
            
            # 记录处理信息
            self.processing_log.append({
                'element': element,
                'operation': 'censoring_imputation',
                'method': method,
                'censored_count': censored_count,
                'detection_limit': detection_limit
            })
            
            if method == 'substitution':
                # 替代法：使用检测限/2或检测限/√2
                if censored_count / len(df) > 0.5:  # 超过50%数据被截断
                    substitution_value = detection_limit / np.sqrt(2)
                else:
                    substitution_value = detection_limit / 2
                    
                processed_df.loc[censored_mask, element] = substitution_value
                
            elif method == 'ros':
                # ROS方法（简化实现）
                # 检测到的数据
                detected_data = df[element][~censored_mask].dropna()
                if len(detected_data) > 0:
                    # 对数变换
                    log_detected = np.log10(detected_data)
                    log_dl = np.log10(detection_limit)
                    
                    # 线性回归外推
                    rank = stats.rankdata(detected_data)
                    log_rank = np.log10(rank)
                    
                    if len(detected_data) > 2:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            log_rank, log_detected
                        )
                        
                        # 为截断数据生成估计值
                        censored_ranks = np.arange(1, censored_count + 1)
                        log_censored_estimates = slope * np.log10(censored_ranks) + intercept
                        censored_estimates = 10 ** log_censored_estimates
                        
                        # 确保不超过检测限
                        censored_estimates = np.minimum(censored_estimates, detection_limit * 0.99)
                        processed_df.loc[censored_mask, element] = censored_estimates
                    else:
                        # 回退到替代法
                        processed_df.loc[censored_mask, element] = detection_limit / 2
                        
            elif method == 'mle':
                # 最大似然估计（简化实现）
                detected_data = df[element][~censored_mask].dropna()
                if len(detected_data) > 5:
                    # 假设对数正态分布
                    log_detected = np.log10(detected_data)
                    mu_hat = log_detected.mean()
                    sigma_hat = log_detected.std(ddof=1)
                    
                    # 使用截断正态分布的期望值
                    from scipy.stats import truncnorm
                    a = (np.log10(detection_limit) - mu_hat) / sigma_hat
                    truncated_mean = mu_hat - sigma_hat * (
                        stats.norm.pdf(a) / (1 - stats.norm.cdf(a))
                    )
                    
                    censored_estimates = 10 ** truncated_mean
                    processed_df.loc[censored_mask, element] = censored_estimates
                else:
                    # 回退到替代法
                    processed_df.loc[censored_mask, element] = detection_limit / 2
        
        return processed_df
    
    def transform_clr(self, df: pd.DataFrame,
                     elements: Optional[List[str]] = None,
                     add_small_constant: float = 1e-6) -> pd.DataFrame:
        """
        中心对数比变换 (Centered Log-ratio Transformation)
        
        根据Aitchison (1986) 组成数据分析方法，消除地球化学
        数据的闭合效应，这是Carranza (2009) 推荐的预处理步骤。
        
        Args:
            df: 输入数据
            elements: 变换元素列表
            add_small_constant: 添加的小常数（避免log(0)）
            
        Returns:
            pd.DataFrame: CLR变换后的数据
            
        Mathematical Background:
        CLR变换公式: clr(x) = [ln(x₁/g(x)), ln(x₂/g(x)), ..., ln(x_D/g(x))]
        其中 g(x) = (x₁ × x₂ × ... × x_D)^(1/D) 是几何平均
        
        Example:
            >>> clr_data = processor.transform_clr(
            ...     geochem_df,
            ...     elements=['Au', 'As', 'Sb', 'Cu', 'Pb', 'Zn']
            ... )
            >>> print(f"变换后数据形状: {clr_data.shape}")
        """
        if elements is None:
            elements = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        
        # 提取数据并添加小常数
        data = df[elements].copy()
        data = data + add_small_constant
        
        # 检查负值
        if (data < 0).any().any():
            raise ValueError("CLR变换要求数据必须为正值")
        
        # 计算几何平均
        geometric_mean = np.exp(np.log(data).mean(axis=1))
        
        # CLR变换
        clr_data = np.log(data.div(geometric_mean, axis=0))
        
        # 添加CLR前缀到列名
        clr_columns = [f'CLR_{elem}' for elem in elements]
        clr_df = pd.DataFrame(clr_data, columns=clr_columns, index=df.index)
        
        # 记录处理信息
        self.processing_log.append({
            'operation': 'clr_transformation',
            'elements': elements,
            'shape_before': data.shape,
            'shape_after': clr_df.shape
        })
        
        return clr_df
    
    def detect_outliers(self, df: pd.DataFrame,
                        elements: Optional[List[str]] = None,
                        method: str = 'robust',
                        contamination: float = 0.05) -> Dict[str, Any]:
        """
        异常值检测
        
        使用多种方法检测地球化学数据中的异常值，
        包括统计方法和基于协方差的方法。
        
        Args:
            df: 输入数据
            elements: 检测元素列表
            method: 检测方法 ('zscore', 'iqr', 'robust', 'elliptic')
            contamination: 异常值比例估计
            
        Returns:
            Dict: 包含异常值检测结果和可视化
            
        Example:
            >>> outliers = processor.detect_outliers(
            ...     geochem_df,
            ...     elements=['Au', 'As', 'Sb'],
            ...     method='robust'
            ... )
            >>> print(f"检测到 {len(outliers['outlier_indices'])} 个异常样品")
        """
        if elements is None:
            elements = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        
        data = df[elements].copy()
        outlier_indices = set()
        outlier_scores = {}
        
        if method == 'zscore':
            # Z-score方法
            z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
            outlier_mask = z_scores > 3
            outlier_indices.update(data[outlier_mask.any(axis=1)].index)
            outlier_scores['zscore'] = z_scores.max(axis=1)
            
        elif method == 'iqr':
            # 四分位距方法
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (data < lower_bound) | (data > upper_bound)
            outlier_indices.update(data[outlier_mask.any(axis=1)].index)
            
        elif method == 'robust':
            # 基于鲁棒统计的方法
            median = data.median()
            mad = np.abs(data - median).median()
            modified_z_scores = 0.6745 * (data - median) / mad
            outlier_mask = np.abs(modified_z_scores) > 3.5
            outlier_indices.update(data[outlier_mask.any(axis=1)].index)
            outlier_scores['robust_zscore'] = modified_z_scores.abs().max(axis=1)
            
        elif method == 'elliptic':
            # 椭圆包络方法
            detector = EllipticEnvelope(contamination=contamination, random_state=42)
            outlier_labels = detector.fit_predict(data.fillna(data.median()))
            outlier_mask = outlier_labels == -1
            outlier_indices.update(data[outlier_mask].index)
            outlier_scores['elliptic'] = detector.decision_function(data.fillna(data.median()))
        
        # 可视化异常值
        if len(elements) >= 2:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Outlier Detection Results ({method.upper()} method)', 
                        fontsize=16, fontweight='bold')
            
            # 前两个元素的散点图
            elem1, elem2 = elements[0], elements[1]
            normal_data = data[~data.index.isin(outlier_indices)]
            outlier_data = data[data.index.isin(outlier_indices)]
            
            axes[0, 0].scatter(normal_data[elem1], normal_data[elem2], 
                             c='blue', label='Normal', alpha=0.6)
            axes[0, 0].scatter(outlier_data[elem1], outlier_data[elem2], 
                             c='red', label='Outliers', alpha=0.8)
            axes[0, 0].set_xlabel(elem1)
            axes[0, 0].set_ylabel(elem2)
            axes[0, 0].set_title(f'{elem1} vs {elem2}')
            axes[0, 0].legend()
            
            # 箱线图
            data_melted = data.melt(var_name='Element', value_name='Concentration')
            outlier_indicator = data_melted.index.isin(list(outlier_indices) * len(elements))
            data_melted['Type'] = ['Outlier' if i in outlier_indices else 'Normal' 
                                  for i in data_melted.index // len(elements)]
            
            sns.boxplot(data=data_melted, x='Element', y='Concentration', 
                       hue='Type', ax=axes[0, 1])
            axes[0, 1].set_title('Boxplot by Element')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 异常值分数分布
            if outlier_scores:
                score_name = list(outlier_scores.keys())[0]
                scores = outlier_scores[score_name]
                axes[1, 0].hist(scores, bins=30, alpha=0.7)
                axes[1, 0].axvline(x=np.percentile(scores, 95), color='r', 
                                 linestyle='--', label='95th percentile')
                axes[1, 0].set_xlabel(f'{score_name} Score')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Outlier Score Distribution')
                axes[1, 0].legend()
            
            # 异常值统计
            outlier_counts = data.index.isin(outlier_indices).groupby(data.index).sum()
            axes[1, 1].bar(range(len(outlier_counts)), outlier_counts.values)
            axes[1, 1].set_xlabel('Sample Index')
            axes[1, 1].set_ylabel('Number of Outlier Elements')
            axes[1, 1].set_title('Outlier Count per Sample')
            
            plt.tight_layout()
        else:
            fig = None
        
        return {
            'outlier_indices': list(outlier_indices),
            'outlier_scores': outlier_scores,
            'method': method,
            'contamination': contamination,
            'visualization': fig,
            'summary': {
                'total_samples': len(data),
                'outlier_samples': len(outlier_indices),
                'outlier_percentage': len(outlier_indices) / len(data) * 100
            }
        }
    
    def standardize_data(self, df: pd.DataFrame,
                        elements: Optional[List[str]] = None,
                        method: str = 'standard') -> Tuple[pd.DataFrame, Any]:
        """
        数据标准化
        
        Args:
            df: 输入数据
            elements: 标准化元素列表
            method: 标准化方法 ('standard', 'robust', 'minmax')
            
        Returns:
            Tuple[pd.DataFrame, scaler]: 标准化后的数据和标准化器
            
        Example:
            >>> scaled_data, scaler = processor.standardize_data(
            ...     geochem_df, method='robust'
            ... )
            >>> print(f"标准化后均值: {scaled_data.mean().mean():.6f}")
        """
        if elements is None:
            elements = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        
        data = df[elements].copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # 处理缺失值
        data_filled = data.fillna(data.median())
        
        # 标准化
        scaled_data = scaler.fit_transform(data_filled)
        scaled_df = pd.DataFrame(scaled_data, columns=elements, index=df.index)
        
        self.scaler = scaler
        
        # 记录处理信息
        self.processing_log.append({
            'operation': 'standardization',
            'method': method,
            'elements': elements,
            'scaler_params': scaler.get_params() if hasattr(scaler, 'get_params') else None
        })
        
        return scaled_df, scaler
    
    def get_processing_summary(self) -> pd.DataFrame:
        """
        获取数据处理摘要
        
        Returns:
            pd.DataFrame: 处理步骤摘要
        """
        if not self.processing_log:
            return pd.DataFrame()
        
        return pd.DataFrame(self.processing_log)
    
    def plot_data_distribution(self, df: pd.DataFrame,
                              elements: Optional[List[str]] = None,
                              plot_type: str = 'histogram',
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        绘制数据分布图
        
        Args:
            df: 输入数据
            elements: 绘图元素列表
            plot_type: 图表类型 ('histogram', 'boxplot', 'violin', 'qq')
            figsize: 图形尺寸
            
        Returns:
            plt.Figure: 图形对象
        """
        if elements is None:
            elements = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        
        data = df[elements].copy()
        n_elements = len(elements)
        
        # 计算子图布局
        cols = min(4, n_elements)
        rows = (n_elements + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_elements == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Data Distribution ({plot_type.title()})', 
                    fontsize=16, fontweight='bold')
        
        for i, element in enumerate(elements):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            element_data = data[element].dropna()
            
            if plot_type == 'histogram':
                ax.hist(element_data, bins=30, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                
            elif plot_type == 'boxplot':
                ax.boxplot(element_data)
                ax.set_ylabel('Value')
                
            elif plot_type == 'violin':
                sns.violinplot(y=element_data, ax=ax)
                
            elif plot_type == 'qq':
                stats.probplot(element_data, dist="norm", plot=ax)
                
            ax.set_title(f'{element}')
            
        # 隐藏多余的子图
        for i in range(n_elements, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def get_correlation_matrix(self, df: pd.DataFrame, 
                               elements: Optional[List[str]] = None) -> pd.DataFrame:
        """
        计算元素相关性矩阵
        
        Args:
            df: 输入数据
            elements: 分析元素列表
            
        Returns:
            pd.DataFrame: 相关性矩阵
        """
        if elements is None:
            elements = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        
        return df[elements].corr()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, 
                                 elements: Optional[List[str]] = None,
                                 figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        绘制相关性热力图 (英文标签)
        
        Args:
            df: 输入数据
            elements: 分析元素列表
            figsize: 图形尺寸
            
        Returns:
            plt.Figure: 图形对象
        """
        if elements is None:
            elements = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        
        corr_matrix = self.get_correlation_matrix(df, elements)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                    square=True, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
        ax.set_title('Element Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_pca_loadings(self, df: pd.DataFrame, 
                          elements: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        绘制PCA载荷图 (英文标签)
        
        Args:
            df: 输入数据
            elements: 分析元素列表
            figsize: 图形尺寸
            
        Returns:
            plt.Figure: 图形对象
        """
        if elements is None:
            elements = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        
        # 标准化数据
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[elements])
        
        # PCA分析
        pca = PCA(n_components=2)
        pca.fit(scaled_data)
        
        # 创建载荷图
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制载荷向量
        for i, element in enumerate(elements):
            ax.arrow(0, 0, pca.components_[0, i], pca.components_[1, i],
                    head_width=0.05, head_length=0.05, fc='red', ec='red')
            ax.text(pca.components_[0, i]*1.1, pca.components_[1, i]*1.1, 
                    element, fontsize=12, ha='center', va='center')
        
        # 添加参考圆
        circle = Circle((0, 0), 1, fill=False, color='blue', linestyle='--')
        ax.add_patch(circle)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} Variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} Variance)', fontsize=12)
        ax.set_title('PCA Loading Plot', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_dendrogram(self, df: pd.DataFrame, 
                       elements: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        绘制R型聚类树状图 (英文标签)
        
        Args:
            df: 输入数据
            elements: 分析元素列表
            figsize: 图形尺寸
            
        Returns:
            plt.Figure: 图形对象
        """
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import pdist
        
        if elements is None:
            elements = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        
        # 计算相关性距离
        corr_matrix = df[elements].corr()
        distance_matrix = 1 - np.abs(corr_matrix)
        condensed_distances = pdist(distance_matrix.values)
        
        # 层次聚类
        linkage_matrix = linkage(condensed_distances, method='ward')
        
        fig, ax = plt.subplots(figsize=figsize)
        dendrogram(linkage_matrix, labels=elements, ax=ax, 
                   leaf_rotation=45, leaf_font_size=12)
        ax.set_title('R-mode Cluster Dendrogram', fontsize=16, fontweight='bold')
        ax.set_xlabel('Elements', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        plt.tight_layout()
        
        return fig
    
    def interpolate_kriging(self, df: pd.DataFrame, 
                           target_element: str = 'Au',
                           x_col: str = 'X', y_col: str = 'Y',
                           grid_resolution: float = 0.01,
                           variogram_model: Literal['spherical', 'exponential', 'gaussian'] = 'spherical') -> Dict[str, Any]:
        """
        专业克里金插值 - 包含变异函数分析
        
        Args:
            df: 包含坐标和元素浓度的数据框
            target_element: 目标元素列名
            x_col: X坐标列名 (经度)
            y_col: Y坐标列名 (纬度)
            grid_resolution: 网格分辨率 (度)
            variogram_model: 变异函数模型 ('spherical', 'exponential')
            
        Returns:
            Dict: 包含插值结果的字典
                {
                    'grid_x': 网格X坐标,
                    'grid_y': 网格Y坐标,
                    'grid_z': 插值结果矩阵,
                    'extent': [xmin, xmax, ymin, ymax],
                    'variogram_params': {'nugget': nugget, 'sill': sill, 'range': range_val},
                    'figure': matplotlib Figure对象
                }
        """
        # 检查必要列是否存在
        required_cols = [x_col, y_col, target_element]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 清理数据（移除NaN值）
        clean_data = df[required_cols].dropna()
        if len(clean_data) < 10:
            raise ValueError(f"Insufficient valid data points for kriging: {len(clean_data)} (minimum 10 required)")
        
        x = clean_data[x_col].values
        y = clean_data[y_col].values
        z = clean_data[target_element].values
        
        # 对数变换（地球化学数据通常呈对数正态分布）
        log_z = np.log10(z + 1e-6)  # 避免log(0)
        
        # 创建网格
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        
        # 扩展边界
        x_range = xmax - xmin
        y_range = ymax - ymin
        xmin -= x_range * 0.1
        xmax += x_range * 0.1
        ymin -= y_range * 0.1
        ymax += y_range * 0.1
        
        # 计算网格点数
        nx = int((xmax - xmin) / grid_resolution) + 1
        ny = int((ymax - ymin) / grid_resolution) + 1
        
        # 内存优化：限制网格大小
        max_grid_size = 500  # 最大网格尺寸
        if nx > max_grid_size or ny > max_grid_size:
            # 自动调整分辨率
            scale_factor = max(nx / max_grid_size, ny / max_grid_size)
            new_resolution = grid_resolution * scale_factor
            nx = int((xmax - xmin) / new_resolution) + 1
            ny = int((ymax - ymin) / new_resolution) + 1
            
            print(f"⚠️ Grid too large ({nx}×{ny}), adjusting resolution to {new_resolution:.6f}")
            print(f"   Original resolution: {grid_resolution}")
            print(f"   Adjusted grid size: {nx}×{ny}")
            grid_resolution = new_resolution
        
        # 检查内存需求
        memory_mb = (nx * ny * 8 * 3) / (1024 * 1024)  # 3个float64数组
        if memory_mb > 1000:  # 超过1GB内存
            raise MemoryError(f"Grid too large: {nx}×{ny} requires {memory_mb:.1f}MB memory. " +
                            f"Please increase resolution or reduce data extent.")
        
        grid_x = np.linspace(xmin, xmax, nx)
        grid_y = np.linspace(ymin, ymax, ny)
        
        print(f"🔍 Kriging Analysis for {target_element}:")
        print(f"   - Data points: {len(x)}")
        print(f"   - Grid resolution: {grid_resolution}°")
        print(f"   - Grid size: {nx} x {ny}")
        print(f"   - Estimated memory usage: {memory_mb:.1f}MB")
        
        # 执行克里金插值
        if PYKRIGE_AVAILABLE:
            try:
                # 使用pykrige进行普通克里金插值
                OK = OrdinaryKriging(
                    x, y, log_z,
                    variogram_model=variogram_model,
                    verbose=True,  # 显示变异函数参数
                    enable_plotting=False,
                    coordinates_type='geographic'
                )
                
                grid_z, _ = OK.execute('grid', grid_x, grid_y)
                
                # 反对数变换
                grid_z = 10 ** grid_z - 1e-6
                
                # 获取变异函数参数
                if hasattr(OK, 'variogram_parameters'):
                    variogram_params = OK.variogram_parameters
                else:
                    # 估算变异函数参数
                    variogram_params = self._estimate_variogram_params(x, y, log_z)
                
                print(f"   - Variogram model: {variogram_model}")
                print(f"   - Nugget: {variogram_params.get('nugget', 'N/A')}")
                print(f"   - Sill: {variogram_params.get('sill', 'N/A')}")
                print(f"   - Range: {variogram_params.get('range', 'N/A')}")
                
            except Exception as e:
                print(f"⚠️ PyKrige failed: {e}")
                print("🔄 Using scipy interpolation as fallback...")
                pykrige_failed = True
            else:
                pykrige_failed = False
        
        if not PYKRIGE_AVAILABLE or pykrige_failed:
            # 使用scipy进行插值（回退方案）
            if not SCIPY_AVAILABLE:
                raise ImportError("Neither pykrige nor scipy is available for interpolation")
            
            grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
            points = np.column_stack((x, y))
            grid_z = griddata(points, z, (grid_xx, grid_yy), method='cubic')
            
            variogram_params = {'nugget': 0, 'sill': np.var(z), 'range': x_range/2}
        
        # 处理插值结果中的NaN值
        if np.any(np.isnan(grid_z)):
            from scipy.interpolate import NearestNDInterpolator
            interp = NearestNDInterpolator(np.column_stack((x, y)), z)
            grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
            nan_mask = np.isnan(grid_z)
            grid_z[nan_mask] = interp(grid_xx[nan_mask], grid_yy[nan_mask])
        
        # 创建可视化图表
        fig = self._plot_kriging_result(grid_x, grid_y, grid_z, x, y, z, target_element, 
                                      [xmin, xmax, ymin, ymax], variogram_params)
        
        return {
            'grid_x': grid_x,
            'grid_y': grid_y,
            'grid_z': grid_z,
            'extent': [xmin, xmax, ymin, ymax],
            'variogram_params': variogram_params,
            'figure': fig
        }
    
    def _estimate_variogram_params(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict[str, float]:
        """估算变异函数参数"""
        # 计算经验变异函数
        distances = []
        semivariances = []
        
        n_points = len(x)
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                semivar = 0.5 * (z[i] - z[j])**2
                distances.append(dist)
                semivariances.append(semivar)
        
        distances = np.array(distances)
        semivariances = np.array(semivariances)
        
        # 简单估算
        nugget = np.min(semivariances)
        sill = np.var(z)
        range_val = np.percentile(distances, 75)  # 使用75%分位数作为变程
        
        return {'nugget': nugget, 'sill': sill, 'range': range_val}
    
    def _plot_kriging_result(self, grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray,
                           x: np.ndarray, y: np.ndarray, z: np.ndarray, element: str,
                           extent: List[float], variogram_params: Dict[str, float]) -> plt.Figure:
        """绘制克里金插值结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 热力图
        im = ax1.contourf(grid_x, grid_y, grid_z, levels=15, cmap='YlOrRd')
        ax1.scatter(x, y, c=z, s=30, edgecolors='black', linewidth=0.5, cmap='YlOrRd')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title(f'Kriging Interpolation ({element})')
        plt.colorbar(im, ax=ax1, label=f'{element} Concentration')
        
        # 等值线图
        contour = ax2.contour(grid_x, grid_y, grid_z, levels=10, colors='black', linewidths=0.5)
        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.scatter(x, y, c='red', s=20, alpha=0.7)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title(f'Contour Map ({element})')
        
        # 添加变异函数参数信息
        param_text = f"Nugget: {variogram_params.get('nugget', 'N/A'):.3f}\n"
        param_text += f"Sill: {variogram_params.get('sill', 'N/A'):.3f}\n"
        param_text += f"Range: {variogram_params.get('range', 'N/A'):.3f}"
        ax2.text(0.02, 0.98, param_text, transform=ax2.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def run_fractal_ca_model(self, df: pd.DataFrame,
                           target_element: str = 'Au',
                           x_col: str = 'X', y_col: str = 'Y',
                           grid_resolution: float = 0.01) -> Dict[str, Any]:
        """
        C-A分形分析 - 基于栅格的浓度-面积分形模型
        
        Args:
            df: 包含坐标和元素浓度的数据框
            target_element: 目标元素列名
            x_col: X坐标列名 (经度)
            y_col: Y坐标列名 (纬度)
            grid_resolution: 栅格分辨率 (度)
            
        Returns:
            Dict: 包含C-A分析结果的字典
                {
                    'threshold_value': 异常阈值,
                    'grid_x': 栅格X坐标,
                    'grid_y': 栅格Y坐标,
                    'grid_z': 栅格化浓度值,
                    'log_area': 累计面积对数,
                    'log_concentration': 浓度对数,
                    'breakpoints': 拐点位置,
                    'figure': matplotlib Figure对象
                }
        """
        # 检查必要列是否存在
        required_cols = [x_col, y_col, target_element]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 清理数据
        clean_data = df[required_cols].dropna()
        if len(clean_data) < 10:
            raise ValueError(f"Insufficient valid data points: {len(clean_data)} (minimum 10 required)")
        
        x = clean_data[x_col].values
        y = clean_data[y_col].values
        z = clean_data[target_element].values
        
        print(f"🔍 C-A Fractal Analysis for {target_element}:")
        print(f"   - Data points: {len(x)}")
        print(f"   - Grid resolution: {grid_resolution}°")
        
        # 检查数据范围和分辨率
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        estimated_grid_points = (x_range / grid_resolution) * (y_range / grid_resolution)
        
        if estimated_grid_points > 100000:  # 超过10万个点
            print(f"⚠️ Large dataset detected: ~{estimated_grid_points:.0f} grid points")
            print(f"   - Data extent: {x_range:.3f}° × {y_range:.3f}°")
            print(f"   - Consider increasing resolution to reduce memory usage")
        
        # Step 1: 栅格化 (Rasterization using IDW)
        grid_x, grid_y, grid_z = self._rasterize_data(x, y, z, grid_resolution)
        
        # Step 2: C-A计算
        log_area, log_concentration = self._calculate_ca_relationship(grid_z)
        
        # Step 3: 自动分割 (寻找拐点)
        breakpoints, threshold_value = self._find_ca_breakpoints(log_area, log_concentration, grid_z)
        
        print(f"   - Anomaly threshold: {threshold_value:.3f}")
        print(f"   - Breakpoints found: {len(breakpoints)}")
        
        # Step 4: 绘图
        fig = self._plot_ca_fractal(log_area, log_concentration, breakpoints, target_element)
        
        return {
            'threshold_value': threshold_value,
            'grid_x': grid_x,
            'grid_y': grid_y,
            'grid_z': grid_z,
            'log_area': log_area,
            'log_concentration': log_concentration,
            'breakpoints': breakpoints,
            'figure': fig
        }
    
    def _rasterize_data(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                       resolution: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """使用IDW将离散点数据栅格化 (内存优化版本)"""
        # 创建网格
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        
        # 扩展边界
        x_range = xmax - xmin
        y_range = ymax - ymin
        xmin -= x_range * 0.1
        xmax += x_range * 0.1
        ymin -= y_range * 0.1
        ymax += y_range * 0.1
        
        nx = int((xmax - xmin) / resolution) + 1
        ny = int((ymax - ymin) / resolution) + 1
        
        # 内存优化：限制网格大小
        max_grid_size = 500  # 最大网格尺寸
        if nx > max_grid_size or ny > max_grid_size:
            # 自动调整分辨率
            scale_factor = max(nx / max_grid_size, ny / max_grid_size)
            new_resolution = resolution * scale_factor
            nx = int((xmax - xmin) / new_resolution) + 1
            ny = int((ymax - ymin) / new_resolution) + 1
            
            print(f"⚠️ Grid too large ({nx}×{ny}), adjusting resolution to {new_resolution:.6f}")
            print(f"   Original resolution: {resolution}")
            print(f"   Adjusted grid size: {nx}×{ny}")
        
        # 检查内存需求
        memory_mb = (nx * ny * 8 * 3) / (1024 * 1024)  # 3个float64数组
        if memory_mb > 1000:  # 超过1GB内存
            raise MemoryError(f"Grid too large: {nx}×{ny} requires {memory_mb:.1f}MB memory. " +
                            f"Please increase resolution or reduce data extent.")
        
        grid_x = np.linspace(xmin, xmax, nx)
        grid_y = np.linspace(ymin, ymax, ny)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
        
        # IDW插值 (优化版本)
        grid_z = np.zeros_like(grid_xx)
        
        # 使用向量化操作优化IDW插值
        for i in range(ny):
            for j in range(nx):
                point_x, point_y = grid_xx[i, j], grid_yy[i, j]
                
                # 计算距离 (向量化)
                distances = np.sqrt((x - point_x)**2 + (y - point_y)**2)
                
                # 找到最近的k个点 (优化计算)
                k = min(12, len(x))  # 使用最近的12个点
                nearest_indices = np.argpartition(distances, k)[:k]
                
                # 计算权重
                nearest_distances = distances[nearest_indices]
                nearest_distances[nearest_distances == 0] = 1e-10
                weights = 1.0 / nearest_distances**2
                weights /= weights.sum()
                
                # 加权平均
                grid_z[i, j] = np.sum(weights * z[nearest_indices])
        
        return grid_x, grid_y, grid_z
    
    def _calculate_ca_relationship(self, grid_z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算C-A关系 (Concentration-Area)"""
        # 将2D网格展平为1D数组
        z_flat = grid_z.flatten()
        
        # 移除NaN值
        z_valid = z_flat[~np.isnan(z_flat)]
        
        # 按浓度降序排列
        z_sorted = np.sort(z_valid)[::-1]
        
        # 计算累计面积 (像素数)
        n_pixels = len(z_sorted)
        areas = np.arange(1, n_pixels + 1)
        
        # 取对数 (避免log(0))
        concentration_positive = z_sorted[z_sorted > 0]
        areas_positive = areas[:len(concentration_positive)]
        
        log_concentration = np.log10(concentration_positive)
        log_area = np.log10(areas_positive)
        
        return log_area, log_concentration
    
    def _find_ca_breakpoints(self, log_area: np.ndarray, log_concentration: np.ndarray,
                           grid_z: np.ndarray) -> Tuple[List[int], float]:
        """使用分段线性回归寻找C-A曲线拐点"""
        try:
            # 尝试使用pwlf库进行分段线性回归
            import pwlf
            
            # 创建分段线性回归模型
            pwlf_model = pwlf.PiecewiseLinFit(log_area, log_concentration)
            
            # 尝试1-2个拐点
            min_bic = float('inf')
            best_breakpoints = []
            
            for n_breaks in range(1, 3):
                try:
                    breaks = pwlf_model.fit(n_breaks)
                    bic = pwlf_model.bic
                    
                    if bic < min_bic:
                        min_bic = bic
                        best_breakpoints = breaks
                except:
                    continue
            
            if len(best_breakpoints) > 0:
                # 获取阈值 (使用第一个拐点对应的浓度值)
                breakpoint_idx = np.argmin(np.abs(log_area - best_breakpoints[0]))
                threshold_idx = int(len(grid_z.flatten()) * (1 - breakpoint_idx / len(log_area)))
                threshold_value = np.sort(grid_z.flatten())[::-1][threshold_idx]
                
                return best_breakpoints, threshold_value
        
        except ImportError:
            print("   - pwlf not available, using simple breakpoint detection")
        except Exception as e:
            print(f"   - pwlf failed: {e}, using simple detection")
        
        # 回退到简单的基于残差的拐点检测
        return self._simple_breakpoint_detection(log_area, log_concentration, grid_z)
    
    def _simple_breakpoint_detection(self, log_area: np.ndarray, log_concentration: np.ndarray,
                                  grid_z: np.ndarray) -> Tuple[List[int], float]:
        """简单的拐点检测方法"""
        n_points = len(log_area)
        
        # 计算所有可能的拐点位置
        min_residual = float('inf')
        best_breakpoint = n_points // 3  # 默认位置
        
        # 搜索最佳拐点位置 (前1/3到2/3范围)
        for i in range(n_points // 3, 2 * n_points // 3):
            # 分段线性拟合
            # 第一段
            x1, y1 = log_area[:i+1], log_concentration[:i+1]
            fit1 = np.polyfit(x1, y1, 1)
            pred1 = np.polyval(fit1, x1)
            residual1 = np.sum((y1 - pred1)**2)
            
            # 第二段
            x2, y2 = log_area[i:], log_concentration[i:]
            fit2 = np.polyfit(x2, y2, 1)
            pred2 = np.polyval(fit2, x2)
            residual2 = np.sum((y2 - pred2)**2)
            
            total_residual = residual1 + residual2
            
            if total_residual < min_residual:
                min_residual = total_residual
                best_breakpoint = i
        
        # 计算阈值
        threshold_idx = int(len(grid_z.flatten()) * (1 - best_breakpoint / n_points))
        threshold_value = np.sort(grid_z.flatten())[::-1][threshold_idx]
        
        return [log_area[best_breakpoint]], threshold_value
    
    def _plot_ca_fractal(self, log_area: np.ndarray, log_concentration: np.ndarray,
                        breakpoints: List[float], element: str) -> plt.Figure:
        """绘制C-A分形分析图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制散点
        ax.scatter(log_concentration, log_area, alpha=0.6, s=20, c='blue', label='Data points')
        
        # 分段线性拟合
        if len(breakpoints) > 0:
            # 添加拐点到数据中
            all_x = np.concatenate([[log_concentration[0]], log_concentration, [log_concentration[-1]]])
            all_y = np.concatenate([[log_area[0]], log_area, [log_area[-1]]])
            
            # 为每个线段拟合
            segments_x = []
            segments_y = []
            
            prev_idx = 0
            for bp in breakpoints:
                # 找到最接近拐点的索引
                bp_idx = np.argmin(np.abs(log_area - bp))
                
                # 拟合线段
                segment_x = log_concentration[prev_idx:bp_idx+1]
                segment_y = log_area[prev_idx:bp_idx+1]
                
                if len(segment_x) > 1:
                    fit = np.polyfit(segment_x, segment_y, 1)
                    x_fit = np.linspace(segment_x[0], segment_x[-1], 50)
                    y_fit = np.polyval(fit, x_fit)
                    
                    ax.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.8)
                    segments_x.extend(x_fit)
                    segments_y.extend(y_fit)
                
                prev_idx = bp_idx
            
            # 最后一段
            if prev_idx < len(log_concentration) - 1:
                segment_x = log_concentration[prev_idx:]
                segment_y = log_area[prev_idx:]
                
                if len(segment_x) > 1:
                    fit = np.polyfit(segment_x, segment_y, 1)
                    x_fit = np.linspace(segment_x[0], segment_x[-1], 50)
                    y_fit = np.polyval(fit, x_fit)
                    
                    ax.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.8)
            
            # 标记拐点
            for bp in breakpoints:
                bp_idx = np.argmin(np.abs(log_area - bp))
                ax.plot(log_concentration[bp_idx], log_area[bp_idx], 'ro', 
                       markersize=8, label=f'Breakpoint')
        
        ax.set_xlabel('Log(Concentration)')
        ax.set_ylabel('Log(Area)')
        ax.set_title(f'C-A Fractal Analysis ({element})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig