"""
Gold-Seeker: AI Mineral Prediction System
Streamlit Frontend Application

基于Streamlit的地球化学找矿预测交互式界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import json
import io
import base64
from pathlib import Path
import sys
import warnings
import requests
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 直接导入后端处理器，避免导入整个agents包
sys.path.append(str(Path(__file__).parent / 'agents' / 'tools' / 'geochem'))
from processor import GeochemProcessor

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# 设置seaborn中文字体
sns.set_style('whitegrid')
sns.set_palette('husl')

# 设置Plotly中文字体
import plotly.io as pio
pio.templates.default = "plotly_white"
# 设置中文字体
font_config = {
    'family': 'Microsoft YaHei, SimHei, FangSong, SimSun, Arial',
    'size': 12,
    'color': '#333333'
}
pio.templates["custom"] = {
    'layout': {
        'font': font_config,
        'title': {
            'font': {
                'family': '"Microsoft YaHei", "SimHei", "Arial", sans-serif',
                'size': 16
            }
        },
        'xaxis': {
            'title': {
                'font': {
                    'family': '"Microsoft YaHei", "SimHei", "Arial", sans-serif',
                    'size': 14
                }
            },
            'tickfont': {
                'family': '"Microsoft YaHei", "SimHei", "Arial", sans-serif',
                'size': 12
            }
        },
        'yaxis': {
            'title': {
                'font': {
                    'family': '"Microsoft YaHei", "SimHei", "Arial", sans-serif',
                    'size': 14
                }
            },
            'tickfont': {
                'family': '"Microsoft YaHei", "SimHei", "Arial", sans-serif',
                'size': 12
            }
        }
    }
}

# 设置页面配置
st.set_page_config(
    page_title="Gold-Seeker: AI Mineral Prediction System",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
def set_custom_style():
    """设置自定义样式"""
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    .stSidebar {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2c3e50;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #34495e;
        color: white;
    }
    .plot-container {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .chat-message {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .agent-message {
        background-color: rgba(52, 152, 219, 0.2);
        border-left: 4px solid #3498db;
    }
    .user-message {
        background-color: rgba(46, 204, 113, 0.2);
        border-left: 4px solid #2ecc71;
    }
    </style>
    """, unsafe_allow_html=True)

# 初始化session state
def init_session_state():
    """初始化session state"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'selected_elements' not in st.session_state:
        st.session_state.selected_elements = ['Au', 'As', 'Sb', 'Hg']
    if 'target_mineral' not in st.session_state:
        st.session_state.target_mineral = 'Au'
    if 'deepseek_api_key' not in st.session_state:
        st.session_state.deepseek_api_key = 'sk-5bb78328de57481ea1f463325f209b02'

# 生成模拟数据
def generate_mock_data(n_samples=200):
    """生成模拟地球化学数据 - 使用正确的地理坐标"""
    np.random.seed(42)
    
    # 生成合理的地理坐标范围 (经度: 100-101°, 纬度: 30-31°)
    data = {
        'X': np.random.uniform(100.0, 101.0, n_samples),  # 经度范围
        'Y': np.random.uniform(30.0, 31.0, n_samples),    # 纬度范围
        'Au': np.random.lognormal(0, 1, n_samples),
        'As': np.random.lognormal(1, 0.8, n_samples),
        'Sb': np.random.lognormal(0.5, 0.9, n_samples),
        'Hg': np.random.lognormal(-0.5, 1.2, n_samples),
        'Cu': np.random.lognormal(2, 0.7, n_samples),
        'Pb': np.random.lognormal(1.5, 0.6, n_samples),
        'Zn': np.random.lognormal(2.2, 0.5, n_samples),
        'Ag': np.random.lognormal(-0.2, 1.0, n_samples),
    }
    
    # 添加一些低于检测限的值
    detection_limits = {'Au': 0.05, 'As': 0.5, 'Sb': 0.2, 'Hg': 0.01}
    for element, limit in detection_limits.items():
        censored_mask = np.random.random(n_samples) < 0.2
        data[element][censored_mask] = np.random.uniform(0, limit, censored_mask.sum())
    
    # 添加训练点标签
    data['Is_Deposit'] = np.zeros(n_samples, dtype=int)
    deposit_indices = np.random.choice(n_samples, size=20, replace=False)
    for idx in deposit_indices:
        data['Is_Deposit'][idx] = 1
        data['Au'][idx] *= np.random.uniform(5, 20)
        data['As'][idx] *= np.random.uniform(3, 10)
        data['Sb'][idx] *= np.random.uniform(2, 8)
    
    return pd.DataFrame(data)

# 生成相关性热力图 (调用后端)
def create_correlation_heatmap(data, elements):
    """创建相关性热力图"""
    processor = GeochemProcessor()
    return processor.plot_correlation_heatmap(data, elements)

# 生成R型聚类树状图 (调用后端)
def create_dendrogram(data, elements):
    """创建R型聚类树状图"""
    processor = GeochemProcessor()
    return processor.plot_dendrogram(data, elements)

# 生成PCA载荷图 (调用后端)
def create_pca_loadings_plot(data, elements):
    """创建PCA载荷图"""
    processor = GeochemProcessor()
    return processor.plot_pca_loadings(data, elements)

# 生成C-A分形图
def create_ca_fractal_plot(data, element):
    """创建C-A分形图"""
    # 模拟C-A分形分析
    concentrations = np.sort(data[element].values)
    areas = np.arange(1, len(concentrations) + 1)
    
    # 对数变换
    log_conc = np.log10(concentrations[concentrations > 0])
    log_area = np.log10(areas[concentrations > 0])
    
    # 模拟拐点
    threshold_idx = int(len(log_conc) * 0.8)
    threshold = concentrations[threshold_idx]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制散点图
    ax.scatter(log_conc, log_area, alpha=0.6, s=30, c='blue', label='数据点')
    
    # 拟合背景线
    bg_mask = log_conc < np.log10(threshold)
    if bg_mask.sum() > 1:
        bg_fit = np.polyfit(log_conc[bg_mask], log_area[bg_mask], 1)
        bg_line = np.poly1d(bg_fit)
        ax.plot(log_conc[bg_mask], bg_line(log_conc[bg_mask]), 
                'r--', linewidth=2, label='背景拟合')
    
    # 拟合异常线
    anom_mask = log_conc >= np.log10(threshold)
    if anom_mask.sum() > 1:
        anom_fit = np.polyfit(log_conc[anom_mask], log_area[anom_mask], 1)
        anom_line = np.poly1d(anom_fit)
        ax.plot(log_conc[anom_mask], anom_line(log_conc[anom_mask]), 
                'g--', linewidth=2, label='异常拟合')
    
    # 标记拐点
    ax.axvline(x=np.log10(threshold), color='red', linestyle=':', 
               linewidth=2, label=f'阈值: {threshold:.3f}')
    
    ax.set_xlabel('log(浓度)', fontsize=12)
    ax.set_ylabel('log(面积)', fontsize=12)
    ax.set_title(f'{element} C-A分形分析', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, threshold

# 创建交互式地图
def create_comprehensive_analysis_panel(data, element):
    """创建综合分析面板 - 类似demo_comprehensive_analysis.png的专业展示"""
    processor = GeochemProcessor()
    
    # 创建4个子图的综合分析
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 原始数据分布图 (左上)
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(data['X'], data['Y'], c=data[element], 
                         cmap='YlOrRd', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Longitude (°)', fontsize=10)
    ax1.set_ylabel('Latitude (°)', fontsize=10)
    ax1.set_title(f'(a) {element} Distribution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label=f'{element} (ppm)')
    
    # 2. 直方图和统计 (右上)
    ax2 = plt.subplot(2, 3, 2)
    element_data = data[element].dropna()
    ax2.hist(element_data, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel(f'{element} Concentration (ppm)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title(f'(b) {element} Histogram', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_val = element_data.mean()
    std_val = element_data.std()
    median_val = element_data.median()
    ax2.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax2.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    ax2.legend(fontsize=8)
    
    # 3. QQ图 (左中)
    ax3 = plt.subplot(2, 3, 3)
    from scipy import stats
    stats.probplot(element_data, dist="norm", plot=ax3)
    ax3.set_xlabel('Theoretical Quantiles', fontsize=10)
    ax3.set_ylabel('Sample Quantiles', fontsize=10)
    ax3.set_title(f'(c) {element} Q-Q Plot', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 箱线图 (右中)
    ax4 = plt.subplot(2, 3, 4)
    box_plot = ax4.boxplot(element_data, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightblue')
    ax4.set_ylabel(f'{element} Concentration (ppm)', fontsize=10)
    ax4.set_title(f'(d) {element} Box Plot', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. 累积分布函数 (左下)
    ax5 = plt.subplot(2, 3, 5)
    sorted_data = np.sort(element_data)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax5.plot(sorted_data, cumulative, linewidth=2, color='blue')
    ax5.set_xlabel(f'{element} Concentration (ppm)', fontsize=10)
    ax5.set_ylabel('Cumulative Probability', fontsize=10)
    ax5.set_title(f'(e) {element} CDF', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. 统计摘要表 (右下)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # 计算统计指标
    stats_summary = {
        'Count': len(element_data),
        'Mean': f'{mean_val:.3f}',
        'Std Dev': f'{std_val:.3f}',
        'Min': f'{element_data.min():.3f}',
        'Max': f'{element_data.max():.3f}',
        'Median': f'{median_val:.3f}',
        'Skewness': f'{stats.skew(element_data):.3f}',
        'Kurtosis': f'{stats.kurtosis(element_data):.3f}'
    }
    
    # 创建统计表格
    table_data = []
    for key, value in stats_summary.items():
        table_data.append([key, value])
    
    table = ax6.table(cellText=table_data, 
                     colLabels=['Statistic', 'Value'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0.1, 0.1, 0.8, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # 设置表格样式
    for i in range(len(table_data) + 1):
        for j in range(2):
            cell = table[i, j]
            if i == 0:  # 表头
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax6.set_title(f'(f) {element} Statistics Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # 添加总标题
    fig.suptitle(f'Comprehensive Geochemical Analysis - {element}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    st.pyplot(fig)
    plt.close(fig)
    
    # 显示关键统计指标
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("样品数量", len(element_data))
    with col2:
        st.metric("平均值", f"{mean_val:.3f} ppm")
    with col3:
        st.metric("标准差", f"{std_val:.3f} ppm")
    with col4:
        st.metric("变异系数", f"{std_val/mean_val*100:.1f}%")
    
    # 添加AI解释按钮
    if st.button("🤖 AI解释综合分析", key="explain_comprehensive"):
        with st.spinner("🤖 AI正在生成解释..."):
            from scipy import stats
            
            explanation_prompt = f"""
请解释以下地球化学元素综合分析结果：

## 综合统计分析结果
- 分析元素: {element}
- 样本数量: {len(element_data)}
- 平均值: {mean_val:.3f} ppm
- 标准差: {std_val:.3f} ppm
- 中位数: {median_val:.3f} ppm
- 最小值: {element_data.min():.3f} ppm
- 最大值: {element_data.max():.3f} ppm
- 偏度: {stats.skew(element_data):.3f}
- 峰度: {stats.kurtosis(element_data):.3f}
- 变异系数: {std_val/mean_val*100:.1f}%

## 分布特征
- 数据分布: {'正态分布' if abs(stats.skew(element_data)) < 0.5 else '偏态分布'}
- 异常值情况: {'存在异常值' if abs(stats.kurtosis(element_data)) > 3 else '无明显异常值'}
- 数据离散程度: {'高' if std_val/mean_val > 0.5 else '中' if std_val/mean_val > 0.2 else '低'}

请从地质学角度解释：
1. 元素分布的统计特征和地质意义
2. 偏度和峰度对成矿作用的指示
3. 数据离散程度与地质过程的关系
4. 对金矿勘探的指导意义
5. 下一步工作建议

请用简洁明了的语言解释，便于地质勘探人员理解。
"""
            
            api_key = st.session_state.get('deepseek_api_key', '')
            explanation = call_deepseek_api(explanation_prompt, api_key)
            
            if not explanation.startswith("❌"):
                st.markdown("**🧠 AI地质解释：**")
                st.markdown(explanation)
            else:
                st.error(explanation)

def create_professional_kriging_display(data, element, kriging_result, threshold=None):
    """创建专业的克里金插值展示 - 类似demo_kriging_result.png"""
    
    # 创建4个子图的专业展示
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 克里金插值热力图 (左上)
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(kriging_result['grid_z'].T, 
                      extent=kriging_result['extent'],
                      origin='lower', 
                      cmap='YlOrRd',
                      alpha=0.9)
    
    # 添加原始数据点
    ax1.scatter(data['X'], data['Y'], c='black', s=30, alpha=0.7, 
               edgecolors='white', linewidth=0.5, zorder=5)
    
    ax1.set_xlabel('Longitude (°)', fontsize=10)
    ax1.set_ylabel('Latitude (°)', fontsize=10)
    ax1.set_title(f'(a) {element} Kriging Interpolation', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label=f'{element} (ppm)')
    ax1.grid(True, alpha=0.3)
    
    # 2. 等值线图 (右上)
    ax2 = plt.subplot(2, 3, 2)
    contour = ax2.contour(kriging_result['grid_x'], kriging_result['grid_y'], 
                         kriging_result['grid_z'].T, levels=15, colors='black', linewidths=0.8)
    ax2.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
    
    # 添加填充等值线
    contourf = ax2.contourf(kriging_result['grid_x'], kriging_result['grid_y'], 
                           kriging_result['grid_z'].T, levels=15, cmap='YlOrRd', alpha=0.7)
    
    # 添加原始数据点
    ax2.scatter(data['X'], data['Y'], c='blue', s=30, alpha=0.7, 
               edgecolors='white', linewidth=0.5, zorder=5)
    
    ax2.set_xlabel('Longitude (°)', fontsize=10)
    ax2.set_ylabel('Latitude (°)', fontsize=10)
    ax2.set_title(f'(b) {element} Contour Map', fontsize=12, fontweight='bold')
    plt.colorbar(contourf, ax=ax2, label=f'{element} (ppm)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 变差函数图 (左中)
    ax3 = plt.subplot(2, 3, 3)
    
    # 模拟变差函数数据
    if 'variogram_params' in kriging_result:
        variogram_params = kriging_result['variogram_params']
        nugget = variogram_params.get('nugget', 0.1)
        sill = variogram_params.get('sill', 1.0)
        range_val = variogram_params.get('range', 0.5)
    else:
        nugget, sill, range_val = 0.1, 1.0, 0.5
    
    # 生成理论变差函数
    distances = np.linspace(0, range_val * 2, 100)
    theoretical_variogram = sill * (1 - np.exp(-3 * distances / range_val)) + nugget
    
    ax3.plot(distances, theoretical_variogram, 'b-', linewidth=2, label='Theoretical Variogram')
    ax3.axhline(y=sill, color='r', linestyle='--', alpha=0.7, label=f'Sill: {sill:.3f}')
    ax3.axhline(y=nugget, color='g', linestyle='--', alpha=0.7, label=f'Nugget: {nugget:.3f}')
    ax3.axvline(x=range_val, color='orange', linestyle='--', alpha=0.7, label=f'Range: {range_val:.3f}')
    
    ax3.set_xlabel('Distance (°)', fontsize=10)
    ax3.set_ylabel('Variogram', fontsize=10)
    ax3.set_title('(c) Variogram Model', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, range_val * 2)
    
    # 4. 交叉验证图 (右中)
    ax4 = plt.subplot(2, 3, 4)
    
    # 模拟交叉验证数据
    actual_values = data[element].values
    predicted_values = []
    
    # 对每个数据点进行插值预测
    for idx, row in data.iterrows():
        # 简单的最近邻插值作为预测
        distances = np.sqrt((data['X'] - row['X'])**2 + (data['Y'] - row['Y'])**2)
        nearest_idx = distances[distances > 0].idxmin()
        predicted_values.append(data.loc[nearest_idx, element])
    
    predicted_values = np.array(predicted_values)
    
    # 绘制散点图
    ax4.scatter(actual_values, predicted_values, alpha=0.6, s=30)
    
    # 添加1:1线
    min_val = min(actual_values.min(), predicted_values.min())
    max_val = max(actual_values.max(), predicted_values.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')
    
    # 计算R²
    correlation = np.corrcoef(actual_values, predicted_values)[0, 1]
    r_squared = correlation ** 2
    
    ax4.set_xlabel('Observed Values (ppm)', fontsize=10)
    ax4.set_ylabel('Predicted Values (ppm)', fontsize=10)
    ax4.set_title(f'(d) Cross-Validation (R² = {r_squared:.3f})', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. 插值误差分布 (左下)
    ax5 = plt.subplot(2, 3, 5)
    
    # 计算插值误差
    residuals = actual_values - predicted_values
    
    ax5.hist(residuals, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax5.set_xlabel('Residuals (ppm)', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title('(e) Residual Distribution', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. 插值统计表 (右下)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # 计算插值统计
    interpolation_stats = {
        'Grid Size': f"{len(kriging_result['grid_x'])}×{len(kriging_result['grid_y'])}",
        'Data Points': str(len(data)),
        'Min Value': f"{kriging_result['grid_z'].min():.3f}",
        'Max Value': f"{kriging_result['grid_z'].max():.3f}",
        'Mean Value': f"{kriging_result['grid_z'].mean():.3f}",
        'Std Dev': f"{kriging_result['grid_z'].std():.3f}",
        'Nugget': f"{nugget:.3f}",
        'Sill': f"{sill:.3f}",
        'Range': f"{range_val:.3f}",
        'R²': f"{r_squared:.3f}",
        'RMSE': f"{np.sqrt(np.mean(residuals**2)):.3f}"
    }
    
    # 创建统计表格
    table_data = []
    for key, value in interpolation_stats.items():
        table_data.append([key, value])
    
    table = ax6.table(cellText=table_data, 
                     colLabels=['Parameter', 'Value'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0.1, 0.1, 0.8, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    
    # 设置表格样式
    for i in range(len(table_data) + 1):
        for j in range(2):
            cell = table[i, j]
            if i == 0:  # 表头
                cell.set_facecolor('#2196F3')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax6.set_title('(f) Interpolation Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # 添加总标题
    fig.suptitle(f'Professional Kriging Analysis - {element}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    st.pyplot(fig)
    plt.close(fig)

def create_heatmap_display(data, element, threshold=None, kriging_result=None):
    """创建简化的热力图显示"""
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 如果有克里金结果，使用克里金插值
    if kriging_result is not None and 'grid_z' in kriging_result:
        # 使用克里金插值结果
        im = ax.imshow(kriging_result['grid_z'].T, 
                      extent=kriging_result['extent'],
                      origin='lower', 
                      cmap='YlOrRd',
                      alpha=0.8)
        
        # 添加原始数据点
        ax.scatter(data['X'], data['Y'], c='black', s=40, alpha=0.8, 
                  edgecolors='white', linewidth=1, zorder=5)
        
        title = f'{element} Kriging Interpolation Heatmap'
    else:
        # 使用原始数据点创建简单热力图
        from scipy.interpolate import griddata
        
        # 创建网格
        xi = np.linspace(data['X'].min(), data['X'].max(), 100)
        yi = np.linspace(data['Y'].min(), data['Y'].max(), 100)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # 插值
        zi = griddata((data['X'], data['Y']), data[element], 
                     (xi_grid, yi_grid), method='cubic')
        
        # 绘制热力图
        im = ax.contourf(xi_grid, yi_grid, zi, levels=20, cmap='YlOrRd', alpha=0.8)
        
        # 添加原始数据点
        scatter = ax.scatter(data['X'], data['Y'], c=data[element], 
                           cmap='YlOrRd', s=50, alpha=0.9, 
                           edgecolors='black', linewidth=1, zorder=5)
        
        title = f'{element} Distribution Heatmap'
    
    # 设置图表属性
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 添加颜色条
    plt.colorbar(im, ax=ax, label=f'{element} (ppm)')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 如果有阈值，添加异常区域
    if threshold is not None and kriging_result is not None and 'grid_z' in kriging_result:
        # 创建异常区域掩码
        anomaly_mask = kriging_result['grid_z'] > threshold
        
        # 绘制异常区域轮廓
        ax.contour(kriging_result['grid_x'], kriging_result['grid_y'], 
                  anomaly_mask.T, levels=[0.5], colors='red', linewidths=2, 
                  linestyles='--', label=f'Anomaly Threshold: {threshold:.3f}')
        ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # 添加AI解释按钮
    if st.button("🤖 AI解释热力图", key="explain_heatmap"):
        with st.spinner("🤖 AI正在生成解释..."):
            element_data = data[element].dropna()
            
            explanation_prompt = f"""
请解释以下地球化学元素热力图分析结果：

## 热力图分析结果
- 分析元素: {element}
- 样本数量: {len(element_data)}
- 数据范围: X=[{data['X'].min():.3f}, {data['X'].max():.3f}], Y=[{data['Y'].min():.3f}, {data['Y'].max():.3f}]
- 元素浓度范围: [{element_data.min():.3f}, {element_data.max():.3f}] ppm
- 平均值: {element_data.mean():.3f} ppm
"""
            
            if kriging_result is not None and 'grid_z' in kriging_result:
                explanation_prompt += f"""
## 克里金插值信息
- 插值方法: 克里金插值
- 网格分辨率: {len(kriging_result['grid_x'])}x{len(kriging_result['grid_y'])}
- 插值范围: [{kriging_result['extent'][0]:.3f}, {kriging_result['extent'][1]:.3f}]
"""
            
            if threshold is not None:
                anomaly_count = (element_data > threshold).sum()
                explanation_prompt += f"""
## 异常分析
- 异常阈值: {threshold:.3f} ppm
- 异常样品数: {anomaly_count}
- 异常率: {anomaly_count/len(element_data)*100:.1f}%
"""
            
            explanation_prompt += f"""

请从地质学角度解释：
1. 热力图中元素分布的空间特征
2. 高值区和低值区的地质意义
3. 空间分布模式与地质构造的关系
4. 异常区域的成矿潜力评价
5. 对勘探靶区优选的指导意义

请用简洁明了的语言解释，便于地质勘探人员理解。
"""
            
            api_key = st.session_state.get('deepseek_api_key', '')
            explanation = call_deepseek_api(explanation_prompt, api_key)
            
            if not explanation.startswith("❌"):
                st.markdown("**🧠 AI地质解释：**")
                st.markdown(explanation)
            else:
                st.error(explanation)

def create_interactive_map(data, element, threshold=None):
    """创建交互式地图"""
    # 计算中心点
    center_lat = data['Y'].mean()
    center_lon = data['X'].mean()
    
    # 创建地图（无底图）
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=None
    )
    
    # 添加采样点
    for idx, row in data.iterrows():
        color = 'red' if row.get('Is_Deposit', 0) == 1 else 'blue'
        size = 8 if row.get('Is_Deposit', 0) == 1 else 5
        
        folium.CircleMarker(
            location=[row['Y'], row['X']],
            radius=size,
            popup=f"点位 {idx}<br>{element}: {row[element]:.3f}",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    # 如果有阈值，添加异常区域
    if threshold is not None:
        anomaly_points = data[data[element] > threshold]
        if len(anomaly_points) > 0:
            # 创建异常区域的凸包
            from scipy.spatial import ConvexHull
            points = anomaly_points[['Y', 'X']].values
            
            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    
                    # 创建多边形
                    folium.Polygon(
                        locations=[[p[0], p[1]] for p in hull_points],
                        color='red',
                        fill=True,
                        fillColor='red',
                        fillOpacity=0.2,
                        popup='异常区域'
                    ).add_to(m)
                except:
                    pass
    
    return m

# 模拟Agent响应
def mock_agent_response(user_input):
    """模拟Agent响应"""
    responses = {
        "相关性": "我正在分析元素之间的相关性。根据计算结果，Au与As的相关系数为0.75，显示出强烈的正相关性，这是金矿成矿的重要地球化学指标。",
        "异常": "我已经完成了智能异常检测分析，识别出Au的异常阈值为1.2 ppb，共有15个样品被归类为异常，这些区域值得进一步勘探。",
        "聚类": "基于机器学习的聚类分析显示，Au、As、Sb、Hg形成一个紧密的元素组合，这是典型的金矿化元素组合特征。",
        "预测": "通过融合地质知识图谱与大模型的智能预测系统，研究区的成矿潜力评分为0.75，属于高潜力区域。",
        "勘探": "根据智能体分析，建议重点关注构造断裂带附近的异常区域，这些区域具有较好的成矿地质条件。",
        "模型": "本平台采用多模态大模型，融合了地质学、地球化学、遥感等多源数据，提供精准的金矿预测服务。"
    }
    
    for key, response in responses.items():
        if key in user_input:
            return response
    
    return "我是金矿智能预测专家，正在分析您的请求。我可以为您提供成矿预测、异常识别、勘探建议等专业服务。"

# 侧边栏配置
def render_sidebar():
    """渲染侧边栏"""
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>⛏️ Gold-Seeker</h1>
        <p style='font-size: 14px; opacity: 0.8;'>金矿智能预测智能体平台</p>
        <p style='font-size: 12px; opacity: 0.6;'>融合领域知识与大模型技术</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 数据上传
    st.sidebar.markdown("### 📁 数据上传")
    uploaded_file = st.sidebar.file_uploader(
        "选择CSV或GeoJSON文件",
        type=['csv', 'geojson'],
        help="上传地球化学数据文件"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                # 简单的GeoJSON处理
                import geopandas as gpd
                gdf = gpd.read_file(uploaded_file)
                data = pd.DataFrame(gdf.drop(columns='geometry'))
            
            st.session_state.data = data
            st.sidebar.success(f"✅ 成功加载数据: {data.shape}")
        except Exception as e:
            st.sidebar.error(f"❌ 加载失败: {str(e)}")
    
    # 使用示例数据
    if st.sidebar.button("🎲 使用示例数据"):
        st.session_state.data = generate_mock_data()
        st.sidebar.success("✅ 已加载示例数据")
    
    # 参数设置
    st.sidebar.markdown("### ⚙️ 参数设置")
    
    # 选择目标矿种
    target_mineral = st.sidebar.selectbox(
        "目标矿种",
        ['Au', 'Ag', 'Cu', 'Pb', 'Zn'],
        index=0,
        help="选择主要找矿目标元素"
    )
    st.session_state.target_mineral = target_mineral
    
    # 选择分析元素
    if st.session_state.data is not None:
        available_elements = [col for col in st.session_state.data.columns 
                           if col not in ['X', 'Y', 'Is_Deposit']]
        
        selected_elements = st.sidebar.multiselect(
            "分析元素",
            available_elements,
            default=['Au', 'As', 'Sb', 'Hg'] if all(e in available_elements for e in ['Au', 'As', 'Sb', 'Hg']) else available_elements[:4],
            help="选择要分析的元素"
        )
        st.session_state.selected_elements = selected_elements
    
    # 初始化Agent
    st.sidebar.markdown("### 🤖 初始化智能体")
    if st.sidebar.button("🚀 Initialize Agent", type="primary"):
        if st.session_state.data is not None:
            # TODO: 替换为真实的SpatialAnalystAgent初始化
            st.session_state.agent = "Mock Agent"
            st.sidebar.success("✅ Agent已初始化")
        else:
            st.sidebar.error("❌ 请先上传数据")

# Agent聊天界面
def render_agent_chat():
    """渲染Agent聊天界面"""
    st.markdown("### 🤖 金矿智能预测对话")
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <p>🤖 <strong>智能体介绍：</strong>我是融合地质领域知识与先进大模型技术的金矿智能预测专家，
        能够为您提供专业的金矿勘探建议、数据分析和成矿预测服务。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 显示聊天历史
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 用户:</strong> {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message agent-message">
                <strong>🤖 Agent:</strong> {message['content']}
            </div>
            """, unsafe_allow_html=True)
    
    # 用户输入
    user_input = st.text_input("💬 输入您的问题:", key="user_input")
    
    if st.button("📤 发送") and user_input:
        # 添加用户消息
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # 模拟Agent响应
        # TODO: 替换为真实的SpatialAnalystAgent调用
        agent_response = mock_agent_response(user_input)
        
        # 添加Agent响应
        st.session_state.chat_history.append({
            'role': 'agent',
            'content': agent_response
        })
        
        # 清空输入框
        st.session_state.user_input = ""
        
        # 重新运行以显示新消息
        st.rerun()
    
    # 清空聊天历史
    if st.button("🗑️ 清空聊天历史"):
        st.session_state.chat_history = []
        st.rerun()

# 数据分析界面
def render_data_analysis():
    """渲染数据分析界面"""
    st.markdown("### 📊 数据预览")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        elements = st.session_state.selected_elements
        
        # 数据概览
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("样本数量", len(data))
        with col2:
            st.metric("分析元素", len(elements))
        with col3:
            st.metric("目标矿种", st.session_state.target_mineral)
        
        # 数据表格
        st.markdown("#### 📋 数据表格")
        st.dataframe(data.head(10))
        
        # 统计信息
        st.markdown("#### 📈 统计信息")
        if elements:
            stats_data = data[elements].describe()
            st.dataframe(stats_data)
        
        # 可视化区域
        st.markdown("#### 📊 可视化分析")
        
        if len(elements) >= 2:
            # 相关性热力图
            with st.expander("🔥 相关性热力图", expanded=True):
                fig = create_correlation_heatmap(data, elements)
                st.pyplot(fig)
                plt.close()
                
                # 添加AI解释按钮
                if st.button("🤖 AI解释相关性热力图", key="explain_correlation"):
                    with st.spinner("🤖 AI正在生成解释..."):
                        correlation_matrix = data[elements].corr()
                        explanation_prompt = f"""
请解释以下地球化学元素相关性热力图的分析结果：

## 相关性矩阵
{correlation_matrix.round(3).to_string()}

## 分析元素
{', '.join(elements)}

请从地质学角度解释：
1. 元素间的相关性强度和方向
2. 高相关性元素的地质意义
3. 负相关性元素的成因解释
4. 对金矿勘探的指示意义

请用简洁明了的语言解释，便于地质勘探人员理解。
"""
                        
                        api_key = st.session_state.get('deepseek_api_key', '')
                        explanation = call_deepseek_api(explanation_prompt, api_key)
                        
                        if not explanation.startswith("❌"):
                            st.markdown("**🧠 AI地质解释：**")
                            st.markdown(explanation)
                        else:
                            st.error(explanation)
            
            # R型聚类树状图
            with st.expander("🌳 R型聚类树状图", expanded=True):
                fig = create_dendrogram(data, elements)
                st.pyplot(fig)
                plt.close()
                
                # 添加AI解释按钮
                if st.button("🤖 AI解释聚类分析", key="explain_clustering"):
                    with st.spinner("🤖 AI正在生成解释..."):
                        clustering_result = analyze_clustering_results(data, elements)
                        
                        explanation_prompt = f"""
请解释以下地球化学元素R型聚类分析结果：

## 聚类分析结果
- 样本数量: {clustering_result['n_samples']}
- 特征数量: {clustering_result['n_features']}
- 聚类方法: Ward层次聚类
- 距离度量: 欧几里得距离

## 分析元素
{', '.join(elements)}

请从地质学角度解释：
1. 元素聚类的主要分组特征
2. 各聚类组的地球化学意义
3. 元素组合的地质成因解释
4. 对金矿勘探的指导意义

请用简洁明了的语言解释，便于地质勘探人员理解。
"""
                        
                        api_key = st.session_state.get('deepseek_api_key', '')
                        explanation = call_deepseek_api(explanation_prompt, api_key)
                        
                        if not explanation.startswith("❌"):
                            st.markdown("**🧠 AI地质解释：**")
                            st.markdown(explanation)
                        else:
                            st.error(explanation)
            
            # PCA载荷图
            with st.expander("🎯 PCA载荷图", expanded=True):
                fig = create_pca_loadings_plot(data, elements)
                st.pyplot(fig)
                plt.close()
                
                # 添加AI解释按钮
                if st.button("🤖 AI解释PCA分析", key="explain_pca"):
                    with st.spinner("🤖 AI正在生成解释..."):
                        pca_result = analyze_pca_results(data, elements)
                        
                        explanation_prompt = f"""
请解释以下地球化学元素PCA主成分分析结果：

## PCA分析结果
- 主成分1解释方差: {pca_result['explained_variance'][0]:.3f} ({pca_result['explained_variance'][0]*100:.1f}%)
- 主成分2解释方差: {pca_result['explained_variance'][1]:.3f} ({pca_result['explained_variance'][1]*100:.1f}%)
- 累积解释方差: {pca_result['cumulative_variance'][1]:.3f} ({pca_result['cumulative_variance'][1]*100:.1f}%)

## 主成分载荷
"""
                        
                        for i, element in enumerate(elements):
                            explanation_prompt += f"""
- {element}: PC1={pca_result['loadings'][i][0]:.3f}, PC2={pca_result['loadings'][i][1]:.3f}
"""
                        
                        explanation_prompt += f"""

## 分析元素
{', '.join(elements)}

请从地质学角度解释：
1. 主成分的地球化学意义
2. 高载荷元素的地质指示
3. 主成分组合的成因解释
4. 对金矿勘探的应用价值

请用简洁明了的语言解释，便于地质勘探人员理解。
"""
                        
                        api_key = st.session_state.get('deepseek_api_key', '')
                        explanation = call_deepseek_api(explanation_prompt, api_key)
                        
                        if not explanation.startswith("❌"):
                            st.markdown("**🧠 AI地质解释：**")
                            st.markdown(explanation)
                        else:
                            st.error(explanation)
            
            # 地质解释面板
            st.markdown("#### 🧠 AI地质解释")
            create_geological_interpretation_panel(data, elements)
        else:
            st.warning("⚠️ 请至少选择2个元素进行分析")
    else:
        st.warning("⚠️ 请先上传数据")

# 空间分析界面
def render_spatial_analysis():
    """渲染空间分析界面"""
    st.markdown("### 🗺️ 空间分析")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        target_element = st.session_state.target_mineral
        
        # 选择分析元素
        analysis_element = st.selectbox(
            "选择分析元素",
            st.session_state.selected_elements,
            index=0 if st.session_state.selected_elements else 0
        )
        
        # 分析选项
        col1, col2, col3 = st.columns(3)
        with col1:
            show_comprehensive = st.checkbox("📊 综合分析", value=True)
        with col2:
            show_variogram = st.checkbox("📈 变差函数", value=True)
        with col3:
            show_3d = st.checkbox("🎯 3D可视化", value=False)
        
        # 综合分析面板
        if show_comprehensive:
            st.markdown("#### 📊 综合地球化学分析")
            create_comprehensive_analysis_panel(data, analysis_element)
        
        # C-A分形分析
        st.markdown("#### 📈 C-A Fractal Analysis")
        
        # 初始化threshold变量
        threshold = None
        
        with st.expander("🔍 C-A Fractal Plot", expanded=True):
            processor = GeochemProcessor()
            ca_result = processor.run_fractal_ca_model(data, analysis_element)
            
            st.pyplot(ca_result['figure'])
            plt.close(ca_result['figure'])
            
            if ca_result['threshold_value']:
                threshold = ca_result['threshold_value']
                st.info(f"📍 Calculated Anomaly Threshold: {threshold:.3f}")
                
                # 添加AI解释按钮
                if st.button("🤖 AI解释C-A分形分析", key="explain_ca_fractal"):
                    with st.spinner("🤖 AI正在生成解释..."):
                        element_data = data[analysis_element].dropna()
                        
                        explanation_prompt = f"""
请解释以下地球化学C-A分形分析结果：

## C-A分形分析结果
- 分析元素: {analysis_element}
- 异常阈值: {threshold:.3f}
- 样本数量: {len(element_data)}
- 数据范围: [{element_data.min():.3f}, {element_data.max():.3f}]
- 平均值: {element_data.mean():.3f}
- 标准差: {element_data.std():.3f}

## 异常统计
- 异常样品数: {(element_data > threshold).sum()}
- 异常率: {(element_data > threshold).sum()/len(element_data)*100:.1f}%

请从地质学角度解释：
1. C-A分形分析的地质意义
2. 异常阈值的合理性评价
3. 异常分布的空间分布特征
4. 对金矿勘探的指导意义
5. 下一步勘探建议

请用简洁明了的语言解释，便于地质勘探人员理解。
"""
                        
                        api_key = st.session_state.get('deepseek_api_key', '')
                        explanation = call_deepseek_api(explanation_prompt, api_key)
                        
                        if not explanation.startswith("❌"):
                            st.markdown("**🧠 AI地质解释：**")
                            st.markdown(explanation)
                        else:
                            st.error(explanation)
        
        # 克里金插值热力图
        st.markdown("#### 🔥 Advanced Kriging Interpolation")
        
        # 初始化克里金结果变量
        kriging_result = None
        
        if st.button("Generate Kriging Heatmap", type="primary"):
            with st.spinner("Generating kriging interpolation..."):
                try:
                    processor = GeochemProcessor()
                    kriging_result = processor.interpolate_kriging(
                        data, 
                        target_element=analysis_element,
                        grid_resolution=0.01
                    )
                    
                    # 创建专业克里金分析展示
                    create_professional_kriging_display(data, analysis_element, kriging_result, threshold)
                    
                    # 显示插值统计
                    st.success("✅ 克里金插值完成!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("网格分辨率", f"{len(kriging_result['grid_x'])}x{len(kriging_result['grid_y'])}")
                    with col2:
                        st.metric("插值范围", f"[{kriging_result['extent'][0]:.2f}, {kriging_result['extent'][1]:.2f}]")
                    with col3:
                        st.metric("有效数据点", len(kriging_result['points']['x']))
                    with col4:
                        if 'variogram_params' in kriging_result:
                            range_val = kriging_result['variogram_params'].get('range', 'N/A')
                            st.metric("变程", f"{range_val:.3f}" if range_val != 'N/A' else 'N/A')
                    
                    # 添加AI解释按钮
                    if st.button("🤖 AI解释克里金插值", key="explain_kriging"):
                        with st.spinner("🤖 AI正在生成解释..."):
                            element_data = data[analysis_element].dropna()
                            
                            explanation_prompt = f"""
请解释以下地球化学克里金插值分析结果：

## 克里金插值结果
- 分析元素: {analysis_element}
- 网格分辨率: {len(kriging_result['grid_x'])}x{len(kriging_result['grid_y'])}
- 插值范围: [{kriging_result['extent'][0]:.2f}, {kriging_result['extent'][1]:.2f}]
- 有效数据点: {len(kriging_result['points']['x'])}
"""
                            
                            if 'variogram_params' in kriging_result:
                                params = kriging_result['variogram_params']
                                explanation_prompt += f"""
## 变差函数参数
- 块金值: {params.get('nugget', 'N/A')}
- 基台值: {params.get('sill', 'N/A')}
- 变程: {params.get('range', 'N/A')}
- 模型类型: {params.get('model', 'N/A')}
"""
                            
                            explanation_prompt += f"""
## 原始数据统计
- 样本数量: {len(element_data)}
- 数据范围: [{element_data.min():.3f}, {element_data.max():.3f}]
- 平均值: {element_data.mean():.3f}
- 标准差: {element_data.std():.3f}

请从地质学角度解释：
1. 克里金插值结果的可靠性评价
2. 变差函数参数的地质意义
3. 空间分布特征和连续性
4. 插值结果的勘探应用价值
5. 局限性和改进建议

请用简洁明了的语言解释，便于地质勘探人员理解。
"""
                            
                            api_key = st.session_state.get('deepseek_api_key', '')
                            explanation = call_deepseek_api(explanation_prompt, api_key)
                            
                            if not explanation.startswith("❌"):
                                st.markdown("**🧠 AI地质解释：**")
                                st.markdown(explanation)
                            else:
                                st.error(explanation)
                    
                except Exception as e:
                    st.error(f"❌ 克里金插值失败: {str(e)}")
                    st.info("💡 提示: 请确保安装了 pykrige 包: `pip install pykrige`")
                    kriging_result = None
        
        # 空间分布热力图
        st.markdown("#### 🗺️ 空间分布热力图")
        
        with st.expander("🌍 热力图展示", expanded=True):
            # 确保threshold有默认值
            if 'threshold' not in locals():
                threshold = None
            
            # 创建热力图
            create_heatmap_display(data, analysis_element, threshold, kriging_result)
        
        # 异常统计
        if threshold:
            anomaly_count = (data[analysis_element] > threshold).sum()
            st.markdown("#### 📊 异常统计")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("异常样品数", anomaly_count)
            with col2:
                st.metric("异常率", f"{anomaly_count/len(data)*100:.1f}%")
            with col3:
                st.metric("阈值", f"{threshold:.3f}")
    else:
        st.warning("⚠️ 请先上传数据")

# 主函数
def main():
    """主函数"""
    # 设置自定义样式
    set_custom_style()
    
    # 初始化session state
    init_session_state()
    
    # 渲染侧边栏
    render_sidebar()
    
    # 主界面标题
    st.markdown("""
    <div style='text-align: center; padding: 30px 0;'>
        <h1>⛏️ Gold-Seeker: AI Mineral Prediction System</h1>
        <p style='font-size: 18px; opacity: 0.9;'>融合领域知识与大模型的金矿智能预测智能体平台</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["🤖 Agent Chat", "📊 Data & R-mode Analysis", "🗺️ Spatial & Anomaly"])
    
    with tab1:
        render_agent_chat()
    
    with tab2:
        render_data_analysis()
    
    with tab3:
        render_spatial_analysis()
    
    # 页脚
    st.markdown("""
    <div style='text-align: center; padding: 20px; margin-top: 50px; border-top: 1px solid rgba(255,255,255,0.2);'>
        <p>© 2025 Gold-Seeker Development Team | 融合领域知识与大模型的金矿智能预测智能体平台</p>
    </div>
    """, unsafe_allow_html=True)

# DeepSeek API配置和地质解释功能
def get_deepseek_client():
    """获取DeepSeek客户端"""
    api_key = st.session_state.get('deepseek_api_key', '')
    if not api_key:
        return None
    return api_key

def call_deepseek_api(prompt, api_key, max_retries=3):
    """调用DeepSeek API"""
    if not api_key:
        return "❌ 请先在设置中配置DeepSeek API密钥"
    
    for attempt in range(max_retries):
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system", 
                        "content": "你是一位专业的地球化学家和地质勘探专家，具有丰富的金矿勘探经验。请基于提供的地球化学数据分析结果，给出专业的地质解释和勘探建议。"
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 1500
            }
            
            # 增加超时时间，第一次尝试30秒，后续尝试60秒
            timeout = 30 if attempt == 0 else 60
            
            response = requests.post(url, headers=headers, json=data, timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"API调用失败: {response.status_code} - {response.text}"
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(3)  # 等待3秒后重试
                continue
            else:
                return "❌ DeepSeek API响应超时，请检查网络连接或稍后重试"
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(5)  # 连接问题等待更长时间
                continue
            else:
                return "❌ 无法连接到DeepSeek API，请检查网络连接"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
                continue
            else:
                return f"调用DeepSeek API时出错: {str(e)}"

def analyze_pca_results(data, elements):
    """分析PCA结果"""
    try:
        # 标准化数据
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[elements])
        
        # PCA分析
        pca = PCA(n_components=min(3, len(elements)))
        pca_result = pca.fit_transform(scaled_data)
        
        # 获取载荷矩阵
        loadings = pca.components_.T
        
        # 解释方差
        explained_variance = pca.explained_variance_ratio_
        
        # 创建PCA结果摘要
        pca_summary = {
            'explained_variance': explained_variance,
            'cumulative_variance': np.cumsum(explained_variance),
            'loadings': loadings,
            'components': pca_result
        }
        
        return pca_summary
        
    except Exception as e:
        return None

def analyze_clustering_results(data, elements):
    """分析聚类结果"""
    try:
        # 计算距离矩阵
        distances = pdist(data[elements].values, metric='euclidean')
        
        # 层次聚类
        linkage_matrix = linkage(distances, method='ward')
        
        # 获取聚类信息
        clustering_info = {
            'linkage_matrix': linkage_matrix,
            'distance_matrix': distances,
            'n_samples': len(data),
            'n_features': len(elements)
        }
        
        return clustering_info
        
    except Exception as e:
        return None

def create_geological_interpretation_panel(data, elements):
    """创建地质解释面板"""
    
    # API密钥显示
    st.markdown("##### 🔑 DeepSeek API配置")
    
    # 显示已配置的API密钥（隐藏部分字符）
    api_key = st.session_state.get('deepseek_api_key', '')
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:]
        st.success(f"✅ API密钥已配置: {masked_key}")
    else:
        st.error("❌ API密钥未配置")
        return
    
    # 分析选项
    st.markdown("##### 🧠 AI地质解释分析")
    
    analysis_options = st.multiselect(
        "选择要分析的内容:",
        ["PCA主成分分析", "聚类分析", "相关性分析", "统计特征分析"],
        default=["PCA主成分分析", "聚类分析"]
    )
    
    if not analysis_options:
        st.info("💡 请选择要分析的内容")
        return
    
    # 生成分析按钮
    if st.button("🚀 生成地质解释", type="primary"):
        with st.spinner("🤖 AI正在分析数据..."):
            
            # 准备分析数据
            analysis_prompt = f"""
请基于以下地球化学数据进行专业的地质解释分析：

## 数据基本信息
- 样本数量: {len(data)} 个
- 分析元素: {', '.join(elements)}
- 目标矿种: {st.session_state.get('target_mineral', 'Au')}

## 统计特征
"""
            
            # 添加统计信息
            for element in elements:
                element_data = data[element].dropna()
                analysis_prompt += f"""
### {element} 元素统计
- 平均值: {element_data.mean():.3f} ppm
- 标准差: {element_data.std():.3f} ppm
- 最小值: {element_data.min():.3f} ppm
- 最大值: {element_data.max():.3f} ppm
- 偏度: {element_data.skew():.3f}
- 峰度: {element_data.kurtosis():.3f}
"""
            
            # 添加PCA分析
            if "PCA主成分分析" in analysis_options and len(elements) >= 2:
                pca_result = analyze_pca_results(data, elements)
                if pca_result:
                    analysis_prompt += f"""

## PCA主成分分析结果
- 主成分1解释方差: {pca_result['explained_variance'][0]:.3f} ({pca_result['explained_variance'][0]*100:.1f}%)
- 主成分2解释方差: {pca_result['explained_variance'][1]:.3f} ({pca_result['explained_variance'][1]*100:.1f}%)
- 累积解释方差: {pca_result['cumulative_variance'][1]:.3f} ({pca_result['cumulative_variance'][1]*100:.1f}%)

### 主成分载荷
"""
                    for i, element in enumerate(elements):
                        analysis_prompt += f"""
- {element}: PC1={pca_result['loadings'][i][0]:.3f}, PC2={pca_result['loadings'][i][1]:.3f}
"""
            
            # 添加聚类分析
            if "聚类分析" in analysis_options and len(elements) >= 2:
                clustering_result = analyze_clustering_results(data, elements)
                if clustering_result:
                    analysis_prompt += f"""

## 聚类分析结果
- 样本数量: {clustering_result['n_samples']}
- 特征数量: {clustering_result['n_features']}
- 聚类方法: Ward层次聚类
- 距离度量: 欧几里得距离
"""
            
            # 添加相关性分析
            if "相关性分析" in analysis_options and len(elements) >= 2:
                correlation_matrix = data[elements].corr()
                analysis_prompt += f"""

## 相关性分析结果
### 元素间相关系数
"""
                for i, element1 in enumerate(elements):
                    for j, element2 in enumerate(elements):
                        if i < j:
                            corr_value = correlation_matrix.loc[element1, element2]
                            analysis_prompt += f"""
- {element1} - {element2}: {corr_value:.3f}
"""
            
            # 添加地质解释请求
            analysis_prompt += """

## 地质解释要求
请基于以上数据分析结果，提供专业的地质解释，包括：

1. **地球化学特征解释**: 
   - 元素分布特征和地球化学行为
   - 元素组合关系和地球化学意义

2. **地质成因分析**:
   - 可能的矿化类型和成矿作用
   - 地质构造和岩浆活动影响

3. **勘探意义**:
   - 地球化学异常的识别和评价
   - 勘探靶区优选和找矿方向

4. **下一步工作建议**:
   - 需要补充的分析测试
   - 勘探方法和技术路线

请以专业地质学家的角度进行解释，提供实用的勘探建议。
"""
            
            # 调用DeepSeek API
            api_key = st.session_state.get('deepseek_api_key', '')
            
            # 添加网络连接检查
            try:
                # 先测试网络连接
                test_response = requests.get("https://www.baidu.com", timeout=5)
                network_ok = True
            except:
                network_ok = False
                st.error("❌ 网络连接异常，请检查网络设置")
                return
            
            if network_ok:
                interpretation = call_deepseek_api(analysis_prompt, api_key)
                
                # 显示结果
                st.markdown("##### 📋 AI地质解释结果")
                
                if interpretation.startswith("❌") or interpretation.startswith("API调用失败"):
                    st.error(interpretation)
                    
                    # 提供重试按钮
                    if st.button("🔄 重试", key="retry_interpretation"):
                        st.rerun()
                else:
                    # 使用markdown显示结果
                    st.markdown(interpretation)
                
                # 提供下载选项
                st.markdown("##### 💾 导出结果")
                if st.button("📄 下载地质解释报告"):
                    report_content = f"""
# Gold-Seeker AI地质解释报告

## 分析时间
{time.strftime('%Y-%m-%d %H:%M:%S')}

## 数据信息
- 样本数量: {len(data)} 个
- 分析元素: {', '.join(elements)}
- 目标矿种: {st.session_state.get('target_mineral', 'Au')}

## AI地质解释

{interpretation}

---
*本报告由Gold-Seeker AI系统生成，基于DeepSeek大模型分析*
"""
                    
                    st.download_button(
                        label="📥 下载报告",
                        data=report_content,
                        file_name=f"geological_interpretation_{time.strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )

if __name__ == "__main__":
    # 抑制警告
    warnings.filterwarnings('ignore')
    
    # 运行应用
    main()