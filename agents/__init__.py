"""
Gold-Seeker: 地球化学找矿预测智能平台

基于Carranza (2009) 《Geochemical Anomaly and Mineral Prospectivity Mapping in GIS》
理论的智能地球化学找矿预测平台，集成多智能体架构与LangChain技术。

主要模块:
- coordinator: 任务协调与工作流管理
- archivist: 知识管理与GraphRAG集成
- spatial_analyst: 地球化学空间分析（核心）
- modeler: 机器学习建模与预测
- critic: 结果验证与报告生成
- tools: 地球化学分析工具集
"""

__version__ = "1.0.0"
__author__ = "Gold-Seeker Development Team"
__email__ = "your-email@example.com"
__license__ = "MIT"

# 导入主要类和函数 (暂时注释，避免依赖问题)
# from .coordinator import CoordinatorAgent
# from .archivist import ArchivistAgent
# from .spatial_analyst import SpatialAnalystAgent
# from .modeler import ModelerAgent
# from .critic import CriticAgent

# 导入工具类
from .tools.geochem.selector import GeochemSelector
from .tools.geochem.processor import GeochemProcessor
from .tools.geochem.fractal import FractalAnomalyFilter
from .tools.geochem.woe import WeightsOfEvidenceCalculator

# 导入配置和工具函数
from .config import load_config, get_default_config
from .utils import setup_logging, validate_data

__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    
    # 智能体类
    "CoordinatorAgent",
    "ArchivistAgent", 
    "SpatialAnalystAgent",
    "ModelerAgent",
    "CriticAgent",
    
    # 工具类
    "GeochemSelector",
    "GeochemProcessor",
    "FractalAnomalyFilter",
    "WeightsOfEvidenceCalculator",
    
    # 配置和工具函数
    "load_config",
    "get_default_config",
    "setup_logging",
    "validate_data",
]

# 平台信息
PLATFORM_INFO = {
    "name": "Gold-Seeker",
    "version": __version__,
    "description": "地球化学找矿预测智能平台",
    "theory": "Carranza (2009) - Geochemical Anomaly and Mineral Prospectivity Mapping in GIS",
    "architecture": "Multi-Agent System with LangChain",
    "features": [
        "R-mode clustering analysis",
        "Principal component analysis", 
        "C-A fractal anomaly filtering",
        "Weights of evidence calculation",
        "Multi-evidence layer integration",
        "AI-enhanced analysis with LangChain",
    ],
    "supported_elements": [
        "Au", "As", "Sb", "Hg", "Cu", "Pb", "Zn", "Ag", 
        "Mo", "W", "Bi", "Co", "Ni", "Cr", "V", "Ti", 
        "Mn", "Fe", "Mg", "Ca", "Na", "K"
    ],
    "dependencies": [
        "langchain", "pandas", "numpy", "geopandas", 
        "scikit-learn", "pysal", "matplotlib", "seaborn"
    ]
}

def get_platform_info():
    """获取平台信息"""
    return PLATFORM_INFO

def print_platform_info():
    """打印平台信息"""
    print(f"🏆 {PLATFORM_INFO['name']} v{PLATFORM_INFO['version']}")
    print(f"📚 {PLATFORM_INFO['description']}")
    print(f"🔬 理论基础: {PLATFORM_INFO['theory']}")
    print(f"🤖 架构: {PLATFORM_INFO['architecture']}")
    print("\n✨ 主要功能:")
    for feature in PLATFORM_INFO['features']:
        print(f"  • {feature}")
    print(f"\n🧪 支持元素: {', '.join(PLATFORM_INFO['supported_elements'][:10])}...")
    print(f"\n📦 核心依赖: {', '.join(PLATFORM_INFO['dependencies'][:5])}...")

# 模块初始化时打印平台信息
if __name__ != "__main__":
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Gold-Seeker v{__version__} 已加载")
    logger.info("基于Carranza (2009) 地球化学异常与成矿预测理论")