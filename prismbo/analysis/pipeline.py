from pathlib import Path
from prismbo.analysis.analysisbase import AnalysisBase
from prismbo.analysis.comparisonbase import ComparisonBase, ShowDatasetInfo
from prismbo.agent.registry import analysis_registry, comparison_registry
from prismbo.analysis.plot import *
from prismbo.analysis.statistics import *



def analysis(data_manager):
    ab = AnalysisBase(data_manager)
    return ab
    # ab.read_data_from_db()
    # Exp_folder = Path(Exper_folder) / 'analysis'
    
    # for name in analysis_registry.list_names():
    #     func = analysis_registry.get(name)
    #     func(ab, Exp_folder)
    

def comparison(Exper_folder, datasets, data_manager):
    cb = ComparisonBase(Exper_folder, datasets, data_manager)
    cb.read_data_from_db()
    Exp_folder = Path(Exper_folder) / 'comparison'
    
    for name in comparison_registry.list_names():
        func = comparison_registry.get(name)
        func(cb, Exp_folder) 


def show(datasets_name, data_manager):
    sd = ShowDatasetInfo(datasets_name, data_manager)
    # print(sd._all_data)
    for name in datasets_name:
        # 分析 f1 值的统计信息
        f1_values = [item['f1'] for item in sd._all_data[name] if 'f1' in item]
        
        if f1_values:
            import numpy as np
            
            f1_array = np.array(f1_values)
            mean_f1 = np.mean(f1_array)
            variance_f1 = np.var(f1_array)
            max_f1 = np.max(f1_array)
            min_f1 = np.min(f1_array)
            
            print(f"\n=== F1 值统计信息 ===")
            print(f"数据点数量: {len(f1_values)}")
            print(f"均值: {mean_f1:.6f}")
            print(f"方差: {variance_f1:.6f}")
            print(f"最大值: {max_f1:.6f}")
            print(f"最小值: {min_f1:.6f}")
            print(f"标准差: {np.std(f1_array):.6f}")
            print(f"范围: {max_f1 - min_f1:.6f}")

    else:
        print("数据为空")


