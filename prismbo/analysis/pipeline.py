from pathlib import Path
from prismbo.analysis.analysisbase import AnalysisBase
from prismbo.analysis.comparisonbase import ComparisonBase
from prismbo.agent.registry import analysis_registry, comparison_registry
from prismbo.analysis.plot import *
from prismbo.analysis.statistics import *



def analysis(Exper_folder, datasets, data_manager, args):
    ab = AnalysisBase(Exper_folder, datasets, data_manager)
    ab.read_data_from_db()
    Exp_folder = Path(Exper_folder) / 'analysis'
    
    for name in analysis_registry.list_names():
        func = analysis_registry.get(name)
        func(ab, Exp_folder)
    

def comparison(Exper_folder, datasets, data_manager, args):
    cb = ComparisonBase(Exper_folder, datasets, data_manager)
    cb.read_data_from_db()
    Exp_folder = Path(Exper_folder) / 'comparison'
    
    for name in comparison_registry.list_names():
        func = comparison_registry.get(name)
        func(cb, Exp_folder) 



