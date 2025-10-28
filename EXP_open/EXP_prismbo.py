import os
import traceback
import argparse
from prismbo.agent.services import Services
from prismbo.analysis.pipeline import analysis, comparison, show
from prismbo.analysis import *


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def set_task(services, args):
    task_info = [{
        "name": args.task_name,
        "num_vars": args.num_vars,
        "num_objs": args.num_objs,
        "fidelity": args.fidelity,
        "workloads": args.workloads,
        "budget_type": args.budget_type,
        "budget": args.budget,
    }]
    services.receive_tasks(task_info)



if __name__ == "__main__":
    services = Services(None, None, None)
    services._initialize_modules()
    
    configurations = services.configer.get_configuration()
    results_dir = f'./results/prismbo_{configurations["experimentName"]}'
    os.makedirs(results_dir, exist_ok=True)

    seeds = configurations['seeds'].split(',')
    seeds = [int(seed) for seed in seeds]

    for seed in seeds:
        try:
            services._run_optimize_process(seed = seed, configurations=configurations)
        except Exception as e:
            traceback.print_exc()

    os.makedirs(results_dir, exist_ok=True)
    experiment_name = configurations["experimentName"]
    
    ab = analysis(services.data_manager)
    datasets = ab.get_data_by_expame(experiment_name)
    with open(f'{results_dir}/datasets.txt', 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        for dataset_name, dataset_data in datasets.items():
            f.write(f"{dataset_name}\n")
        f.write("-----\n")
    


    # comparison_experiment_names = ['exp_1']
    # comparison_datasets = {}
    # analysis_datasets = {}
    # with open('Results/datasets.txt', 'r') as f:
    #     lines = f.readlines()
    #     for i in range(len(lines)):
    #         if lines[i].startswith("Experiment:"):
    #             experiment_name = lines[i].strip().split(":")[1].strip()
    #             dataset_list = []
    #             i += 1
    #             while i < len(lines) and not lines[i].startswith("-----"):
    #                 dataset_list.append(lines[i].strip())
    #                 i += 1
    #             comparison_datasets[experiment_name] = dataset_list
    #             analysis_datasets[experiment_name] = dataset_list
    

    # comparison('Results', comparison_datasets, services.data_manager, args)
    
    # analysis('Results', analysis_datasets, services.data_manager, args)