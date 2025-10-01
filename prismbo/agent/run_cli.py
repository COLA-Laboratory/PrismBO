import os
import traceback
import argparse
from services import Services
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
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("-e", "--experiment_name", type=str, default="exp_2")
    parser.add_argument("-ed", "--experiment_description", type=str, default="")
    # Task

    # Seed
    # parser.add_argument("-s", "--seeds", type=str, default="5")

    args = parser.parse_args()
    services = Services(None, None, None)
    services._initialize_modules()


    # show(['Rastrigin_w0_s0_1753949959', 'Rastrigin_w1_s0_1753949960', 'Rastrigin_w2_s0_1753949961', 'Rastrigin_w0_s1_1753949962', 'Rastrigin_w1_s1_1753949963', 'Rastrigin_w2_s1_1753949964', 'Rastrigin_w0_s2_1753949965', 'Rastrigin_w1_s2_1753949965', 'Rastrigin_w2_s2_1753949966'], services.data_manager, args)

    # show([], services.data_manager, args)
    
    # show(['Ackley_w0_s0_1753947903', 'Ackley_w1_s0_1753947692', 'Ackley_w2_s0_1753947910', 'Ackley_w0_s1_1753947919', 'Ackley_w2_s1_1753947927'], services.data_manager, args)

    configurations = services.configer.get_configuration()
    seeds = configurations['seeds'].split(',')
    seeds = [int(seed) for seed in seeds]
    
    

    for seed in seeds:
        try:
            services._run_optimize_process(seed = seed, configurations=configurations)
        except Exception as e:
            traceback.print_exc()

    os.makedirs('Results', exist_ok=True)
    datasets = []
    experiment_name = args.experiment_name

    with open('Results/datasets.txt', 'a') as f:
        f.write(f"Experiment: {experiment_name}\n")
        for pid, info in services.process_info.items():
            dataset_list = info['dataset_name']
            datasets += dataset_list
            [f.write(f"{dataset}\n") for dataset in dataset_list]
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