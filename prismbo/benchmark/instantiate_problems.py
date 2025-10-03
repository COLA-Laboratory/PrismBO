from prismbo.agent.registry import problem_registry
from prismbo.benchmark.problem_base.transfer_problem import TransferProblem, RemoteTransferOptBenchmark


def InstantiateProblems(
    tasks: list = [], seed: int = 0, remote: bool = False, server_url: str = None
) -> TransferProblem:
    tasks = tasks or []

    if remote:
        if server_url is None:
            raise ValueError("Server URL must be provided for remote testing.")
        transfer_problems = RemoteTransferOptBenchmark(server_url, seed)
    else:
        transfer_problems = TransferProblem(seed)

    for task in tasks:
        task_name = task['name']
        
        budget = int(task.get("budget", 0))
        workloads = task.get("workloads", [])
        if len(workloads) != 0:
            workloads = [int(w) for w in workloads.split(',')]
        else:
            workloads = []
        budget_type = task.get("budget_type", 'Num_FEs')
        params = {'input_dim':int(task.get("num_vars", 1)), 'output_dim':int(task.get("num_objs", 0))}

        if task_name == "MPB":
            problem_cls = problem_registry[task_name]
            if problem_cls is None:
                raise KeyError(f"Task '{task_name}' not found in the problem registry.")
            problem = problem_cls(
                    task_name=f"{task_name}",
                    budget_type=budget_type,
                    budget=budget,
                    seed=seed,
                    workloads=workloads,
                    params=params,
                    description=task.get("description", ""),
                )
            for problem in problem.generate_benchmarks():
                transfer_problems.add_task(problem)
            

        else:
            problem_cls = problem_registry[task_name]

            if problem_cls is None:
                raise KeyError(f"Task '{task_name}' not found in the problem registry.")
            for idx, workload in enumerate(workloads):
                problem = problem_cls(
                    task_name=f"{task_name}",
                    task_id=idx,
                    budget_type=budget_type,
                    budget=budget,
                    seed=seed,
                    workload=workload,
                    params=params,
                    description=task.get("description", ""),
                )
                
                transfer_problems.add_task(problem)

    return transfer_problems
