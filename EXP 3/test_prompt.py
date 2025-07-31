


def generate_test_cases(case_number):
    
    # 定义所有case的基础模板
    base_cases = [
        {
            "id": "query_datasets_clear",
            "expected_function": "get_all_datasets",
            "category": "data_query",
            "input_type": "clear_instruction"
        },
        {
            "id": "query_datasets_vague",
            "expected_function": "get_all_datasets", 
            "category": "data_query",
            "input_type": "vague_semantic"
        },
        {
            "id": "query_dataset_info",
            "expected_function": "get_dataset_info",
            "category": "data_query",
            "input_type": "clear_instruction"
        },
        {
            "id": "explain_problems",
            "expected_function": "get_all_problems",
            "category": "module_explanation", 
            "input_type": "clear_instruction"
        },
        {
            "id": "explain_techniques",
            "expected_function": "get_optimization_techniques",
            "category": "module_explanation",
            "input_type": "synonym_expression"
        },
        {
            "id": "setup_optimization_clear",
            "expected_function": "set_optimization_problem",
            "category": "experiment_config",
            "input_type": "clear_instruction"
        },
        {
            "id": "setup_optimization_vague",
            "expected_function": "set_optimization_problem",
            "category": "experiment_config", 
            "input_type": "vague_semantic"
        },
        {
            "id": "set_model",
            "expected_function": "set_model",
            "category": "experiment_config",
            "input_type": "clear_instruction"
        },
        {
            "id": "compound_query",
            "expected_function": ["get_all_datasets", "get_all_problems"],
            "category": "compound_question",
            "input_type": "compound_question"
        },
        {
            "id": "ambiguous_input",
            "expected_function": None,
            "category": "robustness",
            "input_type": "ambiguous_input"
        },
        {
            "id": "typo_input", 
            "expected_function": "get_all_datasets",
            "category": "robustness",
            "input_type": "typo_input"
        },
        {
            "id": "show_config",
            "expected_function": "show_configuration",
            "category": "status_query",
            "input_type": "clear_instruction"
        },
        {
            "id": "synonym_datasets",
            "expected_function": "get_all_datasets",
            "category": "data_query",
            "input_type": "synonym_expression"
        }
    ]
    
    # 为每个case定义20种不同的输入表达方式
    input_variations = {
        "query_datasets_clear": [
            "What datasets are available in PrismBO?",
            "Can you list all the datasets in PrismBO?",
            "Which datasets can I use in PrismBO?",
            "Give me the available datasets in your system.",
            "Please show me all the datasets available in PrismBO system",
            "I need to see what datasets are in your system",
            "Could you list the available datasets?",
            "Show me the datasets in PrismBO",
            "What datasets do you have in your system?",
            "List all available datasets in PrismBO",
            "Display the datasets available in your system",
            "Tell me what datasets are available",
            "Which datasets are included in PrismBO?",
            "Show me all the datasets you have",
            "What datasets can I access in PrismBO?",
            "Give me a list of all datasets",
            "Show the available datasets in your system",
            "What datasets are supported by PrismBO?",
            "List the datasets available in your system",
            "Show me what datasets you have available"
        ],
        "query_datasets_vague": [
            "Show me what data you have",
            "What kind of data is in your system?",
            "What data do you include?",
            "Let me see your data resources.",
            "What kind of data resources do you have access to?",
            "Tell me about your data collection",
            "What data is available here?",
            "Show me your data",
            "What kind of data do you have?",
            "Let me see what data you've got",
            "What data resources are available?",
            "Show me the data you have",
            "What kind of data can I access?",
            "Tell me about your data",
            "What data do you provide?",
            "Show me your available data",
            "What kind of data is available?",
            "Let me see your data collection",
            "What data resources do you have?",
            "Show me what data you can provide"
        ],
        "query_dataset_info": [
            "Tell me about the Rosenbrock dataset",
            "Give me details on the Rosenbrock dataset.",
            "What's in the Rosenbrock dataset?",
            "Can you explain the Rosenbrock dataset to me?",
            "I'd like to know more about the Rosenbrock dataset",
            "Can you provide details on Rosenbrock?",
            "Tell me about the Rosenbrock dataset",
            "What information do you have about Rosenbrock?",
            "Show me details about the Rosenbrock dataset",
            "Explain the Rosenbrock dataset to me",
            "What can you tell me about Rosenbrock?",
            "Give me information about the Rosenbrock dataset",
            "What's the Rosenbrock dataset like?",
            "Tell me more about Rosenbrock",
            "What details are available for Rosenbrock?",
            "Show me the Rosenbrock dataset information",
            "What do you know about the Rosenbrock dataset?",
            "Give me the details of Rosenbrock",
            "What's special about the Rosenbrock dataset?",
            "Tell me everything about the Rosenbrock dataset"
        ],
        "explain_problems": [
            "List all supported optimization problems",
            "Which optimization problems can I run?",
            "What kinds of optimization problems are available?",
            "Show me the problems your system supports.",
            "What optimization problems does your system support?",
            "Which problems can I optimize with this system?",
            "Show me the available optimization problems",
            "What problems can I solve with your system?",
            "List the optimization problems available",
            "Show me what problems you support",
            "What optimization problems are available?",
            "Tell me about the problems you can solve",
            "What problems does your system handle?",
            "Show me all the problems available",
            "What kinds of problems can I work with?",
            "List all the problems you support",
            "What problems are supported by your system?",
            "Show me the problems I can solve",
            "What optimization problems do you have?",
            "Tell me what problems are available"
        ],
        "explain_techniques": [
            "What optimization techniques do you support?",
            "List the optimization methods you provide.",
            "Which optimization strategies are implemented?",
            "What kinds of optimization approaches can I choose from?",
            "What optimization methods and techniques are available?",
            "Which optimization algorithms do you support?",
            "What techniques can I use for optimization?",
            "What optimization methods are available?",
            "Show me the optimization techniques",
            "What techniques do you support for optimization?",
            "List the optimization strategies available",
            "What optimization approaches can I use?",
            "Show me what optimization methods you have",
            "What techniques are available for optimization?",
            "Tell me about your optimization methods",
            "What optimization strategies do you support?",
            "Show me the available optimization techniques",
            "What methods can I use for optimization?",
            "List all optimization techniques available",
            "What optimization approaches are supported?"
        ],
        "setup_optimization_clear": [
            "Set up an optimization task on Rosenbrock with workload 10 and budget 50",
            "Configure an optimization on Rosenbrock using 10 workload and 50 budget.",
            "Create an experiment for Rosenbrock: 10 workloads, 50 budget.",
            "Initialize optimization with Rosenbrock, workload=10, budget=50.",
            "Please set up an optimization task for Rosenbrock with 10 workload and 50 budget",
            "Configure optimization: Rosenbrock, workload=10, budget=50",
            "Set up Rosenbrock optimization with 10 samples and 50 evaluations",
            "Create an optimization task for Rosenbrock with 10 workload and 50 budget",
            "Initialize Rosenbrock optimization with workload 10 and budget 50",
            "Set up optimization experiment: Rosenbrock, 10 workload, 50 budget",
            "Configure Rosenbrock optimization with 10 workload and 50 budget",
            "Create optimization setup for Rosenbrock: 10 workload, 50 budget",
            "Set up Rosenbrock with workload 10 and budget 50",
            "Initialize optimization task: Rosenbrock, 10 workload, 50 budget",
            "Configure experiment for Rosenbrock optimization: 10 workload, 50 budget",
            "Set up Rosenbrock optimization experiment with 10 workload and 50 budget",
            "Create optimization configuration: Rosenbrock, 10 workload, 50 budget",
            "Initialize Rosenbrock experiment with workload 10 and budget 50",
            "Set up optimization for Rosenbrock: 10 workload, 50 budget",
            "Configure Rosenbrock task with 10 workload and 50 budget"
        ],
        "setup_optimization_vague": [
            "I want to optimize Rosenbrock, use 10 samples and 50 evaluations",
            "Run optimization on Rosenbrock with 10 runs and 50 tries.",
            "Do an optimization for Rosenbrock, 10 points, 50 tests.",
            "Let's try Rosenbrock with about 10 samples and a budget of 50.",
            "I want to run optimization on Rosenbrock using 10 iterations and 50 function calls",
            "Let's optimize Rosenbrock with 10 runs and 50 attempts",
            "Do optimization for Rosenbrock: 10 points, 50 budget",
            "Run optimization on Rosenbrock with 10 samples and 50 evaluations",
            "Let's do Rosenbrock optimization with 10 runs and 50 tries",
            "I want to optimize Rosenbrock using 10 points and 50 budget",
            "Run Rosenbrock optimization with 10 samples and 50 evaluations",
            "Let's try optimizing Rosenbrock with 10 runs and 50 attempts",
            "Do optimization on Rosenbrock: 10 samples, 50 budget",
            "Run Rosenbrock with 10 points and 50 evaluations",
            "Let's optimize Rosenbrock using 10 samples and 50 tries",
            "I want to run Rosenbrock optimization with 10 points and 50 budget",
            "Do optimization for Rosenbrock with 10 runs and 50 evaluations",
            "Let's run Rosenbrock optimization: 10 samples, 50 budget",
            "Run optimization on Rosenbrock using 10 points and 50 tries",
            "Let's do Rosenbrock with 10 samples and 50 evaluations"
        ],
        "set_model": [
            "Set the surrogate model to MTGP",
            "Use MTGP as the surrogate model.",
            "I want MTGP to be the model.",
            "Switch the model to MTGP.",
            "Please configure MTGP as the surrogate model for optimization",
            "Set the model to MTGP please",
            "Use MTGP as the optimization model",
            "Configure MTGP as the surrogate model",
            "Set MTGP as the model for optimization",
            "Use MTGP model for optimization",
            "Switch to MTGP surrogate model",
            "Set the optimization model to MTGP",
            "Configure MTGP as the model",
            "Use MTGP for surrogate modeling",
            "Set MTGP as the optimization model",
            "Switch the surrogate model to MTGP",
            "Use MTGP model",
            "Set MTGP as the model",
            "Configure optimization model to MTGP",
            "Use MTGP surrogate model",
            "Set the model to MTGP for optimization"
        ],
        "compound_query": [
            "What datasets do you have and what problems can I solve with them?",
            "Which datasets are supported and what can I do with them?",
            "List your datasets and the problems associated with each.",
            "Tell me what data you support and which problems they relate to.",
            "What datasets are available and what optimization problems can I solve with them?",
            "Show me your datasets and the problems they're used for",
            "List available data and corresponding optimization tasks",
            "What datasets do you have and what can I do with them?",
            "Show me datasets and their associated problems",
            "Tell me about your datasets and what problems they solve",
            "What data do you have and what problems can I work on?",
            "List your datasets and what problems they're for",
            "Show me what datasets you have and their problems",
            "What datasets are available and what can I solve?",
            "Tell me about your data and the problems they address",
            "Show datasets and what problems they solve",
            "What data do you provide and what problems can I tackle?",
            "List available datasets and their problem types",
            "Show me your data and what problems they're used for",
            "What datasets do you have and what problems do they solve?"
        ],
        "ambiguous_input": [
            "Show me the stuff",
            "Can I see the things you have?",
            "What's in there?",
            "Let me see it all.",
            "Show me everything you have",
            "What's available?",
            "Give me the overview",
            "Show me what you have",
            "What do you have?",
            "Let me see everything",
            "Show me all the things",
            "What's in your system?",
            "Give me a look at everything",
            "Show me what's available",
            "What do you have to offer?",
            "Let me see what you've got",
            "Show me all available options",
            "What's in your collection?",
            "Give me an overview of everything",
            "Show me what you can do"
        ],
        "typo_input": [
            "What datasts are avalable?",
            "Wht datasest do you hv?",
            "Which dtaasets are availabe?",
            "List all availble datats.",
            "What datasts are avalble in your sytem?",
            "Which datsets can I use?",
            "List all availble datatsets",
            "What datasts do you hv?",
            "Which dtaasets are avalable?",
            "List all availble datats",
            "What datasts are in your sytem?",
            "Which datsets can I acces?",
            "List all availble datatsets",
            "What datasts do you have?",
            "Which dtaasets are availble?",
            "List all availble datats",
            "What datasts are availble?",
            "Which datsets do you hv?",
            "List all availble datatsets",
            "What datasts can I use?"
        ],
        "show_config": [
            "Show me the current configuration",
            "Display current setup.",
            "What is the configuration right now?",
            "Tell me how things are configured currently.",
            "Can you show me the current configuration settings?",
            "What's the current setup?",
            "Display the present configuration",
            "Show me the current setup",
            "What's the configuration now?",
            "Tell me the current configuration",
            "Show me how things are set up",
            "What's the current configuration?",
            "Display current configuration",
            "Show me the present setup",
            "What configuration do you have now?",
            "Tell me about the current setup",
            "Show me the current configuration settings",
            "What's the present configuration?",
            "Display how things are configured",
            "Show me the current configuration state"
        ],
        "synonym_datasets": [
            "Display available datasets",
            "List all the datasets that are available.",
            "What datasets can I choose from?",
            "Show datasets I can access.",
            "Which datasets are accessible in your system?",
            "What datasets can I work with?",
            "Show me the datasets I can use",
            "Display all available datasets",
            "Show me what datasets I can access",
            "List datasets that are available",
            "What datasets can I use?",
            "Show me available datasets",
            "Display datasets I can work with",
            "List all accessible datasets",
            "What datasets are available to me?",
            "Show me datasets I can choose from",
            "Display available data collections",
            "List datasets I can access",
            "What datasets can I work on?",
            "Show me the available data collections"
        ]
    }
    
    # 生成指定序号的cases
    cases = []
    for base_case in base_cases:
        case_id = base_case["id"]
        input_text = input_variations[case_id][(case_number - 1) % len(input_variations[case_id])]
        
        case = {
            "id": case_id,
            "input": input_text,
            "expected_function": base_case["expected_function"],
            "category": base_case["category"],
            "input_type": base_case["input_type"]
        }
        cases.append(case)
    
    return cases

# 使用示例：
# cases1 = generate_test_cases(1)
# cases2 = generate_test_cases(2)
# ...
# cases20 = generate_test_cases(20)

# 生成所有20个cases
all_cases = {}
for i in range(1, 21):
    all_cases[f"cases{i}"] = generate_test_cases(i)

# 可以直接使用 all_cases["cases1"], all_cases["cases2"], etc.
# 或者单独获取：
# cases1 = all_cases["cases1"]
# cases2 = all_cases["cases2"]
# ...