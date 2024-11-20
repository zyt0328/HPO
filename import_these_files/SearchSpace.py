import numpy as np
import itertools



# 定义超参数搜索空间
search_space = {
    "num_layers": {"values": [1, 2, 3, 4], "type": "discrete"},
    "units_per_layer": {"min": 2, "max": 128, "type": "continuous"},
    "activation": {"values": ["relu", "tanh", "sigmoid"], "type": "discrete"},
    "learning_rate": {"min": 0.0001, "max": 0.3, "scale": "log", "type": "continuous"},
    "batch_size": {"values": [16, 32, 64, 128, 256], "type": "discrete"},
    "dropout_rate": {"min": 0.05, "max": 0.6, "type": "continuous"},
    "l2_reg_strength": {"min": 1e-7, "max": 1e-2, "scale": "log", "type": "continuous"}
}
# 搜索空间函数，根据不同策略处理连续和离散参数
def get_search_space(strategy="random"):
    space = {}
    for param, config in search_space.items():
        if strategy in ["grid", "mab"] and config["type"] == "continuous":
            # 网格搜索和多臂赌博机：将连续参数离散化为4个值
            values = np.logspace(np.log10(config["min"]), np.log10(config["max"]), num=4) if config.get("scale") == "log" else np.linspace(config["min"], config["max"], num=4)
            # 转换 units_per_layer 为整数列表
            if param == "units_per_layer":
                values = list(map(int, values))
            space[param] = {"values": values}
        elif strategy in ["random", "genetic", "bayesian", "hyperband", "successive_halving"] and config["type"] == "continuous":
            # 对于随机搜索、遗传算法、贝叶斯优化、Hyperband、Successive Halving：保留连续范围
            space[param] = config
        else:
            # 离散值和分类值
            space[param] = config
    return space

# 构建参数组合的函数
def create_param_combinations(strategy="grid"):
    space = get_search_space(strategy)
    param_combinations = []

    # 处理不同的搜索策略
    if strategy in ["grid", "mab"]:
        # 对离散化后的空间进行组合
        keys, values = zip(*[(key, config["values"]) for key, config in space.items() if "values" in config])
        for combination in itertools.product(*values):
            param_combination = {k: v for k, v in zip(keys, combination)}
            param_combinations.append(param_combination)
    else:
        # 随机搜索、贝叶斯优化、遗传算法、Hyperband、Successive Halving
        param_combination = {}
        for param, config in space.items():
            if "values" in config:
                # 离散的选择
                param_combination[param] = np.random.choice(config["values"])
            elif config["type"] == "continuous":
                # 连续范围的随机值
                if config.get("scale") == "log":
                    value = np.exp(np.random.uniform(np.log(config["min"]), np.log(config["max"])))
                else:
                    value = np.random.uniform(config["min"], config["max"])

                # 确保 units_per_layer 为整数
                if param == "units_per_layer":
                    value = int(value)
                
                param_combination[param] = value  # 将连续变量添加到组合中
        
        # 将单次完整组合添加到组合列表中
        param_combinations.append(param_combination)
    
    return param_combinations

# 示例：生成参数组合
grid_params = create_param_combinations(strategy="grid")
print("Grid Search Combinations:", grid_params)

# 示例：生成遗传算法的随机参数组合
genetic_params = create_param_combinations(strategy="genetic")
print("Genetic Search Combinations:", genetic_params)


# # 使用示例
# grid_params = create_param_combinations(strategy="grid")
# random_params = create_param_combinations(strategy="random")
# mab_params = create_param_combinations(strategy="mab")
# hyperband_params = create_param_combinations(strategy="hyperband")
# successive_halving_params = create_param_combinations(strategy="successive_halving")

# # 示例输出
# grid_params = create_param_combinations(strategy="grid")
# print("Grid Search Combinations:", grid_params)
# print("Random Search Combinations:", random_params)
# print("Multi-Armed Bandit Combinations:", mab_params)
# print("Hyperband Combinations:", hyperband_params)
# print("Successive Halving Combinations:", successive_halving_params)
