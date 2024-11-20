import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 激活函数映射
activation_map = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid()
}

class CustomModel(nn.Module):
    def __init__(self, num_layers, units_per_layer, activation, dropout_rate, input_size = 28*28, output_size=10):
        super(CustomModel, self).__init__()
        
        # 定义隐藏层
        layers = []
        input_size = input_size 
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, units_per_layer))
            layers.append(activation_map[activation])  # 添加激活函数
            layers.append(nn.Dropout(dropout_rate))    # 添加 Dropout
            input_size = units_per_layer
        self.hidden_layers = nn.Sequential(*layers)
        
        # 输出层
        self.output_layer = nn.Linear(input_size, output_size)  # 假设10类分类任务

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

# 模型构建函数
def build_model(params):
    # 创建模型实例
    model = CustomModel(
        num_layers=params["num_layers"],
        units_per_layer=params["units_per_layer"],
        activation=params["activation"],
        dropout_rate=params["dropout_rate"]
    )

    
    return model

# # 示例参数
# params = {
#     "num_layers": 3,
#     "units_per_layer": 64,
#     "activation": "relu",
#     "learning_rate": 0.001,
#     "batch_size": 32,
#     "dropout_rate": 0.2,
#     "l2_reg_strength": 1e-5
# }

# # 使用示例
# model, optimizer, criterion = build_model(params)
# print(model)
