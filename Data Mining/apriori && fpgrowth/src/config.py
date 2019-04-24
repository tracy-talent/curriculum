import os

# 工程路径
PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))

# 输入文件所在目录
INPUT_PATH = os.path.join(PROJECT_PATH, 'input')

# Grocery dataset
GROCERY_PATH = os.path.join(INPUT_PATH, 'GroceryStore')

# UNIX_usage dataset
UNIX_PATH = os.path.join(INPUT_PATH, 'UNIX_usage')

# 输出文件所在目录
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'output')

# mnist数据集路径
MNIST_PATH = os.path.join(INPUT_PATH, 'MNIST_data')
