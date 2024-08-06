import torch
import torch.nn as nn
import subprocess

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(20000,20000)
        self.linear2 = nn.Linear(20000,20000)

def print_GPU_usage():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stdout
    lines = output.split('\n')[8:10]

    for line in lines:
        print(line)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MLP()
# linear2 = model.linear2 # modelをdelしてもmodelに関係するオブジェクトが存在する場合には，その分のメモリを解放できない

print("Before transfering the model")
print_GPU_usage()

model.to(device)

print("The model transferred to GPU.")
print_GPU_usage()

del model
print("The model was deleted.")
print_GPU_usage()

torch.cuda.empty_cache()
print("Freed GPU memory.")
print_GPU_usage()
