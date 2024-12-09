import torch
from models import *
from torch_flops import TorchFLOPsByFX


net  = ASTP_s()  # 定义好的网络模型

input_tensor = torch.randn(1, 3, 3, 256, 256) 

flops_counter = TorchFLOPsByFX(net)
    # flops_counter.graph_model.graph.print_tabular()
flops_counter.propagate(input_tensor)
flops_counter.print_result_table()
flops_1 = flops_counter.print_total_flops(show=False)
print(f"torch_flops: {flops_1} FLOPs")
print("=" * 80)