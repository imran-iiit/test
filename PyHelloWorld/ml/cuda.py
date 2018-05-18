import torch

a = torch.randn(10).cuda()
# a = torch.randn(10)
print(a)
print(a + 2)