
from models import UnetModel
import torch

model = UnetModel(1,1,32,4,0)
x = torch.rand([1,1,256,256])
y = model(x)
print (model)
print (y[0].shape)
