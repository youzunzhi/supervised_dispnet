import torch.utils.model_zoo as model_zoo
import models

params=model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth')
for i in params.keys():
	print(i)

model=models.Disp_vgg()
print(model._module)

for i in model.state_dict().keys():
	print(i)

#params["features.{}.weight"]

#print(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))	