import torch.utils.model_zoo as model_zoo
import models
import torchvision
import pdb
import torch
import torchvision.models

# params=model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
# for i in params.keys():
# 	print(i)
model_vgg = models.DORN();pdb.set_trace()


#print(model_vgg.children)


#print(model_vgg._modules['features'])#;pdb.set_trace()
#model=models.Disp_vgg()

#print(model.state_dict)
#print(model._modules)

#print(model)

# transfer_cfg = {
#             "conv1": {0: 0, 2: 2},
#             "conv2": {0: 5, 2: 7},
#             "conv3": {0: 10, 2: 12, 4: 14},
#             "conv4": {0: 17, 2: 19, 4: 21},
#             "conv5": {0: 24, 2: 26, 4: 28}
# }
# for i, j in transfer_cfg.items():
# 	print(i)
# 	print(model._modules[i])
# 	for to_id, from_id in j.items():
# 		print(to_id)
# 		print(from_id)
# for i in model.state_dict().keys():
# 	print(i)

#self.load_vgg_params(model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth'))



# #params["features.{}.weight"]

# #print(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))	
# params_1=model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
# for i in params_1.keys():
# 	print(i)
# model_res=models.Disp_res()
# #print(model_res._modules)
# for i in model_res.state_dict().keys():
# 	print(i)