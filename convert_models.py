import torch
from unet import *


# BETTI MODEL
model = UNet(1, 1, 128, True, True)
model.load_state_dict(torch.load("betti_best.pth",map_location=torch.device('cpu')))


# print model architecture
# print(model)
model.eval()

model.to('cpu')

X = torch.rand(1, 1, 304, 304)
# onnx_program = torch.onnx.dynamo_export(model, X, opset_version=10)
# Export the model
torch.onnx.export(model,               # model being run
                  X,                         # model input (or a tuple for multiple inputs)
                  "betti_oo.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
# onnx_program.save("bettiO.onnx")

# traced_model = torch.jit.trace(model.forward, X)

# traced_model.save('betti_best.pt')



# Vanilla
model = UNet(1, 1, 128, True, True)
model.load_state_dict(torch.load("vanilla_best.pth",map_location=torch.device('cpu')))


# print model architecture
# print(model)
model.eval()

model.to('cpu')

X = torch.rand(1, 1, 304, 304)
torch.onnx.export(model,               # model being run
                  X,                         # model input (or a tuple for multiple inputs)
                  "vanilla_oo.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

# onnx_program = torch.onnx.dynamo_export(model, X, opset_version=10, do_constant_folding=True)
# onnx_program.save("vanillaO.onnx")

# traced_model = torch.jit.trace(model.forward, X)

# traced_model.save('vanilla_best.pt')