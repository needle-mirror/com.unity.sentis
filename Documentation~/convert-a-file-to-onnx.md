## Convert a file to the ONNX format

Neural networks can be saved in a few different formats. Currently Sentis only supports the ONNX format which is the open standard, so if your model is not in that format you will need to convert it.

There are two types of neural network files; those that contain the graph of the model (how the layers are connected) and those that contain only the weights. Files that contain the whole model are easiest to convert.

There are some tools available to help you do this. 

### Converting Tensorflow files to ONNX
Tensorflow model files usually have the *.pb file extension. You can use the commandline tool [tf2onnx](https://github.com/onnx/tensorflow-onnx). It works best if you provide the full path names.

Tensorflow also has another format called "checkpoints" which saves the graph and weights separately. These have extensions *.ckpt.meta and *.ckpt respectively. If you have these you can use the same tool and point it to the meta file.

If you have only the *.ckpt file you will need to find the python code which constructs the model and load in the weights. Then [export the model to ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html).

### Converting PyTorch files to ONNX
PyTorch model files usually have the *.pt file extension. You will need to first [load the model](https://pytorch.org/tutorials/beginner/saving_loading_models.html) in python and [export it](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html) as an ONNX file. Opset 15 or higher is recommended when exporting. The *.pt file may or may not contain the model graph. If it doesn't you will need to find the python script that constructs the graph before loading the weights.

