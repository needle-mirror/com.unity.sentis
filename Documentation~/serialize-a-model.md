# Serialize a Model

For larger models it is best to use a serialized asset. These have the file extension ".sentis"

## Create a Serialized Assset
After you have brought the ONNX file into your asset folder you can click on it. In the inspector window, click on the button "Serlialize to StreamingAssets". This will create a serialized version of your model and save it to the StreamingAssets folder.

## Loading a Serialized Asset
The code to load a serialized model is much the same. For example if your model is called "mymodel.sentis" the code is:

```
Model model = ModelLoader.Load( Application.streamingAssetsPath + "/mymodel.sentis" );
```

## Advantages of using Serialized Models
Some advantages of using a serialized model are:

* Saves disk space in your project
* Faster loading times
* Validated to work in Unity
* Easier to share

