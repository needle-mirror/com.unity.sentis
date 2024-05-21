# Serialize a Model

For larger models, it is advisable to use a serialized asset, which typically comes with the file extension `.sentis`.

## Create a Serialized Asset

Once you've imported the ONNX file into your asset folder, you can click on it. In the **Inspector** window, click  **Serialize to StreamingAssets**. This will create a serialized version of your model and save it in the **StreamingAssets** folder.

## Loading a Serialized Asset

The code to load a serialized model is much the same. For example, if your model is called `mymodel.sentis`, the corresponding code is:

```
Model model = ModelLoader.Load(Application.streamingAssetsPath + "/mymodel.sentis");
```

## Advantages of using Serialized Models

Some advantages of using a serialized model are:

* Saves disk space in your project
* Faster loading times
* Validated to work in Unity
* Easier to share


## Serialization layout

A `.sentis` file is serialized using `FlatBuffers` as follows

```
             ┌───────────────────────────────────┐
             │Flatbuffer-serialized              |
             | model desription                  │
             │                                   │
          ┌─ ├───────────────────────────────────┤
          │  │ Weight chunk data                 |
          │  │                                   │
          │  │                                   │
Weights  ─┤  ├───────────────────────────────────┤
          │  │ Weight chunk data                 │
          │  │                                   │
          │  │                                   │
          │  ├───────────────────────────────────┤
          │  │...                                │
          └─ └───────────────────────────────────┘
```
Refer to `Sentis/Runtime/Core/Serialization/program.fbs` for more info
