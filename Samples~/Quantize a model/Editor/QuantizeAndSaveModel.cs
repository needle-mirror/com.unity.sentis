#if UNITY_EDITOR
using System;
using Unity.Sentis;
using UnityEditor;
using UnityEngine;

// A custom editor window to demonstrate saving an onnx model as an Quantized Sentis model
public class QuantizeAndSaveModel : EditorWindow
{
    public ModelAsset modelAsset;
    public QuantizationType quantizationType;

    [MenuItem("Sentis/Sample/Quantize and save model")]
    public static void ShowExample()
    {
        QuantizeAndSaveModel wnd = GetWindow<QuantizeAndSaveModel>();
        wnd.titleContent = new GUIContent("Quantize and save model");
    }

    void OnGUI()
    {
        EditorGUILayout.BeginHorizontal();
        modelAsset = EditorGUILayout.ObjectField("Source model", modelAsset, typeof(ModelAsset), true) as ModelAsset;
        EditorGUILayout.EndHorizontal();
        quantizationType = (QuantizationType)EditorGUILayout.EnumPopup("Quantization type", quantizationType);
        GUILayout.Space(10);

        if (!GUILayout.Button("Quantize and save") || modelAsset == null)
            return;

        var path = EditorUtility.SaveFilePanel("Save Quantized model", "", modelAsset.name + "_" + Enum.GetName(typeof(QuantizationType), quantizationType), "sentis");
        if (string.IsNullOrEmpty(path))
            return;

        // Load model using ModelLoader.
        var model = ModelLoader.Load(modelAsset);
        // Quantize model using ModelQuantizer.
        ModelQuantizer.QuantizeWeights(quantizationType, ref model);
        // Save model using ModelWriter.
        ModelWriter.Save(path, model);
        AssetDatabase.Refresh();
    }
}
#endif
