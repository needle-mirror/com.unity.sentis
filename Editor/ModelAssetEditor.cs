using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;
using System;
using UnityEngine.UIElements;
using Unity.Sentis.Compiler.Analyser;
using System.IO;

namespace Unity.Sentis.Editor
{
[CustomEditor(typeof(ModelAsset))]
public class ModelAssetEditor : UnityEditor.Editor
{
    Model m_Model;

    Foldout CreateFoldoutListView(List<string> items, string name)
    {
        Func<VisualElement> makeItem = () => new Label();
        Action<VisualElement, int> bindItem = (e, i) => (e as Label).text = items[i];

        var listView = new ListView(items, 16, makeItem, bindItem);
        listView.showAlternatingRowBackgrounds = AlternatingRowBackground.All;
        listView.showBorder = true;
        listView.selectionType = SelectionType.Multiple;
        listView.style.flexGrow = 1;
        listView.horizontalScrollingEnabled = true;

        var inputMenu = new Foldout();
        inputMenu.text = name;
        inputMenu.style.maxHeight = 400;
        inputMenu.Add(listView);

        return inputMenu;
    }

    void CreateInputListView(VisualElement rootElement)
    {
        var inputs = m_Model.inputs;
        var items = new List<string>(inputs.Count);
        foreach (var input in inputs)
            items.Add($"<b>{input.name}</b> index: {input.index}, shape: {input.shape}, dataType: {input.dataType}");

        var inputMenu = CreateFoldoutListView(items, $"<b>Inputs ({inputs.Count})</b>");
        rootElement.Add(inputMenu);
    }

    void CreateOutputListView(VisualElement rootElement)
    {
        var outputs = m_Model.outputs;
        var items = new List<string>(outputs.Count);
        try
        {
            var ctx = PartialInferenceAnalysis.InferModelPartialTensors(m_Model);
            foreach (var output in outputs)
            {
                var partialTensor = ctx.GetPartialTensor(output.index);
                items.Add($"<b>{output.name}</b> index: {output.index}, shape: {partialTensor.shape}, dataType: {partialTensor.dataType}");
            }
        }
        catch (Exception)
        {
            foreach (var output in outputs)
                items.Add($"<b>{output.name}</b> index: {output.index}");
        }
        var inputMenu = CreateFoldoutListView(items, $"<b>Outputs ({outputs.Count})</b>");
        rootElement.Add(inputMenu);
    }

    void CreateLayersListView(VisualElement rootElement)
    {
        var layers = m_Model.layers;
        var items = new List<string>(layers.Count);
        foreach (var layer in layers)
        {
            string ls = layer.ToString();
            string layerType = layer.GetType().Name;
            items.Add($"<b>{layerType}</b> {ls.Substring(ls.IndexOf('-') + 2)}");
        }

        var layerMenu = CreateFoldoutListView(items, $"<b>Layers ({layers.Count})</b>");
        rootElement.Add(layerMenu);
    }

    void CreateConstantsListView(VisualElement rootElement)
    {
        long totalWeightsSizeInBytes = 0;
        var constants = m_Model.constants;
        var items = new List<string>(constants.Count);
        foreach (var constant in constants)
        {
            string cs = constant.ToString();
            items.Add($"<b>Constant</b> {cs.Substring(cs.IndexOf('-') + 2)}");
            totalWeightsSizeInBytes += constant.lengthBytes;
        }

        var constantsMenu = CreateFoldoutListView(items, $"<b>Constants ({constants.Count})</b>");
        rootElement.Add(constantsMenu);
        rootElement.Add(new Label($"Total weight size: {totalWeightsSizeInBytes / (1024 * 1024):n0} MB"));
    }

    public void LoadAndSerializeModel(ModelAsset modelAsset, string name)
    {
        if (!Directory.Exists(Application.streamingAssetsPath))
            Directory.CreateDirectory(Application.streamingAssetsPath);
        var path = Path.Combine(Application.streamingAssetsPath, $"{name}.sentis");
        ModelWriter.Save(path, modelAsset);
        AssetDatabase.Refresh();
    }

    void CreateSerializeButton(VisualElement rootElement, ModelAsset modelAsset, string name)
    {
        var button = new Button(() => LoadAndSerializeModel(modelAsset, name));
        button.text = "Serialize To StreamingAssets";
        rootElement.Add(button);
    }

    public override VisualElement CreateInspectorGUI()
    {
        var rootInspector = new VisualElement();

        var modelAsset = target as ModelAsset;
        if (modelAsset == null)
            return rootInspector;
        if (modelAsset.modelAssetData == null)
            return rootInspector;

        m_Model ??= ModelLoader.LoadModelDescription(modelAsset.modelAssetData.value);

        CreateSerializeButton(rootInspector, modelAsset, target.name);
        CreateInputListView(rootInspector);
        CreateOutputListView(rootInspector);
        CreateLayersListView(rootInspector);
        CreateConstantsListView(rootInspector);

        rootInspector.Add(new Label($"Producer Name: {m_Model.ProducerName}"));

        return rootInspector;
    }
}
}
