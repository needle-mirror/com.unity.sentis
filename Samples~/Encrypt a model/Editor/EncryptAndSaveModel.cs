#if UNITY_EDITOR
using System.IO;
using Unity.Sentis;
using UnityEditor;
using UnityEngine;
using System.Security.Cryptography;

// A custom editor window to demonstrate saving an onnx model as an encrypted Sentis model
public class EncryptAndSaveModel : EditorWindow
{
    public ModelAsset modelAsset;

    [MenuItem("Sentis/Sample/Encrypt And Save Model")]
    public static void ShowExample()
    {
        EncryptAndSaveModel wnd = GetWindow<EncryptAndSaveModel>();
        wnd.titleContent = new GUIContent("Encrypt And Save Model");
    }

    void OnGUI()
    {
        EditorGUILayout.BeginHorizontal();
        modelAsset = EditorGUILayout.ObjectField(modelAsset, typeof(ModelAsset), true) as ModelAsset;
        EditorGUILayout.EndHorizontal();

        GUILayout.Space(10);

        if (!GUILayout.Button("Encrypt"))
            return;

        var path = EditorUtility.SaveFilePanel("Save encrypted model", "", modelAsset.name + ".bytes", "bytes");

        var model = ModelLoader.Load(modelAsset);

        using FileStream fileStream = new FileStream(path, FileMode.OpenOrCreate);
        using Aes aes = Aes.Create();
        // The same key must be used to decrypt the model at runtime
        aes.Key = new byte[] { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16 };

        byte[] iv = aes.IV;
        fileStream.Write(iv, 0, iv.Length);

        using CryptoStream cryptoStream = new CryptoStream(fileStream, aes.CreateEncryptor(), CryptoStreamMode.Write);
        ModelWriter.Save(cryptoStream, model);
    }
}
#endif
