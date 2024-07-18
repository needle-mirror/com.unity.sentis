using System.IO;
using UnityEngine;
using Unity.Sentis;
using System.Security.Cryptography;

public class LoadEncryptedModel : MonoBehaviour
{
    // A .bytes asset that has been saved with using AES encryption
    // with the 'Sentis > Sample Model Encryption' editor window
    [SerializeField]
    public TextAsset encryptedModel;

    void OnEnable()
    {
        // This key must match the key that was used to encrypt the model
        var key = new byte[] { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16 };

        using var memoryStream = new MemoryStream(encryptedModel.bytes);
        using Aes aes = Aes.Create();
        // Read the initialization vector from the encrypted data
        byte[] iv = new byte[aes.IV.Length];
        memoryStream.Read(iv, 0, iv.Length);

        using CryptoStream cryptoStream = new(memoryStream, aes.CreateDecryptor(key, iv), CryptoStreamMode.Read);
        // Use the ModelLoader.Load method passing in the created CryptoStream
        var model = ModelLoader.Load(cryptoStream);

        // Use the decrypted model as usual
    }
}
