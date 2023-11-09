# Encrypt a model

You can encrypt a Sentis model to disk using the `ModelWriter` and `ModelLoader` APIs, so that the model description and weights can only be read from disk by a user with the correct key.

## Encrypt a model and save to disk

Follow these steps, usually in the Unity Editor before building and distributing your project.

1. Get a Sentis model by importing an ONNX file or using the Sentis model API.
2. Create a `Stream` object for the encrypted model using a cryptography API with your key.
3. Encrypt and write the serialized model to the stream using the `ModelWriter.Save` method.

Certain types of streams such as `MemoryStream` may not be compatible with large models over 2GB.

## Decrypt a model from disk

Follow these steps, usually before running the model.

1. Create a `Stream` object for the encrypted model using a cryptography API with your key.
2. Decrypt and deserialize the model using the `ModelLoader.Load` method.

## Example using AES Encryption

The following code samples demonstrate how to serialize a runtime model to disk, encrypt it using the Advanced Encryption Standard (AES), and how to decrypt the encrypted model for inference.

This code sample uses [AES encryption in C#](https://learn.microsoft.com/en-us/dotnet/standard/security/encrypting-data) to encrypt a model.

```
using System.Security.Cryptography;

void SaveModelAesEncrypted(Model model, string path, byte[] key)
{
    // Create a `FileStream` with the path of the encrypted asset
    using FileStream fileStream = new FileStream(path, FileMode.OpenOrCreate);
    using Aes aes = Aes.Create();
    aes.Key = key;

    byte[] iv = aes.IV;

    // Write the initialization vector to the file
    fileStream.Write(iv, 0, iv.Length);

    // Create a `CryptoStream` that writes to the `FileStream` 
    using CryptoStream cryptoStream = new CryptoStream(fileStream, aes.CreateEncryptor(), CryptoStreamMode.Write);

    // Serialize the model to the `CryptoStream`
    ModelWriter.Save(cryptoStream, model);
}
```

This code sample uses [AES decryption in C#](https://learn.microsoft.com/en-us/dotnet/standard/security/decrypting-data) to decrypt a model.

```
using System.Security.Cryptography;

Model LoadModelAesEncrypted(string path, byte[] key)
{
    // Create a `FileStream` with the path of the encrypted asset
    using var fileStream = new FileStream(path, FileMode.Open);
    using Aes aes = Aes.Create();

    // Read the initialization vector from the file
    byte[] iv = new byte[aes.IV.Length];
    int numBytesToRead = aes.IV.Length;
    int numBytesRead = 0;
    while (numBytesToRead > 0)
    {
        int n = fileStream.Read(iv, numBytesRead, numBytesToRead);
        if (n == 0) break;

        numBytesRead += n;
        numBytesToRead -= n;
    }

    // Create a `CryptoStream` that reads from the `FileStream`
    using CryptoStream cryptoStream = new CryptoStream(fileStream, aes.CreateDecryptor(key, iv), CryptoStreamMode.Read);

    // Deserialize the model from the `CryptoStream`
    return ModelLoader.Load(cryptoStream);
}
```

Refer to the `Encrypt a model` example in the [sample scripts](package-samples.md) for an example.
