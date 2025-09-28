using UnityEngine;
using UnityEngine.Networking;
using System;
using System.IO;
using System.Threading.Tasks;
using Vosk;
using System.Collections.Concurrent;
using System.IO.Compression;

public class VoskRunner : MonoBehaviour, IASRRunner
{
    [Header("Vosk Settings")]
    public string voskModelFolderName = "vosk-model-small-ko-0.22";

    public event Action<string> OnPartialResult;
    public event Action<string> OnFinalResult;

    private Model _voskModel;
    private VoskRecognizer _voskRecognizer;

    // Thread-safe queue and reusable buffers for result handling.
    private readonly ConcurrentQueue<VoskResult> _resultQueue = new ConcurrentQueue<VoskResult>();
    private short[] _shortBuffer;
    private byte[] _byteBuffer;


    [Serializable]
    private class VoskResult { public string text; public string partial; }

    public async Task Initialize()
    {
        string modelPath;

#if UNITY_EDITOR
        modelPath = Path.Combine(Application.streamingAssetsPath, voskModelFolderName);
        Debug.Log($"[VoskRunner-Editor] Using direct model path: {modelPath}");

        if (!Directory.Exists(modelPath))
        {
            throw new DirectoryNotFoundException($"[VoskRunner-Editor] Model folder not found at {modelPath}. Please place the unzipped model folder in StreamingAssets.");
        }

#else
        modelPath = Path.Combine(Application.persistentDataPath, voskModelFolderName);

        if (!Directory.Exists(modelPath))
        {
            Debug.Log($"[VoskRunner-Build] Model not found at {modelPath}. Starting copy and unzip process...");

            string zipFileName = voskModelFolderName + ".zip";
            string zipSourcePath = Path.Combine(Application.streamingAssetsPath, zipFileName);
            string zipDestPath = Path.Combine(Application.persistentDataPath, zipFileName);

            // For Android, StreamingAssets are in a compressed JAR file, so we need UnityWebRequest to access them.
            // For other platforms, it's a direct file path.
            #if UNITY_ANDROID
                Debug.Log($"[VoskRunner-Build] Copying model for Android from: {zipSourcePath}");
                using (UnityWebRequest www = UnityWebRequest.Get(zipSourcePath))
                {
                    var asyncOp = www.SendWebRequest();
                    while (!asyncOp.isDone)
                    {
                        await Task.Yield();
                    }

                    if (www.result != UnityWebRequest.Result.Success)
                    {
                        throw new Exception($"[VoskRunner-Build] Failed to load model zip from StreamingAssets: {www.error}. Path: {zipSourcePath}");
                    }

                    await File.WriteAllBytesAsync(zipDestPath, www.downloadHandler.data);
                    Debug.Log($"[VoskRunner-Build] Zip file copied to {zipDestPath}.");
                }
            #else
                Debug.Log($"[VoskRunner-Build] Copying model for Standalone from: {zipSourcePath}");
                if (!File.Exists(zipSourcePath))
                {
                     throw new FileNotFoundException($"[VoskRunner-Build] Model zip file not found at {zipSourcePath}.");
                }
                File.Copy(zipSourcePath, zipDestPath, true);
                Debug.Log($"[VoskRunner-Build] Zip file copied to {zipDestPath}.");
            #endif

            try
            {
                await Task.Run(() => ZipFile.ExtractToDirectory(zipDestPath, Application.persistentDataPath));
                Debug.Log($"[VoskRunner-Build] Model successfully unzipped to {modelPath}.");
            }
            catch (Exception e)
            {
                throw new Exception($"[VoskRunner-Build] Error unzipping Vosk model: {e.Message}");
            }
            finally
            {
                if (File.Exists(zipDestPath))
                {
                    File.Delete(zipDestPath);
                    Debug.Log($"[VoskRunner-Build] Deleted temporary zip file: {zipDestPath}");
                }
            }
        }
        else
        {
            Debug.Log($"[VoskRunner-Build] Model already exists at {modelPath}.");
        }
#endif
        Vosk.Vosk.SetLogLevel(-1); // Set to -1 to disable logs
        _voskModel = new Model(modelPath);
        Debug.Log("VoskRunner initialized successfully.");
    }

    public void StartSpeechSegment()
    {
        if (_voskModel == null) 
        {
             Debug.LogError("Vosk model is not initialized. Cannot start speech segment.");
             return;
        }
        _voskRecognizer = new VoskRecognizer(_voskModel, 16000.0f);
    }

    public void ProcessAudioChunk(float[] audioChunk)
    {
        if (_voskRecognizer == null) return;

        // Reduce GC pressure by reusing buffers.
        // Reallocate buffer if it's null or smaller than the incoming audio chunk.
        if (_shortBuffer == null || _shortBuffer.Length < audioChunk.Length)
        {
            _shortBuffer = new short[audioChunk.Length];
            _byteBuffer = new byte[audioChunk.Length * 2];
        }

        for (int i = 0; i < audioChunk.Length; i++)
        {
            _shortBuffer[i] = (short)(audioChunk[i] * 32767.0f);
        }

        Buffer.BlockCopy(_shortBuffer, 0, _byteBuffer, 0, audioChunk.Length * 2);

        // Pass data to Vosk. If a result is returned, enqueue it.
        if (_voskRecognizer.AcceptWaveform(_byteBuffer, audioChunk.Length * 2))
        {
            var result = JsonUtility.FromJson<VoskResult>(_voskRecognizer.Result());
            if (!string.IsNullOrEmpty(result.text))
            {
                _resultQueue.Enqueue(result);
            }
        }
        else
        {
            var partialResult = JsonUtility.FromJson<VoskResult>(_voskRecognizer.PartialResult());
            if (!string.IsNullOrEmpty(partialResult.partial))
            {
                _resultQueue.Enqueue(partialResult);
            }
        }
    }

    public void EndSpeechSegment()
    {
        if (_voskRecognizer == null) return;

        var result = JsonUtility.FromJson<VoskResult>(_voskRecognizer.FinalResult());
        if (!string.IsNullOrEmpty(result.text))
        {
            _resultQueue.Enqueue(result);
        }

        _voskRecognizer.Dispose();
        _voskRecognizer = null;
    }

    private void Update()
    {
        // Dequeue results and safely invoke events on the main Unity thread.
        while (_resultQueue.TryDequeue(out var result))
        {
            if (!string.IsNullOrEmpty(result.text)) // This is a final result
            {
                OnFinalResult?.Invoke(result.text);
            }
            else if (!string.IsNullOrEmpty(result.partial)) // This is a partial result
            {
                OnPartialResult?.Invoke(result.partial);
            }
        }
    }

    public void Dispose()
    {
        _voskRecognizer?.Dispose();
        _voskModel?.Dispose();
        _voskRecognizer = null;
        _voskModel = null;
    }

    void OnDestroy()
    {
        Dispose();
    }
}
