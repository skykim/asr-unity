using UnityEngine;
using Unity.InferenceEngine;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

public class Wav2Vec2Runner : MonoBehaviour, IASRRunner
{
    [Header("Wav2Vec2 Settings")]
    public ModelAsset modelAsset;
    public TextAsset vocabFile;

    public event Action<string> OnPartialResult;
    public event Action<string> OnFinalResult;

    private readonly List<float> _speechAudioBuffer = new List<float>();
    private Model _model;
    private Worker _worker;
    private bool _isDisposed = false;
    private Dictionary<int, string> _idToToken;
    private int _padTokenId = -1;
    private int _unkTokenId = -1;
    
    private const int SAMPLE_RATE = 16000;
    private const string WORD_DELIMITER_TOKEN = "|";
    private const string REPLACE_WORD_DELIMITER_CHAR = " ";
    
    public async Task Initialize()
    {
        var tokenToId = JsonConvert.DeserializeObject<Dictionary<string, int>>(vocabFile.text);
        _idToToken = tokenToId.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
        
        if (tokenToId.TryGetValue("<pad>", out int padId1)) _padTokenId = padId1;
        else if (tokenToId.TryGetValue("[PAD]", out int padId2)) _padTokenId = padId2;
        else Debug.LogWarning("Pad token ('<pad>' or '[PAD]') not found.");

        if (tokenToId.TryGetValue("<unk>", out int unkId1)) _unkTokenId = unkId1;
        else if (tokenToId.TryGetValue("[UNK]", out int unkId2)) _unkTokenId = unkId2;
        else Debug.LogWarning("Unknown token ('<unk>' or '[UNK]') not found.");
        
        Model baseModel = ModelLoader.Load(modelAsset);
        FunctionalGraph graph = new FunctionalGraph();
        FunctionalTensor input = graph.AddInput(baseModel, 0);

        var mean = Functional.ReduceMean(input, dim: new[] { 1 }, keepdim: true);
        var centered = Functional.Sub(input, mean);
        var variance = Functional.ReduceMean(Functional.Pow(centered, 2f), dim: new[] { 1 }, keepdim: true);
         var stdDev = Functional.Sqrt(Functional.Add(variance, Functional.Constant(1e-5f)));
        var normalizedInput = Functional.Div(centered, stdDev);

        FunctionalTensor output = Functional.Forward(baseModel, normalizedInput)[0];
        FunctionalTensor reshapedOutput = Functional.Squeeze(output);
        var argMax = Functional.ArgMax(reshapedOutput, dim: 1);
        _model = graph.Compile(argMax);
        _worker = new Worker(_model, BackendType.GPUCompute);

        Debug.Log("Warming up the Wav2Vec2 model...");
        float[] dummyAudio = new float[1024];
        await ProcessAudioAsync(dummyAudio);
        Debug.Log("Wav2Vec2Runner initialized successfully.");
    }

    public void StartSpeechSegment()
    {
        _speechAudioBuffer.Clear();
    }

    public void ProcessAudioChunk(float[] audioChunk)
    {
        _speechAudioBuffer.AddRange(audioChunk);
    }

    public async void EndSpeechSegment()
    {
        try
        {
            float[] audioToProcess = _speechAudioBuffer.ToArray();
            _speechAudioBuffer.Clear();

            float[] paddedAudio = new float[audioToProcess.Length];
            Array.Copy(audioToProcess, 0, paddedAudio, 0, audioToProcess.Length);
            
            string result = await ProcessAudioAsync(paddedAudio);

            if (!string.IsNullOrEmpty(result))
            {
                OnFinalResult?.Invoke(result);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[Wav2Vec2Runner] An error occurred during speech processing: {e.Message}\n{e.StackTrace}");
        }
    }
    
    private async Task<string> ProcessAudioAsync(float[] audioArray)
    {
        if (_isDisposed) throw new ObjectDisposedException(nameof(Wav2Vec2Runner));
        if (audioArray == null || audioArray.Length == 0) return string.Empty;

        using Tensor<float> inputTensor = new(new TensorShape(1, audioArray.Length), audioArray);
        _worker.SetInput(_model.inputs[0].name, inputTensor);
        _worker.Schedule();
        var output = _worker.PeekOutput(_model.outputs[0].name);
        using var outputTensor = await output.ReadbackAndCloneAsync() as Tensor<int>;
        var tokenIds = outputTensor.DownloadToArray();
        
        return Decode(tokenIds);
    }
    
    private string Decode(IEnumerable<int> tokenIds)
    {
        if (tokenIds == null || !tokenIds.Any()) return "";
        
        List<int> groupedIds = new List<int>();
        int previousId = -1;
        foreach (var id in tokenIds)
        {
            if (id != previousId) groupedIds.Add(id);
            previousId = id;
        }

        var nonBlankIds = groupedIds.Where(id => id != _padTokenId);
        
        var builder = new StringBuilder();
        foreach (var id in nonBlankIds)
        {
            if (id == _unkTokenId) continue;

            if (_idToToken.TryGetValue(id, out string token))
            {
                builder.Append(token == WORD_DELIMITER_TOKEN ? REPLACE_WORD_DELIMITER_CHAR : token);
            }
        }
        return builder.ToString().Trim();
    }

    public void Dispose()
    {
        if (_isDisposed) return;
        _worker?.Dispose();
        _isDisposed = true;
    }
}