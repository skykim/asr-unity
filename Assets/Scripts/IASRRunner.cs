using System;
using System.Threading.Tasks;

public interface IASRRunner : IDisposable
{
    event Action<string> OnPartialResult;
    event Action<string> OnFinalResult;

    Task Initialize();
    void StartSpeechSegment();
    void ProcessAudioChunk(float[] audioChunk);
    void EndSpeechSegment();
}