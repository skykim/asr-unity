using UnityEngine;
using UnityEngine.UI;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading.Tasks;

public class ASRManager : MonoBehaviour
{
    private enum State
    {
        Initializing, Idle, Speaking, Error
    }
    private State _currentState = State.Initializing;

    [Header("VAD Settings")]
    [Range(0f, 1f)] public float vadThreshold = 0.5f;
    [SerializeField, Range(0f, 1f)] private float _currentVadProbability;
    [Range(1, 50)] public int preBufferFrames = 20;
    [Range(1, 50)] public int postBufferFrames = 20;
    public float maxRecordingSeconds = 10f;

    [Header("Connections")]
    public MonoBehaviour asrRunnerComponent;
    private IASRRunner _activeRunner;

    [Header("UI Connection")]
    public Text fpsText;
    public Text statusText;
    public Text resultText;
    public Text partialResultText;

    private TenVADRunner _vad;
    private string _selectedMicrophone;
    private AudioClip _microphoneClip;
    private int _lastPosition = 0;
    private int _consecutiveSilenceFrames = 0;
    private float _currentRecordingTime = 0f;

    private const int HOP_SIZE = 256;
    private const int TARGET_SAMPLE_RATE = 16000;

    private CircularBuffer _microphoneCircularBuffer;
    private CircularBuffer _preSpeechCircularBuffer;
    private float[] _reusableReadBuffer;
    private float[] _reusableProcessChunk;
    private short[] _reusableShortChunk;

    private readonly ConcurrentQueue<string> _partialResultsQueue = new ConcurrentQueue<string>();
    private readonly ConcurrentQueue<string> _finalResultsQueue = new ConcurrentQueue<string>();

    private float deltaTime = 0.0f;
    private const int MAX_CHUNKS_PER_FRAME = 5;

    private async void Start()
    {
        SetState(State.Initializing);
        try
        {
            InitializeBuffers();
            await InitializeASRRunner();
            _vad = new TenVADRunner((UIntPtr)HOP_SIZE, vadThreshold);
            InitializeMicrophone();
            Debug.Log($"[ASRManager] Initialized successfully with '{asrRunnerComponent.GetType().Name}'.");
            SetState(State.Idle);
        }
        catch (Exception e)
        {
            Debug.LogError($"[ASRManager] Initialization failed: {e.Message}\n{e.StackTrace}");
            SetState(State.Error);
        }
    }

    private void Update()
    {
        UpdateFPS();

        if (_currentState == State.Idle || _currentState == State.Speaking)
        {
            ReadMicrophoneData();
            ProcessAudioChunks();
            CheckMicrophoneStatus();
        }

        ProcessResultQueues();
    }

    private void OnDestroy()
    {
        if (_activeRunner != null)
        {
            _activeRunner.OnPartialResult -= OnPartialResultReceived;
            _activeRunner.OnFinalResult -= OnFinalResultReceived;
        }

        if (_microphoneClip != null && !string.IsNullOrEmpty(_selectedMicrophone) && Microphone.IsRecording(_selectedMicrophone))
        {
            Microphone.End(_selectedMicrophone);
        }
        _vad?.Dispose();
        _activeRunner?.Dispose();
    }

    private void InitializeBuffers()
    {
        _microphoneCircularBuffer = new CircularBuffer(TARGET_SAMPLE_RATE * 2);
        _preSpeechCircularBuffer = new CircularBuffer(HOP_SIZE * preBufferFrames);
        _reusableReadBuffer = new float[TARGET_SAMPLE_RATE];
        _reusableProcessChunk = new float[HOP_SIZE];
        _reusableShortChunk = new short[HOP_SIZE];
    }

    private async Task InitializeASRRunner()
    {
        if (asrRunnerComponent == null)
        {
            throw new ArgumentNullException("ASR Runner Component is not assigned in the Inspector.");
        }
        _activeRunner = asrRunnerComponent as IASRRunner;
        if (_activeRunner == null)
        {
            throw new InvalidCastException($"The component '{asrRunnerComponent.GetType().Name}' must implement IASRRunner.");
        }
        _activeRunner.OnPartialResult += OnPartialResultReceived;
        _activeRunner.OnFinalResult += OnFinalResultReceived;
        await _activeRunner.Initialize();
    }

    private void InitializeMicrophone()
    {
        if (Microphone.devices.Length == 0) throw new InvalidOperationException("No microphone found.");
        _selectedMicrophone = Microphone.devices[0];
        _microphoneClip = Microphone.Start(_selectedMicrophone, true, (int)maxRecordingSeconds + 1, TARGET_SAMPLE_RATE);
        _lastPosition = 0;
        Debug.Log($"[ASRManager] Started recording from microphone '{_selectedMicrophone}'.");
    }

    private void ReadMicrophoneData()
    {
        if (_microphoneClip == null) return;

        int currentPosition = Microphone.GetPosition(_selectedMicrophone);
        if (currentPosition == _lastPosition) return;

        int sampleCount = (currentPosition > _lastPosition)
            ? (currentPosition - _lastPosition)
            : (_microphoneClip.samples - _lastPosition + currentPosition);

        if (sampleCount > 0)
        {
            int readLength = Mathf.Min(sampleCount, _reusableReadBuffer.Length);
            _microphoneClip.GetData(_reusableReadBuffer, _lastPosition);
            _microphoneCircularBuffer.Write(_reusableReadBuffer, readLength);
        }
        _lastPosition = currentPosition;
    }

    private void ProcessAudioChunks()
    {
        int chunksProcessed = 0;
        while (_microphoneCircularBuffer.Count >= HOP_SIZE && chunksProcessed < MAX_CHUNKS_PER_FRAME)
        {
            _microphoneCircularBuffer.Read(_reusableProcessChunk, HOP_SIZE);

            for (int i = 0; i < HOP_SIZE; i++) _reusableShortChunk[i] = (short)(_reusableProcessChunk[i] * 32767.0f);
            _vad.Process(_reusableShortChunk, out _currentVadProbability, out int flag);
            bool voiceDetected = flag == 1;

            switch (_currentState)
            {
                case State.Idle:
                    _preSpeechCircularBuffer.Write(_reusableProcessChunk, HOP_SIZE);
                    if (voiceDetected)
                    {
                        StartSpeech();
                    }
                    break;

                case State.Speaking:
                    _activeRunner.ProcessAudioChunk(_reusableProcessChunk);
                    _currentRecordingTime += (float)HOP_SIZE / TARGET_SAMPLE_RATE;
                    if (voiceDetected)
                    {
                        _consecutiveSilenceFrames = 0;
                    }
                    else
                    {
                        _consecutiveSilenceFrames++;
                        if (_consecutiveSilenceFrames >= postBufferFrames)
                        {
                            EndSpeech();
                        }
                    }
                    if (_currentRecordingTime >= maxRecordingSeconds)
                    {
                        Debug.Log($"Max recording time ({maxRecordingSeconds}s) reached. Ending speech segment.");
                        EndSpeech();
                    }
                    break;
            }
            chunksProcessed++;
        }
    }

    private void StartSpeech()
    {
        SetState(State.Speaking);
        _currentRecordingTime = 0f;
        _consecutiveSilenceFrames = 0;

        _activeRunner.StartSpeechSegment();

        int preSpeechDataLength = _preSpeechCircularBuffer.Count;
        if (preSpeechDataLength > 0)
        {
            float[] preSpeechData = new float[preSpeechDataLength];
            _preSpeechCircularBuffer.Read(preSpeechData, preSpeechDataLength);
            
            int offset = 0;
            while(offset < preSpeechData.Length)
            {
                int length = Mathf.Min(HOP_SIZE, preSpeechData.Length - offset);
                Array.Copy(preSpeechData, offset, _reusableProcessChunk, 0, length);
                if(length < HOP_SIZE)
                {
                    var tempChunk = new float[length];
                    Array.Copy(_reusableProcessChunk, tempChunk, length);
                    _activeRunner.ProcessAudioChunk(tempChunk);
                } else {
                    _activeRunner.ProcessAudioChunk(_reusableProcessChunk);
                }
                offset += length;
            }
        }
        _activeRunner.ProcessAudioChunk(_reusableProcessChunk);
    }

    private void EndSpeech()
    {
        if (_currentState != State.Speaking) return;

        _activeRunner.EndSpeechSegment();
        _preSpeechCircularBuffer.Clear();
        SetState(State.Idle);
    }

    private void OnPartialResultReceived(string partial) => _partialResultsQueue.Enqueue(partial);
    private void OnFinalResultReceived(string final)
    {
        if (!string.IsNullOrWhiteSpace(final))
        {
             _finalResultsQueue.Enqueue(final);
        }
    }
    
    private void ProcessResultQueues()
    {
        if (_partialResultsQueue.TryDequeue(out string partialResult))
        {
            if (partialResultText != null) partialResultText.text = partialResult;
        }
        
        if (_finalResultsQueue.TryDequeue(out string finalResult))
        {
            Debug.Log($"[Final Result]: {finalResult}");
            if (resultText != null) resultText.text += finalResult + " ";
            if (partialResultText != null) partialResultText.text = "";
        }
    }

    private void SetState(State newState)
    {
        if (_currentState == newState) return;
        _currentState = newState;

        if (statusText == null) return;
        switch (newState)
        {
            case State.Initializing:
                statusText.text = "Initializing...";
                statusText.color = Color.yellow;
                break;
            case State.Idle:
                statusText.text = "Listening...";
                statusText.color = Color.white;
                break;
            case State.Speaking:
                statusText.text = "Speaking";
                statusText.color = Color.green;
                break;
            case State.Error:
                statusText.text = "ERROR";
                statusText.color = Color.red;
                break;
        }
    }

    private void UpdateFPS()
    {
        deltaTime += (Time.unscaledDeltaTime - deltaTime) * 0.1f;
        if (fpsText != null) fpsText.text = "FPS: " + Mathf.Ceil(1.0f / deltaTime);
    }

    private void CheckMicrophoneStatus()
    {
        if (!string.IsNullOrEmpty(_selectedMicrophone) && !Microphone.IsRecording(_selectedMicrophone))
        {
            Debug.LogWarning($"[ASRManager] Microphone '{_selectedMicrophone}' stopped recording. Attempting to restart.");
            InitializeMicrophone();
        }
    }

    private class CircularBuffer
    {
        private readonly float[] _buffer;
        private int _head;
        private int _tail;
        private readonly int _capacity;
        public int Count { get; private set; }

        public CircularBuffer(int capacity)
        {
            _capacity = capacity;
            _buffer = new float[capacity];
            Clear();
        }

        public void Write(float[] data, int length)
        {
            for (int i = 0; i < length; i++)
            {
                _buffer[_tail] = data[i];
                _tail = (_tail + 1) % _capacity;
            }
            Count = Mathf.Min(Count + length, _capacity);
        }

        public void Read(float[] destination, int length)
        {
            if (length > Count) throw new InvalidOperationException("Not enough data to read.");
            for (int i = 0; i < length; i++)
            {
                destination[i] = _buffer[_head];
                _head = (_head + 1) % _capacity;
            }
            Count -= length;
        }
        
        public void Clear()
        {
            _head = 0;
            _tail = 0;
            Count = 0;
        }
    }
}

