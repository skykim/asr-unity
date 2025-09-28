using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class TenVADRunner : IDisposable
{
    private const string DllName = "ten_vad";

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int ten_vad_create(out IntPtr handle, UIntPtr hop_size, float threshold);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int ten_vad_process(IntPtr handle, short[] audio_data, UIntPtr audio_data_length, out float out_probability, out int out_flag);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int ten_vad_destroy(IntPtr handle);

    private IntPtr vadHandle = IntPtr.Zero;
    private bool isDisposed = false;

    public TenVADRunner(UIntPtr hopSize, float threshold)
    {
        int result = ten_vad_create(out vadHandle, hopSize, threshold);
        if (result != 0 || vadHandle == IntPtr.Zero)
        {
            throw new Exception($"Failed to create VAD Handle. (Error Code: {result})");
        }
        Debug.Log("VAD Handle created successfully.");
    }

    public int Process(short[] audioData, out float probability, out int flag)
    {
        if (isDisposed)
        {
            throw new ObjectDisposedException(nameof(TenVADRunner), "The VAD instance has already been disposed.");
        }
        if (audioData == null || audioData.Length == 0)
        {
            probability = 0;
            flag = 0;
            Debug.LogWarning("Audio data is null or empty. Skipping processing.");
            return -1;
        }
        return ten_vad_process(vadHandle, audioData, (UIntPtr)audioData.Length, out probability, out flag);
    }
    
    public void Dispose()
    {
        if (vadHandle != IntPtr.Zero)
        {
            //ten_vad_destroy(vadHandle);
            vadHandle = IntPtr.Zero;
            Debug.Log("VAD Handle released safely.");
        }
    }
}