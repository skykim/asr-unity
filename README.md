# asr-unity

A fast, on-device Automatic Speech Recognition (ASR) system for Unity. This project integrates **TEN VAD** for voice activity detection with **Vosk** and **Wav2Vec2** for real-time, high-accuracy speech recognition across multiple platforms.

## Overview

This repository provides a complete ASR solution built for Unity developers. It uses TEN VAD to efficiently detect speech, then processes it locally using your choice of the Vosk or Wav2Vec2 engine for transcription. This on-device approach ensures privacy and low latency without requiring an internet connection.

You can use a wide variety of pre-trained models, including:
* [Vosk Models](https://alphacephei.com/vosk/models) (20+ languages)
* [Wav2Vec2 ONNX Models](https://huggingface.co/darjusul/wav2vec2-ONNX-collection) (German, English, Spanish, French, Italian, and more)

## Features

- ✅ **Multi-platform Support**: Works on Windows, macOS, and Android.
- ✅ **On-Device Processing**: No internet connection or server-side processing required.
- ✅ **Pre-built Libraries**: Includes ready-to-use binaries, so you can get started immediately.
- ✅ **Multilingual Support**: Access a wide variety of pre-trained language models.

## Requirements

-   **Unity**: `6000.0.50f1`
-   **Inference Engine**: `2.2.1`

## Architecture

### 1. Voice Activity Detection (VAD)

This project uses a Unity port of [TEN VAD](https://github.com/TEN-framework/ten-vad). TEN VAD is part of the TEN open-source ecosystem for conversational AI and is used here to detect when a user is speaking, which optimizes performance by only running the STT engine when necessary.

### 2. Speech-to-Text (STT) Engines

You can choose between two powerful STT engines:

* **[Vosk](https://alphacephei.com/vosk)**: It offers a great balance of performance and size. Its small models (~50MB) are perfect for mobile devices and desktops, while its larger models provide server-grade accuracy. (CPU)

* **[Wav2Vec2](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2)**: An STT model that works directly with raw audio waveforms. It was introduced in the paper [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://huggingface.co/papers/2006.11477) and is known for its high accuracy. (CPU/GPU)

## Getting Started

### 1. Project Setup

1.  Clone or download this repository.
2.  **For Vosk**: Download a [Vosk model](https://alphacephei.com/vosk/models) (`.zip`). Unzip it and place the entire model folder inside the `/Assets/StreamingAssets/` directory.
3.  **For Wav2Vec2**: Download your chosen Wav2Vec2 model files (`.onnx` and `vocab.json`). Place them inside the `/Assets/Models/Wav2Vec2/` directory.

### 2. Run the Demo

1.  In the Unity Editor, open the `/Assets/Scenes/ASRScene.unity` scene.
2.  Press the **Play** button to see the speech recognition demo in action.

### 3. Configuration

You can easily switch between STT engines in the demo scene:

1.  Select the `ASRManager` GameObject in the Hierarchy.
2.  In the Inspector, find the **ASR Runner** component.
3.  Drag either the **Vosk** or **Wav2Vec2** GameObject into the `Asr Runner Component` field to select your desired engine.
    * **To configure Vosk**: Select the `Vosk` object and set the `Vosk Model Folder Name` to match the folder you added in `StreamingAssets`.
    * **To configure Wav2Vec2**: Select the `Wav2Vec2` object and assign your `.onnx` file to `Model Asset` and your `vocab.json` to `Vocab File`.

## Demo

See `asr-unity` in action! Check out our demo video below.

[![ASR Unity Demo](https://img.youtube.com/vi/RU9qRlYNVm8/0.jpg)](https://www.youtube.com/watch?v=RU9qRlYNVm8)

## Links

-   [TEN VAD](https://github.com/TEN-framework/ten-vad)
-   [Vosk](https://alphacephei.com/vosk)
-   [Wav2Vec2](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2)

## License

This project is subject to the licenses of its core components. Please refer to the original repositories for TEN VAD, Vosk, and Wav2Vec2 for detailed license information.
