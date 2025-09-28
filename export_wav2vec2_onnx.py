import argparse
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torchaudio.models.wav2vec2.utils import import_huggingface_model
import torch
import torch.onnx
from pathlib import Path
import shutil
import sys
import torchaudio
import onnxruntime
import numpy as np

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="ONNX model export directory",
    )
    parser.add_argument(
        "-r",
        "--repo",
        type=str,
        default="kresnik/wav2vec2-large-xlsr-korean",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--test-wav",
        type=str,
        help="Path to a WAV file for testing the exported ONNX model",
    )
    return parser.parse_args()


def hf2onnx(hf_model, output_dir):
    temp_dir = Path(".temp")
    temp_dir.mkdir(exist_ok=True)

    # Export the model
    dummy_input = torch.randn(1, 160000)
    torch.onnx.export(
        hf_model,
        dummy_input,
        f"{temp_dir}/model.onnx",
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "output_sequence"},
        },
    )

    shutil.move(f"{temp_dir}/model.onnx", f"{output_dir}/model.onnx")
    shutil.rmtree(temp_dir)
    print(f"ONNX model saved to {output_dir}/model.onnx")


def test_onnx_model(onnx_model_path, wav_file_path, hf_repo):
    """
    Tests speech recognition using an ONNX model and a WAV file.

    Args:
        onnx_model_path (str): Path to the ONNX model file.
        wav_file_path (str): Path to the WAV audio file for testing.
        hf_repo (str): Hugging Face repository name for decoding the model output.
    """
    print("\n--- Running ONNX Model Test ---")

    # 1. Load processor (for audio preprocessing and decoding results)
    try:
        processor = Wav2Vec2Processor.from_pretrained(hf_repo)
    except Exception as e:
        print(f"Error loading processor from {hf_repo}: {e}")
        return

    # 2. Load and preprocess the audio file
    try:
        waveform, sample_rate = torchaudio.load(wav_file_path)
        
        # The model expects a 16kHz sampling rate, so resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        # Use the processor to convert to the model's input format
        input_values = processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values

    except FileNotFoundError:
        print(f"Error: Test WAV file not found at {wav_file_path}")
        return
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return

    # 3. Create ONNX runtime session and run inference
    print(f"Running inference with {onnx_model_path}...")
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = ort_session.get_inputs()[0].name
    
    # ONNX runtime expects numpy arrays as input
    ort_inputs = {input_name: input_values.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # 4. Decode the output result
    # Predict the most likely token ID from the model output (logits)
    predicted_ids = np.argmax(ort_outs[0], axis=-1)
        
    print(predicted_ids)
    transcription = processor.batch_decode(predicted_ids)
    
    print(processor)
    print(processor.batch_decode)
    
    print("-" * 30)
    print(f"Input WAV: {wav_file_path}")
    print(f"Transcription: {transcription[0]}")
    print("--- Test Complete ---\n")


def main(args):
    org = Wav2Vec2ForCTC.from_pretrained(args.repo)
    hf_model = import_huggingface_model(org)
    hf_model.eval()
    
    print(Wav2Vec2ForCTC)
    
    model_out_dir = Path(f"{args.output_dir}/onnx")
    model_out_dir.mkdir(parents=True, exist_ok=True)

    # Export the ONNX model
    hf2onnx(hf_model, model_out_dir)

    # If the --test-wav argument is provided, call the test function
    if args.test_wav:
        onnx_model_path = model_out_dir / "model.onnx"
        if onnx_model_path.exists():
            test_onnx_model(str(onnx_model_path), args.test_wav, args.repo)
        else:
            print(f"Error: ONNX model not found at {onnx_model_path}")


if __name__ == "__main__":
    args = getArgs()
    main(args)