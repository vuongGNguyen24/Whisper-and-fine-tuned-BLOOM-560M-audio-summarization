import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

asr_pipeline = None  # Singleton

def load_whisper_model():
    global asr_pipeline
    if asr_pipeline is not None:
        return asr_pipeline  # already initialized

    print("Loading Whisper model...")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30,
        batch_size=4,
        return_timestamps=False,
    )

    print("Whisper model loaded.")
    return asr_pipeline
