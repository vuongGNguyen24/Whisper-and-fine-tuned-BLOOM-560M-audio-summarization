# 🎙️ Speech Summarizer App

A demo application that converts **speech to text** and **summarizes the content**, built using state-of-the-art models like **Whisper** and a **LoRA fine-tuned BLOOM-560M** for Vietnamese text summarization.

## 🚀 Technologies Used

- 🤖 **Hugging Face Transformers**
  - [Whisper](https://huggingface.co/openai/whisper-large-v3): Automatic Speech Recognition (ASR)
  - [BLOOM-560M](https://huggingface.co/bigscience/bloom-560m): Summarization model
- 🧩 **LoRA (Low-Rank Adaptation)**: Efficient parameter-efficient fine-tuning
- 🧑‍💻 **Streamlit**: Frontend for user interaction
- ⚡ **FastAPI**: Backend API to serve models
- 📚 **Dataset**: [OpenHust/vietnamese-summarization](https://huggingface.co/datasets/OpenHust/vietnamese-summarization)

## How to use

### Install the lastest version of these packages

- Pytorch
- Transformer
- Peft
- Streamlit
- FastAPI

### Setup Backend server (FastAPI)

Run this command:

```bash
uvicorn api.main:app --reload
```

### Setup Frontend app (Streamlit)

Open another terminal, and run this command:

```bash
streamlit run app/app.py
```

### Usage

Upload an audio file (.mp3, .wav, .m4a)

Click summarize audio file

## Evaluation Results

| Model            | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR    |
| ---------------- | ------- | ------- | ------- | --------- |
| Original BLOOM   | 38.01   | 14.42   | 24.97   | 21.58     |
| LoRA-Tuned BLOOM | 33.73   | 14.03   | 24.77   | **24.63** |

Although ROUGE scores slightly decreased, the fine-tuned model improved semantic quality (as reflected by a higher METEOR score) and performs better on subjective evaluation.

## Notes

For faster performance, use smaller Whisper models (like whisper-small) if your GPU is limited (e.g. GTX 1660 Ti).

All components are modular and can be extended for multilingual or cross-domain summarization.
