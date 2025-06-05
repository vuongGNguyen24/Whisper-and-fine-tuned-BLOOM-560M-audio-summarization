import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load model at module level for efficiency
model_id = "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
model = PeftModel.from_pretrained(base_model, "./model").to("cuda")

def summarize_text(text: str) -> str:
    prompt = f"Tóm tắt văn bản sau:\n{text}\nTóm tắt:"
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

    output = model.generate(
        **input_ids,
        max_new_tokens=150,          # allow summary up to ~150 tokens (adjust for your dataset)
        num_beams=4,             # balanced beam search, better quality
        length_penalty=-2,     
        top_p=0.9,
        temperature=0.5,
        do_sample=True,
        early_stopping=True,     # stop when beams are done
        no_repeat_ngram_size=3,
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    summary = decoded.split("Tóm tắt:")[1].strip()
    return summary
