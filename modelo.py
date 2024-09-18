import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

class ModeloIAImaginativa(nn.Module):
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        super().__init__()
        
        # Configuración para cuantización de 4-bit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        # Configurar el pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def generar_texto(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Eliminamos la clase VAE ya que no la usaremos en este enfoque inicial
