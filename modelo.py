import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer

class AtenciónMultimodal(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.texto_query = nn.Linear(dim, dim)
        self.imagen_key = nn.Linear(dim, dim)
        self.imagen_value = nn.Linear(dim, dim)
    
    def forward(self, texto, imagen):
        q = self.texto_query(texto)
        k = self.imagen_key(imagen)
        v = self.imagen_value(imagen)
        atención = torch.matmul(q, k.transpose(-2, -1)) / (texto.size(-1) ** 0.5)
        return torch.matmul(atención.softmax(dim=-1), v)

class ModeloIAImaginativa(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        super().__init__()
        self.llama = LlamaForCausalLM.from_pretrained(model_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.atención_multimodal = AtenciónMultimodal(self.llama.config.hidden_size)
        self.fusión = nn.Linear(self.llama.config.hidden_size * 2, self.llama.config.hidden_size)
        
    def forward(self, input_ids, attention_mask, imagen_features):
        salida_llama = self.llama(input_ids, attention_mask, output_hidden_states=True)
        último_hidden = salida_llama.hidden_states[-1]
        atención_imagen = self.atención_multimodal(último_hidden, imagen_features)
        fusionado = self.fusión(torch.cat([último_hidden, atención_imagen], dim=-1))
        return self.llama.lm_head(fusionado)

    def generar_texto(self, prompt, imagen_features, max_length=200):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.llama.device) for k, v in inputs.items()}
        imagen_features = imagen_features.to(self.llama.device)
        
        # Ajusta la longitud máxima
        max_length = min(max_length, self.llama.config.max_position_embeddings)
        
        with torch.no_grad():
            generated = self.llama.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
        
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

# Eliminamos la clase VAE ya que no la usaremos en este enfoque inicial
