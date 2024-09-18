import torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

def evaluar_modelo(modelo, datos, device):
    modelo.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(datos, desc="Evaluación"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = modelo(**batch)
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / len(datos)
    return avg_loss

def calcular_novedad(salida, referencia):
    # Implementar métrica de novedad
    pass

def calcular_coherencia(salida):
    # Implementar métrica de coherencia
    pass

def calcular_bleu(referencia, candidato):
    referencia_tokens = word_tokenize(referencia)
    candidato_tokens = word_tokenize(candidato)
    return sentence_bleu([referencia_tokens], candidato_tokens)

def calcular_longitud_promedio(texto):
    palabras = texto.split()
    return len(palabras)

def evaluar_creatividad(texto):
    # Implementa una métrica simple de creatividad
    # Por ejemplo, contar palabras únicas o poco comunes
    palabras_unicas = set(texto.lower().split())
    return len(palabras_unicas) / len(texto.split())
