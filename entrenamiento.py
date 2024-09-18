import torch
from tqdm import tqdm
from utils import salto_cuantico

def entrenar_modelo(modelo, optimizador, datos, epochs, device):
    modelo.to(device)
    for epoch in tqdm(range(epochs), desc="Épocas"):
        modelo.train()
        for batch in tqdm(datos, desc="Lotes", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizador.zero_grad()
            outputs = modelo(**batch)
            loss = outputs.loss
            loss.backward()
            optimizador.step()
            
            if epoch % 10 == 0:
                with torch.no_grad():
                    espacio_latente = modelo.obtener_espacio_latente(batch['input_ids'])
                    nuevo_espacio = salto_cuantico(espacio_latente)
                    modelo.actualizar_espacio_latente(nuevo_espacio)
        
        print(f"Época {epoch+1}/{epochs}, Pérdida: {loss.item():.4f}")
