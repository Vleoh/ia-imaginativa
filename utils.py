import torch

def salto_cuantico(espacio_latente, temperatura=1.0):
    ruido = torch.randn_like(espacio_latente) * temperatura
    nuevo_espacio = espacio_latente + ruido
    return nuevo_espacio
