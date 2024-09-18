import torch
from modelo import ModeloIAImaginativa
from config import config

def probar_modelo():
    modelo = ModeloIAImaginativa(config)
    modelo.load_state_dict(torch.load('ruta/al/modelo_guardado.pth'))
    modelo.eval()

    prompt = "En un futuro lejano, la humanidad descubrió que"
    texto_generado = modelo.generar_texto(prompt)
    print(f"Prompt: {prompt}")
    print(f"Texto generado: {texto_generado}")

    # Aquí puedes añadir más pruebas según sea necesario

if __name__ == "__main__":
    probar_modelo()
