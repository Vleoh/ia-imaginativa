import torch
from modelo import ModeloIAImaginativa

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    modelo = ModeloIAImaginativa()
    modelo.model.to(device)
    
    print("Modelo inicializado correctamente.")
    
    prompts = [
        "En un mundo donde la magia y la tecnología coexisten,",
        "El último invento revolucionario fue",
        "En el año 3000, la humanidad descubrió que"
    ]
    
    for prompt in prompts:
        texto_generado = modelo.generar_texto(prompt, max_length=150)
        print(f"\nPrompt: {prompt}")
        print(f"Texto generado: {texto_generado}")

if __name__ == "__main__":
    main()
