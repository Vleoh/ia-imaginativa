import torch
from modelo import ModeloIAImaginativa
from gan_imaginativo import GANImaginativo
from evolucion_conceptos import EvolucionConceptos

def main():
    try:
        # Inicializar modelos
        modelo = ModeloIAImaginativa()
        gan = GANImaginativo(latent_dim=100, output_dim=768)
        evolucion = EvolucionConceptos(población_size=50, gen_size=100)
        
        # Configurar dispositivo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        modelo.to(device)
        
        # Simular la generación de una idea del GAN
        idea_gan = gan.generar_idea().detach()
        
        # Simular la evolución de conceptos
        evolucion.evolucionar(generaciones=50)
        mejor_concepto = torch.tensor(evolucion.población[0].genes, dtype=torch.float32)
        
        # Preparar imagen_features
        idea_gan = idea_gan.view(1, -1)
        mejor_concepto = mejor_concepto.view(1, -1)
        imagen_features = torch.cat([idea_gan, mejor_concepto], dim=1)
        imagen_features = imagen_features.to(device)
        
        # Definir prompts para generar texto
        prompts = [
            "En un mundo donde la realidad se mezcla con la imaginación,",
            "El último invento revolucionario fue",
            "En el año 3000, la humanidad descubrió que"
        ]
        
        # Generar texto para cada prompt
        for prompt in prompts:
            try:
                texto_generado = modelo.generar_texto(prompt, imagen_features)
                print("\nPrompt:", prompt)
                print("Texto generado:", texto_generado)
            except Exception as e:
                print(f"Error al generar texto para '{prompt}': {str(e)}")
                print(f"Tipo de error: {type(e)}")
                print(f"Detalles adicionales: {e.args}")
        
    except Exception as e:
        print(f"Error en la ejecución principal: {str(e)}")
        print(f"Tipo de error: {type(e)}")
        print(f"Detalles adicionales: {e.args}")

if __name__ == "__main__":
    main()
