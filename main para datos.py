import torch
from modelo import ModeloIAImaginativa
from entrenamiento import entrenar_modelo
from evaluacion import evaluar_modelo, calcular_bleu, calcular_longitud_promedio, evaluar_creatividad
from datos import cargar_datos
from config import config, EPOCHS, BATCH_SIZE, LEARNING_RATE
from prueba import probar_modelo
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    modelo = ModeloIAImaginativa(config)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=LEARNING_RATE)
    
    datos_entrenamiento = cargar_datos("ruta/a/datos_entrenamiento.txt", BATCH_SIZE)
    datos_evaluacion = cargar_datos("ruta/a/datos_evaluacion.txt", BATCH_SIZE)
    
    perdidas = []
    bleu_scores = []
    creatividad_scores = []
    
    for epoch in range(EPOCHS):
        entrenar_modelo(modelo, optimizador, datos_entrenamiento, 1, device)
        
        if epoch % 5 == 0:  # Realizar pruebas cada 5 épocas
            texto_generado = modelo.generar_texto("En un mundo donde la magia y la tecnología coexisten,")
            print(f"Época {epoch}, Texto generado: {texto_generado}")
            
            bleu = calcular_bleu("Referencia de ejemplo", texto_generado)
            longitud = calcular_longitud_promedio(texto_generado)
            creatividad = evaluar_creatividad(texto_generado)
            
            print(f"BLEU: {bleu:.4f}, Longitud promedio: {longitud}, Creatividad: {creatividad:.4f}")
            
            perdidas.append(perdida_evaluacion)
            bleu_scores.append(bleu)
            creatividad_scores.append(creatividad)
        
        perdida_evaluacion = evaluar_modelo(modelo, datos_evaluacion, device)
        print(f"Pérdida de evaluación: {perdida_evaluacion:.4f}")

    visualizar_progreso(perdidas, bleu_scores, creatividad_scores)

    # Cargar datos de prueba
    datos_prueba = cargar_datos("ruta/a/datos_prueba.csv", BATCH_SIZE)
    
    # Probar modelo
    probar_modelo(modelo, datos_prueba)

if __name__ == "__main__":
    main()
