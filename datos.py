import csv
import json
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

class DatosImaginativos(Dataset):
    def __init__(self, archivo_csv):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.datos = []
        
        with open(archivo_csv, 'r', encoding='utf-8') as f:
            lector = csv.DictReader(f)
            for fila in lector:
                self.datos.append({
                    'id': fila['id'],
                    'texto': fila['texto'],
                    'metadatos': json.loads(fila['metadatos'])
                })
    
    def __len__(self):
        return len(self.datos)
    
    def __getitem__(self, idx):
        item = self.datos[idx]
        encodings = self.tokenizer(item['texto'], truncation=True, padding='max_length', max_length=1024, return_tensors='pt')
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'metadatos': item['metadatos']
        }

def cargar_datos(ruta_archivo, batch_size):
    dataset = DatosImaginativos(ruta_archivo)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
