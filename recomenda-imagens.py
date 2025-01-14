#Sistema de Recomendação por Similaridade de Imagens
#Instalação das Dependências

!pip install tensorflow
!pip install numpy
!pip install scikit-learn
!pip install pillow
!pip install tensorflow-hub
  
#Importação das Bibliotecas
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
  
#Carregamento do Modelo Base
class ImageFeatureExtractor:
    def __init__(self):
        # Carrega o modelo MobileNet V2 pré-treinado
        self.model = tf.keras.Sequential([
            hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
                          output_shape=[1280],
                          trainable=False)
        ])

    def extract_features(self, img_path):
        # Carrega e pré-processa a imagem
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Extrai as características
        features = self.model.predict(x)
        return features
  
#Sistema de Recomendação
class ImageRecommender:
    def __init__(self):
        self.feature_extractor = ImageFeatureExtractor()
        self.features_database = {}
        self.knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        
    def add_to_database(self, image_path, product_id):
        # Extrai e armazena características
        features = self.feature_extractor.extract_features(image_path)
        self.features_database[product_id] = features
        
    def build_index(self):
        # Constrói o índice KNN
        features_matrix = np.array(list(self.features_database.values())).reshape(
            len(self.features_database), -1)
        self.knn.fit(features_matrix)
        
    def get_recommendations(self, query_image_path, n_recommendations=5):
        # Extrai características da imagem de consulta
        query_features = self.feature_extractor.extract_features(query_image_path)
        
        # Encontra os vizinhos mais próximos
        distances, indices = self.knn.kneighbors(
            query_features.reshape(1, -1),
            n_neighbors=n_recommendations
        )
        
        # Recupera os IDs dos produtos recomendados
        product_ids = list(self.features_database.keys())
        recommendations = [product_ids[idx] for idx in indices[0]]
        
        return recommendations
  
#Exemplo de Uso
# Inicializa o sistema de recomendação
recommender = ImageRecommender()

# Adiciona imagens ao banco de dados
image_directory = "produtos/"
for filename in os.listdir(image_directory):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(image_directory, filename)
        product_id = filename.split(".")[0]
        recommender.add_to_database(image_path, product_id)

# Constrói o índice
recommender.build_index()

# Obtém recomendações para uma imagem de consulta
query_image = "consulta.jpg"
recommendations = recommender.get_recommendations(query_image)
print(f"Produtos recomendados: {recommendations}")
  
6. Processamento em Lote
def process_batch(image_paths, batch_size=32):
    feature_extractor = ImageFeatureExtractor()
    all_features = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        
        for path in batch_paths:
            img = image.load_img(path, target_size=(224, 224))
            x = image.img_to_array(img)
            batch_images.append(x)
            
        batch_images = np.array(batch_images)
        batch_images = preprocess_input(batch_images)
        
        batch_features = feature_extractor.model.predict(batch_images)
        all_features.extend(batch_features)
    
    return np.array(all_features)
  
#Observações Importantes:
#O sistema utiliza MobileNet V2 como extrator de características base
#As características são comparadas usando similaridade por cosseno
#O sistema pode ser facilmente expandido para incluir mais metadados dos produtos
#Para melhor desempenho, recomenda-se usar GPU para processamento em lote
#O sistema pode ser integrado com uma interface web para visualização das recomendações
#Avaliação do Sistema
def evaluate_recommendations(recommender, test_cases):
    results = []
    
    for test_image, expected_category in test_cases:
        recommendations = recommender.get_recommendations(test_image)
        
        # Calcula precisão das recomendações
        correct = sum(1 for rec in recommendations 
                     if rec.split('_')[0] == expected_category)
        precision = correct / len(recommendations)
        
        results.append({
            'test_image': test_image,
            'precision': precision,
            'recommendations': recommendations
        })
    
    return results
  
