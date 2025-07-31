from typing import Generator, Iterable, List, TypeVar
import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
import os

V = TypeVar("V")

def create_batches(sequence: Iterable[V], batch_size: int) -> Generator[List[V], None, None]:
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    def __init__(self, player_detection_model, device=None):
        self.player_detection_model = player_detection_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32
        self.features_model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)
        self.is_fitted = False

    def extract_features(self, crops: List[np.ndarray], cache_path=None) -> np.ndarray:
        # Ne pas utiliser de cache pour les prédictions en temps réel
        if cache_path and os.path.exists(cache_path):
            print("Loading cached embeddings...")
            return np.load(cache_path)
        
        if len(crops) == 0:
            return np.array([]).reshape(0, 768)  # DINOv2-base a 768 dimensions
        
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []

        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction', leave=False):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = outputs.pooler_output.cpu().numpy()
                data.append(embeddings)
        
        if data:
            data = np.concatenate(data)
        else:
            data = np.array([]).reshape(0, 768)
        
        if cache_path:
            np.save(cache_path, data)
        return data

    def fit(self, crops: List[np.ndarray]) -> None:
        if len(crops) == 0:
            print("Warning: No crops provided for training")
            return
            
        print(f"Training avec {len(crops)} crops...")
        data = self.extract_features(crops, cache_path="embeddings.npy")
        
        if len(data) == 0:
            print("Warning: No features extracted")
            return
            
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)
        self.is_fitted = True
        print("Team classifier trained successfully")

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.array([], dtype=np.int32)
        
        if not self.is_fitted:
            print("Warning: Model not fitted yet, returning default predictions")
            return np.zeros(len(crops), dtype=np.int32)

        try:
            data = self.extract_features(crops, cache_path=None)  # Pas de cache pour les prédictions
            
            if len(data) == 0:
                return np.zeros(len(crops), dtype=np.int32)
                
            projections = self.reducer.transform(data)
            predictions = self.cluster_model.predict(projections)
            
            # S'assurer que nous retournons le bon nombre de prédictions
            if len(predictions) != len(crops):
                print(f"Warning: Prediction mismatch. Expected {len(crops)}, got {len(predictions)}")
                return np.zeros(len(crops), dtype=np.int32)
                
            return predictions.astype(np.int32)
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return np.zeros(len(crops), dtype=np.int32)

    def resolve_goalkeepers_team_id(self, players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
        if len(goalkeepers) == 0:
            return np.array([], dtype=np.int32)
        
        if len(players) == 0:
            return np.zeros(len(goalkeepers), dtype=np.int32)
        
        try:
            goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            
            # Vérifier qu'il y a des joueurs pour chaque équipe
            team_0_players = players_xy[players.class_id == 0]
            team_1_players = players_xy[players.class_id == 1]
            
            if len(team_0_players) == 0:
                return np.ones(len(goalkeepers), dtype=np.int32)  # Tous équipe 1
            if len(team_1_players) == 0:
                return np.zeros(len(goalkeepers), dtype=np.int32)  # Tous équipe 0
            
            team_0_centroid = team_0_players.mean(axis=0)
            team_1_centroid = team_1_players.mean(axis=0)
            
            goalkeepers_team_id = []
            for goalkeeper_xy in goalkeepers_xy:
                dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
                dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
                goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

            return np.array(goalkeepers_team_id, dtype=np.int32)
            
        except Exception as e:
            print(f"Error resolving goalkeeper teams: {e}")
            return np.zeros(len(goalkeepers), dtype=np.int32)

    def crop(self, source_video_path):
        PLAYER_ID = 2
        STRIDE = 30
        frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)

        crops = []
        for frame in tqdm(frame_generator, desc='collecting crops'):
            result = self.player_detection_model(frame, conf=0.3)[0]
            detections = sv.Detections.from_ultralytics(result)
            players_detections = detections[detections.class_id == PLAYER_ID]
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            crops += players_crops
        
        print(f"Collected {len(crops)} player crops")
        return crops