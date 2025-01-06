import pandas as pd
from fpgrowth_py import fpgrowth
import pickle
import os
import sys
import logging
from collections import defaultdict
from typing import List, Set, Dict
import numpy as np
from datetime import datetime
import hashlib
import time

def get_file_hash(filepath: str) -> str:
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        return "Error hashing file: " + str(e)

class MusicRecommender:
    def __init__(self, dataset_path: str, model_path: str = './shared/models/fpgrowth_model.pkl'):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.model = None
        self.playlists_df = None
        self.song_to_playlists: Dict[str, Set[int]] = defaultdict(set)
        self.playlist_to_songs: Dict[int, Set[str]] = defaultdict(set)
        self.popular_songs: List[str] = []
        self.itemset_index: Dict[str, Set[frozenset]] = defaultdict(set)
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('MusicRecommender')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def load_data(self) -> None:
        self.logger.info("Loading dataset...")
        
        cols = ['pid', 'track_name', 'artist_name']
        self.playlists_df = pd.read_csv(
            self.dataset_path, 
            usecols=cols,
            dtype={
                'pid': np.int32,
                'track_name': str,
                'artist_name': str
            }
        )
        
        for _, row in self.playlists_df.iterrows():
            self.song_to_playlists[row['track_name']].add(row['pid'])
            self.playlist_to_songs[row['pid']].add(row['track_name'])
        
        self.popular_songs = (self.playlists_df['track_name']
                            .value_counts()
                            .head(50)
                            .index.tolist())
        
        self.logger.info(f"Loaded {len(self.playlists_df)} entries. Contains {len(self.song_to_playlists)} songs")

    def train_model(self, minSupRatio: float = 0.05, minConf: float = 0.2) -> None:
        self.logger.info("Training model...")
        start_time = datetime.now()

        transactions = list(self.playlist_to_songs.values())

        freq_itemsets, _ = fpgrowth(transactions, minSupRatio=minSupRatio, minConf=minConf)

        for itemset in freq_itemsets:
            frozen_itemset = frozenset(itemset)
            for item in itemset:
                self.itemset_index[item].add(frozen_itemset)
        
        self.model = freq_itemsets

        self.logger.info("Model trained successfully.")
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'itemset_index': self.itemset_index,
                'popular_songs': self.popular_songs
            }, f)
        
        training_time = datetime.now() - start_time
        self.logger.info(f"Model trained in {training_time.total_seconds():.2f} seconds")
        self.logger.info(f"Model saved to {self.model_path}")

    def load_trained_model(self) -> None:
        self.logger.info("Loading trained model...")
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.itemset_index = data['itemset_index']
            self.popular_songs = data['popular_songs']
        self.logger.info("Model loaded successfully!")

    def recommend(self, song_list: List[str], max_recommendations: int = 10) -> List[str]:
        recommendations = set()
        start_time = datetime.now()
        
        for song in song_list:
            if song in self.itemset_index:
                for itemset in self.itemset_index[song]:
                    recommendations.update(itemset)
        
        recommendations.difference_update(song_list)
        
        if not recommendations:
            self.logger.info("No recommendations in model, generating new one")
            recommendations = set(np.random.choice(
                self.popular_songs[:50], 
                size=min(max_recommendations, len(self.popular_songs[:50])), 
                replace=False
            ))
        
        recommendations_list = list(recommendations)[:max_recommendations]
        
        processing_time = datetime.now() - start_time
        self.logger.info(f"Generated {len(recommendations_list)} recommendations in {processing_time.total_seconds():.3f} seconds")
        
        return recommendations_list

def main():
    main_logger = logging.getLogger('Main')
    main_logger.setLevel(logging.INFO)

    if not main_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        main_logger.addHandler(handler)

    central_dataset = os.getenv('CENTRAL_DATASET_PATH')
    user_dataset = os.getenv('USER_DATASET_PATH')
    model_path = os.getenv('MODEL_PATH')
    check_interval = int(os.getenv('CHECK_INTERVAL', '30'))

    if not central_dataset or not user_dataset:
        main_logger.error("Error: CENTRAL_DATASET_PATH and USER_DATASET_PATH environment variables must be set")
        sys.exit(1)

    dataset_hashes = {
        central_dataset: "",
        user_dataset: ""
    }

    recommender = MusicRecommender(central_dataset, model_path)

    try:
        main_logger.info("Starting training with main dataset...")
        recommender.load_data()
        recommender.train_model(minSupRatio=0.03, minConf=0.1)
        
        for dataset in [central_dataset, user_dataset]:
            dataset_hashes[dataset] = get_file_hash(dataset)
            
    except Exception as e:
        main_logger.error(f"Initial training failed: {str(e)}")
        sys.exit(1)

    main_logger.info("Starting dataset monitoring...")
    while True:
        try:
            datasets_changed = False
            
            for dataset in [central_dataset, user_dataset]:
                current_hash = get_file_hash(dataset)
                if current_hash and current_hash != dataset_hashes[dataset]:
                    main_logger.info(f"Dataset update detected in: {dataset}")
                    datasets_changed = True
                    dataset_hashes[dataset] = current_hash

            if datasets_changed:
                main_logger.info("Dataset changes detected. Retraining model...")
                recommender.dataset_path = central_dataset 
                recommender.load_data()
                recommender.train_model(minSupRatio=0.03, minConf=0.1)

            time.sleep(check_interval)
            
        except Exception as e:
            main_logger.error(f"Error in monitoring loop: {str(e)}")
            time.sleep(check_interval)

if __name__ == "__main__":
    main()
