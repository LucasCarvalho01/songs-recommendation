from flask import Flask, request, jsonify
from datetime import datetime
import os
import pickle
from typing import List, Dict, Any
import logging
from functools import lru_cache
import time
import threading
import hashlib

app = Flask(__name__)

API_VERSION = os.getenv('API_VERSION', '1.0')
MODEL_PATH = os.getenv('MODEL_PATH', './shared/models/fpgrowth_model.pkl')
MAX_RECOMMENDATIONS = int(os.getenv('MAX_RECOMMENDATIONS', '10'))
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '30'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_file_hash(filepath: str) -> str:
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Error hashing file: {str(e)}")
        return str(e)

class RecommendationAPI:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.itemset_index = None
        self.popular_songs = None
        self.model_date = None
        self.current_hash = ""
        self.load_model()
        self.start_model_monitor()

    def load_model(self) -> None:
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.itemset_index = model_data['itemset_index']
                self.popular_songs = model_data['popular_songs']
            
            self.model_date = datetime.fromtimestamp(
                os.path.getmtime(self.model_path)
            ).strftime('%Y-%m-%d %H:%M:%S')

            self.current_hash = get_file_hash(self.model_path)
            
            logger.info(f"Model loaded successfully. Model date: {self.model_date}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def monitor_model_file(self) -> None:
        while True:
            try:
                current_hash = get_file_hash(self.model_path)

                if current_hash and current_hash != self.current_hash:
                    logger.info("Model file change detected. Reloading model...")
                    self.load_model()

                    logger.info("Model reloaded successfully")  
                    self.get_recommendations.cache_clear()

                time.sleep(CHECK_INTERVAL)
              
            except Exception as e:
              logger.error(f"Error in model monitoring: {str(e)}")
              time.sleep(CHECK_INTERVAL)

    def start_model_monitor(self) -> None:
        monitor_thread = threading.Thread(
            target=self.monitor_model_file, 
            daemon=True
        )
        monitor_thread.start()
        logger.info("Model monitoring thread started")

    def get_recommendations(self, songs_tuple: tuple) -> List[str]:
        import random

        recommendations = set()
        songs_list = list(songs_tuple)
        
        for song in songs_list:
            if song in self.itemset_index:
                for itemset in self.itemset_index[song]:
                    recommendations.update(itemset)
        
        recommendations.difference_update(songs_list)
        
        if not recommendations:
            logger.info("No recommendations found in model, using popular songs")
            recommendations = set(random.sample(
                self.popular_songs[:50], 
                k=min(MAX_RECOMMENDATIONS, len(self.popular_songs[:50]))
            ))
        
        return list(recommendations)[:MAX_RECOMMENDATIONS]

service = RecommendationAPI(MODEL_PATH)

@app.before_request
def initialize():
    global service
    if service.model is None:
        service.load_model()

@app.route('/api/recommend', methods=['POST'])
def recommend() -> Dict[str, Any]:
    start_time = time.time()
    
    try:
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type deve ser application/json'
            }), 400

        data = request.get_json()
        
        if 'songs' not in data:
            return jsonify({
                'error': 'Request precisa ter lista "songs"'
            }), 400

        if not isinstance(data['songs'], list):
            return jsonify({
                'error': '"songs" deve ser uma list'
            }), 400

        if not data['songs']:
            return jsonify({
                'error': '"songs" nao deve ser vazio'
            }), 400

        recommendations = service.get_recommendations(tuple(data['songs']))

        response = {
            'songs': recommendations,
            'version': API_VERSION,
            'model_date': service.model_date
        }

        processing_time = time.time() - start_time
        logger.info(f"Request processed in {processing_time:.3f} seconds")
        
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check() -> Dict[str, Any]:
    return jsonify({
        'status': 'healthy',
        'version': API_VERSION,
        'model_date': service.model_date
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', '52043'))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )