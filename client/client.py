import csv
import random
import requests
import json
import os
from typing import List

def read_songs_from_csv(file_path):
    songs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader) 
        for row in csv_reader:
            if len(row) >= 2: 
                songs.append((row[0], row[1]))
    return songs

def get_random_songs(songs, min_songs, max_songs):
    num_songs = random.randint(min_songs, max_songs)
    selected_songs = random.sample(songs, num_songs)
    return [song[1] for song in selected_songs]

def send_request(songs, api_url):
    headers = {'Content-Type': 'application/json'}
    payload = {'songs': songs}
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def save_response(request_num, response, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"\nRequest {request_num}\n")
        f.write(json.dumps(response, indent=2))
        f.write("\n" + "\n")

def main():
    CSV_PATH = os.environ['CSV_PATH']
    API_URL = os.environ['API_URL']
    OUTPUT_PATH = os.environ['OUTPUT_PATH']
    NUM_REQUESTS = int(os.environ.get('NUM_REQUESTS', '5'))
    
    open(OUTPUT_PATH, 'w').close()
    
    songs = read_songs_from_csv(CSV_PATH)
    if not songs:
        with open(OUTPUT_PATH, 'w') as f:
            f.write("Nenhuma msica no arquivo CSV")
        return
    
    for i in range(NUM_REQUESTS):
        selected_songs = get_random_songs(songs, min_songs=1, max_songs=5)
        response = send_request(selected_songs, API_URL)
        save_response(i, response, OUTPUT_PATH)

if __name__ == "__main__":
    main()
