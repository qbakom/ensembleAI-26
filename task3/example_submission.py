import os
import requests
from dotenv import load_dotenv

# Wczytaj token i URL z pliku .env
load_dotenv()

# Konfiguracja
ENDPOINT = "task3"
API_TOKEN = os.getenv("TEAM_TOKEN")
SERVER_URL = os.getenv("SERVER_URL")
MY_SUBMISSION_FILE = "submission_full(5).csv"  # Twoja nazwa pliku

def main():
    if not API_TOKEN or not SERVER_URL:
        print("BŁĄD: Brak TEAM_TOKEN lub SERVER_URL w pliku .env")
        return

    if not os.path.exists(MY_SUBMISSION_FILE):
        print(f"BŁĄD: Nie znaleziono pliku {MY_SUBMISSION_FILE}")
        return

    headers = {
        "X-API-Token": API_TOKEN
    }

    # Wysłanie pliku
    print(f"Wysyłanie pliku {MY_SUBMISSION_FILE} do {ENDPOINT}...")
    
    with open(MY_SUBMISSION_FILE, "rb") as f:
        response = requests.post(
            f"{SERVER_URL}/{ENDPOINT}",
            files={"csv_file": f},
            headers=headers
        )

    # Obsługa odpowiedzi
    try:
        data = response.json()
    except:
        data = response.text

    print(f"Status: {response.status_code}")
    print("Odpowiedź serwera:", data)

if __name__ == "__main__":
    main()