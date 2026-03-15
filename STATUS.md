# EnsembleAI 2026 Hackathon - Status Report

## Team Token
`b2c6083ba78b4039a6db64a4bb5e07ca`

## Server
`http://149.156.182.9:6060` (dostepny tylko z sieci Cyfronet/Athena)

---

## Task 1: ChEBI Multi-Label Classification
**Status: Zaimplementowane, wymaga submitu i walidacji**

### Co zrobione
- 6 iteracji rozwiazania (`solve_task1.py` → `solve_task1_v6.py`)
- Fingerprint-based approach: Morgan/RDKit fingerprints z RDKit
- v6: SGDClassifier z checkpointami (resume-capable)
- Ridge Regression wariant (`solve_task1_ridge.py`)
- SLURM skrypt gotowy (`slurm/task1.slurm`)

### Podejscie
- SMILES → fingerprints molekularne (Morgan 2048-bit)
- Multi-label klasyfikacja: osobny klasyfikator na kazda klase (~2000 klas)
- SGDClassifier (v6) lub Ridge Regression — szybkie trenowanie na Athenie
- Metryka: macro-averaged F1

### Problemy
- Duza liczba klas (~2000) — trenowanie trwa dlugo
- Brak walidacji na leaderboardzie — nie wiadomo jaki mamy wynik
- Nie zaimplementowano hierarchicznej konsystencji DAG (parent > child)
- Brak zaawansowanych metod: GNN, pretrained chemical models (ChemBERTa), SMARTS patterns

### Do zrobienia
- [ ] Submitnac z Atheny i sprawdzic wynik
- [ ] Dodac DAG post-processing (zapewnic hierarchiczna konsystencje)
- [ ] Rozwazyc pretrained model (np. ChemBERTa) zamiast fingerprints
- [ ] Feature engineering: MACCS keys, pharmacophore fingerprints
- [ ] Ensemble: polaczenie Ridge + SGD + LightGBM

---

## Task 2: Context Collection Pipeline (JetBrains Challenge)
**Status: Zaimplementowane v3, gotowe do submitu**

### Co zrobione
- 3 iteracje rozwiazania (v1 → v3)
- v3: AST chunk-based retrieval z multi-signal rankingiem
- Przetestowane lokalnie: 123/123 datapointow, avg 40.8K chars kontekstu
- SLURM skrypt gotowy (`slurm/task2.slurm`)
- Nowy dataset (python-dataset.zip) pobrany i przeslany na Athene

### Podejscie (v3 — obecne)
- **AST Chunking**: Parsowanie plikow Python do semantycznych blokow (klasy, metody, funkcje) zamiast calych plikow
- **Multi-signal ranking** (9 sygnalow):
  - Import resolution: 8.0 (bezposrednio importowane pliki)
  - Symbol/definition overlap: 6.0 (definicje uzywane w prefix/suffix)
  - Imported names overlap: 4.0 (from X import Y → definicje Y)
  - BM25 (prefix-weighted): 3.0 (prefix wazony 3x nad suffix)
  - Jaccard na identyfikatorach: 3.0 (overlap zbiorow identyfikatorow)
  - Same directory proximity: 3.0
  - TF-IDF cosine similarity: 2.5 (rzadkie tokeny)
  - Recently modified: 2.0
  - Kind/structure bonuses: 1.5
  - Test/doc penalty: -3.0 (redukcja szumu)
- **Context ordering**: najwazniejsze chunki na KONCU (kontekst trimowany od lewej)
- **Budget**: ~60K chars — pelne pokrycie dla 16K modeli, tail dla 8K (Mellum)

### Ewaluacja
- 3 modele: Mellum (8K tokens), Codestral (16K), Qwen2.5-Coder (16K)
- Metryka: ChrF Score (character n-gram F-score)
- Stage: `public`
- Ewaluacja trwa ~20-30 min po submicie

### Problemy
- Serwer API dostepny tylko z sieci Cyfronet — submit musi isc z Atheny
- Brak dotychczasowych wynikow na leaderboardzie (nie submitowano jeszcze)
- SyntaxWarnings z parsowania starych repozytoriow (nie wplywaja na dzialanie)
- Duze repo (dipy) — chunking generuje duzo chunkow, wolniejsze przetwarzanie

### Do zrobienia
- [ ] Submitnac v3 z Atheny (stage=public)
- [ ] Sprawdzic wynik ChrF i porownac z baselinami
- [ ] Rozwazyc: graph-based import traversal (transitive deps)
- [ ] Rozwazyc: code2vec / CodeBERT embeddings zamiast BM25
- [ ] Tuning wag na podstawie wynikow
- [ ] Optymalizacja budzetu: osobne budgety dla 8K/16K modeli

---

## Task 3: Heat Pump Load Forecasting
**Status: Zaimplementowane v2, wymaga submitu**

### Co zrobione
- 2 iteracje rozwiazania (`solve_task3.py`, `solve_task3_v2.py`)
- Dane rozpakowane i przetworzone
- SLURM skrypt gotowy (`slurm/task3.slurm`)

### Podejscie
- Dane 5-minutowe z Oct 2024 - Apr 2025
- Ekstrapolacja na May-Oct 2025 (6 miesiecy per device)
- v2: per-device seasonal patterns + korelacja z temperatura
- Device-specific monthly trends

### Problemy
- To jest zadanie ekstrapolacyjne (prognoza poza zakres treningowy)
- Sezonowosc: dane treningowe to jesien/zima, prognoza na lato — fundamentalnie inny wzorzec zuzycia
- Brak danych pogodowych dla okresu prognozy
- Nie submitowano — brak informacji o wyniku

### Do zrobienia
- [ ] Submitnac z Atheny
- [ ] Rozwazyc bardziej zaawansowane modele (Prophet, LightGBM z features)
- [ ] Dodac external weather data jesli dostepne
- [ ] Cross-validation na dostepnych danych

---

## Task 4: ECG Digitization
**Status: Niezaimplementowane — tylko example submission**

### Co zrobione
- Dane pobrane (train: 3000 obrazow + ground truth, test: 500 obrazow)
- Przykladowy skrypt submisji (`example_submission.py`)
- Brak wlasciwego rozwiazania

### Podejscie (planowane)
- Segmentacja linii ECG z obrazu (CV pipeline)
- Ekstrakcja 1D sygnalu z krzywej ECG
- 12 odprowadzen standardowych (I, II, III, AVR, AVL, AVF, V1-V6)
- Output: 500Hz float16 numpy arrays

### Problemy
- Najbardziej zlozony task — wymaga pipeline CV
- Roznorodnoscobrazow: skany, zdjecia smartfonowe, pognieciony papier
- Brak implementacji — trzeba budowac od zera
- Wymaga GPU do trenowania modelu segmentacji

### Do zrobienia
- [ ] Zaimplementowac baseline: color filtering → line detection → signal extraction
- [ ] Rozwazyc pretrained model (np. U-Net dla segmentacji)
- [ ] Kalibracja: grid detection → skala mm/pixel → voltage/time mapping
- [ ] Obsluga roznych orientacji i jakosci obrazow

---

## Infrastruktura

### Athena (Cyfronet)
- Login: `tutorial243@login01.athena.cyfronet.pl`
- HOME: `/net/people/tutorial/tutorial243`
- SCRATCH: `/net/tscratch/people/tutorial243`
- Kolejka: `tutorial` (account: `tutorial`)
- Python: `GCCcore/13.2.0 Python/3.11.5`
- Venv: `$SCRATCH/venvs/hackathon/`

### Submit
- Endpoint: `http://149.156.182.9:6060/{task}`
- Token header: `X-API-Token`
- Rate limit: 600s miedzy submisja
- Status check: `shared/get_task_status.py --request-id <ID>`

### Repo
- GitHub: `github.com:qbakom/ensembleAI-26.git`
- Branch: `dev`
