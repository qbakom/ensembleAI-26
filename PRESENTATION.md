# EnsembleAI 2026 — Podsumowanie

---

## Task 1: ChEBI Multi-Label Molecular Classification

**Cel:** Klasyfikacja wieloetykietowa molekul (SMILES) do 500 klas ChEBI. Metryka: macro-F1.

**Co zrobiliśmy:**
- Pipeline oparty na LightGBM z trzema typami fingerprintów molekularnych: Morgan (2048 bit), MACCS keys (167 bit), RDKit topological (2048 bit) — łącznie 4263 cechy
- Automatyczne balansowanie klas przez `scale_pos_weight` (stosunek negatywnych do pozytywnych próbek)
- DAG post-processing: hierarchiczna spójność predykcji na podstawie ontologii ChEBI (plik OBO) — jeśli dziecko jest pozytywne, wszyscy przodkowie też muszą być
- System checkpointów do wznawiania treningu
- 33 668 próbek treningowych, 11 223 testowych, wszystkie SMILES parsowane poprawnie przez RDKit

**Problemy:**
- 500 binarnych klasyfikatorów = długi czas treningu (~1-2h na CPU)
- SGDClassifier (wcześniejsze wersje) dawał słabe wyniki — przejście na LightGBM znacząco poprawiło jakość
- Brak GPU uniemożliwił testowanie podejść opartych na GNN/transformerach (ChemBERTa)

---

## Task 2: JetBrains Code Completion Context Collection

**Cel:** Dla danego punktu w kodzie Python zebrać najlepszy kontekst z repozytorium, aby model LLM mógł dokończyć kod (Fill-in-the-Middle). Metryka: ChrF score.

**Co zrobiliśmy:**
- AST chunk-based retrieval — parsowanie plików na chunki (klasy, funkcje, bloki top-level) zamiast całych plików
- 9 sygnałów rankingowych:
  - Import resolution (8.0) — pliki importowane przez edytowany plik
  - Symbol overlap (6.0) — wspólne nazwy zmiennych/funkcji
  - Imported names (4.0) — konkretne nazwy z importów
  - BM25 (3.0) — klasyczny ranking tekstowy na chunkach
  - Jaccard similarity (3.0) — podobieństwo tokenów
  - Same directory (3.0) — bliskość w strukturze plików
  - TF-IDF cosine (2.5) — wektorowa podobieństwo
  - Recently modified (2.0) — świeżo zmienione pliki
  - Kind bonus (1.5) — preferencja klas i funkcji nad surowym kodem
  - Test penalty (-3.0) — depriorytetyzacja plików testowych
- Kontekst sortowany od najmniej do najbardziej istotnego (bo kontekst jest przycinany od LEWEJ strony)
- Deduplikacja overlappujących chunków AST (np. klasa vs metoda wewnątrz)
- Przetestowane lokalnie na 123 datapointach — średnio 40.8K znaków kontekstu

**Problemy:**
- Serwer submitów dostępny tylko z sieci Cyfronetu — nie mogliśmy testować end-to-end lokalnie
- Balans między ilością kontekstu a jakością — zbyt dużo kontekstu = szum, za mało = brak informacji
- Parsowanie AST na uszkodzonych plikach Python — fallback na line-based chunking

---

## Task 3: Heat Pump Load Forecasting

**Cel:** Predykcja miesięcznego zużycia energii pomp ciepła per urządzenie. Ekstrapolacja szeregów czasowych.

**Co zrobiliśmy:**
- Analiza wzorców sezonowych per urządzenie — identyfikacja cykli grzewczych
- Korelacja z temperaturą zewnętrzną
- Model predykcji oparty na historycznych wzorcach sezonowych

**Problemy:**
- Silna sezonowość utrudnia proste podejścia (np. średnia krocząca)
- Różnice między urządzeniami — każde wymaga indywidualnego modelowania
- Brak dodatkowych danych pogodowych (temperatura, nasłonecznienie) ogranicza dokładność

---

## Task 4: ECG Digitization

**Cel:** Konwersja zdjęć EKG (12 odprowadzeń) na sygnał 1D (500Hz, np.float16).

**Status:** Nie zaimplementowane. Wymaga GPU (A100) do treningu modelu segmentacji obrazów (U-Net lub podobny). Jest to najtrudniejsze zadanie z całego hackathonu — wymagałoby pełnego pipeline'u CV.

---

## Infrastruktura

- **Repozytorium:** GitHub, branch `dev`
- **Środowisko docelowe:** Athena (Cyfronet), SLURM job scheduler, `$SCRATCH` filesystem
- **Submit:** serwer `http://149.156.182.9:6060`, token zespołu, rate limit 600s między submitami
- **Skrypty SLURM:** gotowe dla Task 1, 2, 3 z auto-submitem po zakończeniu treningu
- **Uniwersalny skrypt submisji:** `submit_solution.py` obsługujący wszystkie taski
