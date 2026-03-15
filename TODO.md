# EnsembleAI 2026 — Priorytety i Plan Dzialan

## Priorytet 1: Submit istniejacych rozwiazan
- [ ] **Task 2** — submitnac v3 z Atheny (`sbatch slurm/task2.slurm`)
- [ ] **Task 1** — submitnac obecne rozwiazanie i zobaczyc baseline F1
- [ ] **Task 3** — submitnac v2 i sprawdzic wynik

## Priorytet 2: Quick wins (ulepszenia z duzym potencjalnym impactem)

### Task 1 (ChEBI)
- [ ] DAG post-processing — wymuszenie hierarchicznej konsystencji (parent >= child probability)
- [ ] LightGBM zamiast SGD — lepsze dla sparse multi-label
- [ ] MACCS keys + pharmacophore fingerprints jako dodatkowe features

### Task 2 (Context Collection)
- [ ] Tuning wag po pierwszych wynikach ChrF
- [ ] Transitive import graph — jesli A importuje B, a B importuje C, to C tez jest relevantne
- [ ] Lepsze obcinanie prefixu/suffixu (custom prefix/suffix w submisji)

### Task 3 (Heat Pump)
- [ ] Prophet / seasonal decomposition dla lepszej ekstrapolacji
- [ ] Per-device clustering (grupowanie podobnych urzadzen)

## Priorytet 3: Ambitne ulepszenia (jesli jest czas)

### Task 2
- [ ] CodeBERT / code2vec embeddings zamiast BM25
- [ ] Fine-tuned embedding model na code completion tasks
- [ ] Chunk re-ranking z cross-encoder

### Task 1
- [ ] GNN na grafach molekularnych
- [ ] ChemBERTa embeddings
- [ ] Ensemble z voting

### Task 4 (ECG — od zera)
- [ ] Baseline: color segmentation → contour detection → signal extraction
- [ ] U-Net segmentacja linii ECG
- [ ] Grid detection i kalibracja skali

## Znane problemy
1. **Serwer API** dostepny tylko z Atheny — caly workflow submit/check musi isc stamtad
2. **SSH do Atheny** nie dziala z naszego serwera dev — trzeba reczne operacje lub SLURM
3. **Rate limit 600s** — miedzy submisja trzeba czekac 10 minut
4. **Task 4** — najbardziej zlozony, wymaga CV pipeline, brak nawet baseline
5. **Brak wynikow** — zadne rozwiazanie nie zostalo jeszcze zsubmitowane i zweryfikowane
