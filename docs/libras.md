# Libras integration

Libras (Língua Brasileira de Sinais) é uma feature de primeira classe do
SignFlow. Toda a lógica específica de Libras vive no pacote
`src/implementations/libras/`, encaixada sobre a arquitetura existente
sem duplicar câmera, extrator MediaPipe, buffer temporal, smoothing,
UI ou servidor de captions.

## Como a feature se encaixa na arquitetura

```
┌──────────────────────────────────────────────────────────────────────┐
│  Bootstrap  (classifier.backend: libras)                             │
│     ├─ OpenCVCamera                    [existente]                   │
│     ├─ MediaPipeHandLandmarkExtractor  [existente]                   │
│     ├─ LandmarkNormalizer              [existente — gera (T, F)]     │
│     ├─ SequenceBuffer                  [existente]                   │
│     ├─ LibrasSequenceClassifier        [novo]                        │
│     │     └─ LibrasFeatureExtractor    [novo — enriquece p/ (T, 2F)] │
│     ├─ PredictionSmoother              [existente]                   │
│     └─ RunTranslationPipeline(source="libras")   [param novo]        │
│                                                                      │
│  Saída                                                               │
│     ├─ TranslationViewModel            [existente — UI]              │
│     └─ LibrasCaptionService  ──►  WebSocketCaptionPublisher [existente] │
└──────────────────────────────────────────────────────────────────────┘
```

Os componentes com `[novo]` são os únicos adicionados pela integração.
Nada do código generalizado foi reescrito.

## Arquivos relevantes

| Caminho | Papel |
|---|---|
| `src/models/libras_vocabulary.py` | `LibrasSign` Enum + `LIBRAS_BASE_VOCABULARY` |
| `src/implementations/libras/libras_feature_extractor.py` | Enriquece `(T, F)` → `(T, 2F)` adicionando canais de velocidade |
| `src/implementations/libras/libras_sequence_classifier.py` | Classificador Keras dedicado; carrega `artifacts/libras_lstm.keras` + `libras_labels.json` |
| `src/implementations/libras/libras_caption_service.py` | Publisher que tagea toda predição com `source="libras"` |
| `src/implementations/libras/libras_inference_service.py` | Facade fino sobre `RunTranslationPipeline` com API Libras-oriented |
| `training/libras/train_libras_model.py` | Treinador validador de vocabulário; aplica o mesmo `LibrasFeatureExtractor` do runtime |

## Por que canais de velocidade

Em Libras, sinais dinâmicos dependem do movimento, não só da
configuração final da mão. "Obrigado" e "amor" têm handshapes
parecidas — o que os separa é a **trajetória**. O `LibrasFeatureExtractor`
entrega ao modelo, por frame, `[posição; velocidade]` (com velocidade
sendo a diferença para o frame anterior, zero no primeiro frame). Isso
é computado **uma única vez**, no runtime e no treino, pelo mesmo
objeto — eliminando o risco de schema drift entre produção e
desenvolvimento.

## Vocabulário inicial

10 sinais seed, declarados como `LibrasSign` em
`src/models/libras_vocabulary.py`:

```
olá · tchau · sim · não · obrigado · por favor · ajuda · eu · você · amor
```

Esses rótulos são o mínimo pra exercitar o loop completo (coleta →
treino → inferência). Libras real tem mais de 10.000 sinais; a
arquitetura não assume vocabulário fixo. O `LibrasSequenceClassifier`
lê o vocabulário real de `artifacts/libras_labels.json` — o arquivo que
sai do treino — e o modelo gera softmax sobre essa ordem.

Pra crescer o vocabulário: colete os novos sinais em `samples/<label>/`,
rode `build_dataset` de novo, retreine. O runtime descobre a nova
ordem ao carregar o modelo.

## Como usar

### 1. Coletar amostras

```
python -m training.data_collection.collect_samples --label olá --samples 40
python -m training.data_collection.collect_samples --label tchau --samples 40
# ... repita para cada sinal do vocabulário
```

### 2. Construir o dataset

```
python -m training.preprocessing.build_dataset
```

### 3. Treinar o modelo Libras

```
pip install tensorflow
python -m training.libras.train_libras_model
```

O script:
1. valida que todos os rótulos estão em `LIBRAS_BASE_VOCABULARY` (passe
   `--allow-unknown-labels` se estiver estendendo deliberadamente);
2. enriquece cada amostra com velocidade usando `LibrasFeatureExtractor`;
3. treina um Bi-LSTM sobre `(T, 2F)`;
4. salva `artifacts/libras_lstm.keras` e `artifacts/libras_labels.json`.

### 4. Ativar o backend

No `config.yaml`:
```yaml
classifier:
  backend: libras
```

Rode o app:
```
python -m src.main
```

Os logs vão mostrar `libras.model_loaded`. Cada legenda estabilizada é
publicada na UI, no endpoint `/predictions/latest` e via WebSocket em
`/ws/captions` com `source: "libras"`:

```json
{
  "text": "ajuda",
  "confidence": 0.93,
  "timestamp": "2026-04-22T15:30:00Z",
  "source": "libras",
  "sequence_size": 30
}
```

### Se o modelo ainda não foi treinado

O bootstrap captura `FileNotFoundError` (modelo ou labels ausentes) e
cai pra um mock seedado com `LIBRAS_BASE_VOCABULARY`, logando
`libras.fallback_to_mock` com o hint de como treinar. O app continua
bootando; você sabe que precisa treinar porque os logs são explícitos e
o mock apenas cicla os rótulos em timer.

## Evolução futura

A arquitetura está preparada pra crescer sem quebrar camadas:

- **Mais sinais** — colete, retreine. Nenhuma mudança de código.
- **Mais modalidades** (face, corpo) — troque o extrator por
  `MediaPipe Holistic` implementando `IHandLandmarkExtractor`; o
  `LibrasFeatureExtractor` pode ser versionado pra incluir esses
  canais.
- **Segmentação contínua** — substitua a janela deslizante por uma
  detecção de início/fim; `RunTranslationPipeline` já aceita o hook.
- **Seq2seq** — implemente um `ISequenceClassifier` que emita frases;
  o smoother continua compatível.

Nada disso exige refatorar o pacote `src/implementations/libras/`; são
adições em cima de interfaces estáveis.
