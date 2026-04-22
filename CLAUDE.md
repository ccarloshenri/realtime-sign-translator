# CLAUDE.md

Notas para o Claude Code sobre como trabalhar neste repositório.

## Visão geral

SignFlow Realtime é uma aplicação desktop Python que traduz sinais de mão
em legendas, em tempo real. O coração do sistema é o pipeline temporal:
frames da webcam → landmarks MediaPipe → normalização → buffer deslizante
de N quadros → classificador temporal → suavização → legenda publicada na
UI e em um WebSocket local.

O projeto está organizado em camadas dentro de `src/`. A regra de
dependência é estrita: as camadas externas podem depender das internas,
nunca o contrário.

```
src/
├── models/           # Tipos de dados (entities + value objects)
├── interface/        # Protocols que definem contratos
├── implementations/  # Adapters concretos + serviços + pipeline
├── server/           # FastAPI + WebSocket
├── ui/               # Desktop CustomTkinter
├── bootstrap.py      # Composition root
└── main.py           # Entry point
```

`models` e `interface` não importam nada de `implementations`, `server` ou
`ui`. `implementations` importa apenas de `models` e `interface`. O
composition root em `bootstrap.py` é o único lugar onde todas as peças se
encontram.

## Convenções

- Python 3.10+. Use `from __future__ import annotations` em todos os
  módulos para tipos postponed.
- Prefira `dataclass(frozen=True, slots=True)` para value objects e
  `dataclass(slots=True)` para entidades que mutam. Para instâncias
  `slots=True`, use `dataclasses.replace()` ao copiar — não existe
  `__dict__`.
- Tipagem em todos os parâmetros públicos. Classes sem tipagem são um
  cheiro ruim.
- Logs são estruturados: `logger.info("evento.nome", key=value, ...)`. Não
  concatene strings no log.
- Use `src/interface/*.py` quando precisar abstrair qualquer recurso
  externo (câmera, modelo, publisher, etc.). O pipeline nunca deve
  depender de uma classe concreta.
- Strings de negócio (rótulos, mensagens para o usuário) vão em
  `src/models/language.py` ou em `Enum`s. Evite strings soltas
  espalhadas pelo código.
- Não adicione comentários redundantes. Comentários existem para explicar
  o *porquê*, não o *quê*. Se o comentário só repete o nome do identificador,
  apague.

## Adicionando um novo adapter

Fluxo padrão:

1. Defina o contrato em `src/interface/<nome>.py` como um `Protocol`.
2. Crie a implementação em `src/implementations/<grupo>/<nome>.py` — ela
   deve importar apenas de `src.models` e `src.interface`.
3. Plugue a implementação no `src/bootstrap.py`.
4. Se o adapter tem um formato de saída novo, adicione o value object em
   `src/models/` — não coloque lógica em models, eles devem ser dados
   puros.
5. Adicione testes em `tests/` usando Mock/Fake no lugar do adapter
   quando a dependência for cara (câmera, rede, disco).

## Pipeline temporal

O componente central é `SequenceBuffer` (em
`src/implementations/services/sequence_buffer.py`). Ele é thread-safe,
tem tamanho fixo e só libera uma `snapshot` quando está cheio. Todo o
resto do pipeline depende dessa garantia: o classificador sempre recebe
`(sequence_length, feature_size)`.

Se você for mexer no feature encoder, mexa também no `FeatureEncoder` em
`src/implementations/ml/feature_encoder.py` — ele é o único lugar que o
pipeline offline (de treino) e o pipeline online compartilham para gerar
features. Mantenha os dois no mesmo formato, senão o modelo treinado vai
ver um input diferente em produção.

## Classificador

O classificador é plugável via `classifier.backend` no `config.yaml`:

- `mock` — `MockSequenceClassifier`, produz predições determinísticas
  baseadas em estatísticas simples da sequência. Útil para demonstrar o
  pipeline end-to-end sem precisar treinar nada.
- `keras` — `KerasSequenceClassifier`, carrega um `.keras` treinado pelo
  pipeline em `training/` e faz inferência real.

Para trocar o backend para PyTorch ou ONNX Runtime, implemente
`ISequenceClassifier`, registre no `bootstrap.py`, e pronto. Nenhum outro
arquivo precisa mudar.

## Threads

- **Main thread**: Tk event loop da UI.
- **Pipeline thread**: captura + inferência. Vive em
  `RunTranslationPipeline._run()`.
- **API thread**: uvicorn com seu próprio asyncio loop.

O `WebSocketBroadcaster` expõe `publish_from_any_thread()` para permitir
que a thread do pipeline publique no loop assíncrono da API com
segurança. Não invente novos pontos de ponte entre threads sem seguir
esse padrão.

## Testes

Os testes unitários estão em `tests/` e cobrem os serviços puros
(SequenceBuffer, LandmarkNormalizer, PredictionSmoother,
MockSequenceClassifier, SignPrediction). Eles não requerem OpenCV,
MediaPipe ou CustomTkinter — quanto mais isolado for o teste, mais rápido
ele roda e mais cedo pega regressões.

Para testar o pipeline end-to-end sem hardware, use as fakes no padrão
`FakeCamera` / `FakeExtractor` (implementam as Protocols mas não
dependem de bibliotecas pesadas).

Rode tudo com:

```
python -m pytest -q
```

## Configuração

Toda configuração de runtime vem de `config.yaml`, validada por Pydantic
v2 em `src/implementations/config/yaml_configuration.py`. Não leia YAML em
nenhum outro lugar — dependa sempre do `YamlConfigurationProvider` ou do
`AppConfig` já validado.

## O que não fazer

- Não importe nada de `src.ui` ou `src.server` dentro de
  `src.implementations`.
- Não adicione lógica de negócio em `src.models`.
- Não leia `config.yaml` fora do `YamlConfigurationProvider`.
- Não crie threads silenciosas. Toda thread precisa ser daemon e precisa
  ser encerrada de forma determinística pelo componente que a criou.
- Não escreva comentários que repitam o nome da função ou da variável.
