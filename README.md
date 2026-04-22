# SignFlow Realtime

SignFlow Realtime é uma aplicação desktop de tradução de sinais em tempo real.
Ela abre a webcam do usuário, detecta as mãos quadro a quadro, observa a
sequência temporal do movimento e transforma isso em legendas que aparecem
na tela — enquanto você faz o sinal.

O projeto não é um reconhecedor de poses estáticas. O classificador trabalha
sempre sobre uma janela temporal de vários quadros consecutivos, porque é
isso que define um sinal de verdade: o movimento, a trajetória e a forma da
mão ao longo do tempo, não um único frame.

## O que o sistema faz

- Abre a webcam e mostra um preview em tempo real.
- Detecta uma ou duas mãos e extrai os pontos anatômicos delas
  (21 landmarks por mão).
- Normaliza esses pontos para um formato invariante a translação e escala,
  de modo que o mesmo sinal feito mais perto ou mais longe da câmera produza
  features parecidas.
- Mantém uma janela temporal com os últimos N quadros.
- Quando a janela está cheia, alimenta o classificador temporal e recebe
  uma distribuição de probabilidade sobre o vocabulário de sinais.
- Aplica suavização, limiar de confiança e confirmação temporal antes de
  promover uma legenda — para evitar que a legenda fique pulando a cada
  quadro.
- Mostra a legenda atual na interface desktop, junto com medidores de
  confiança, FPS e preenchimento do buffer temporal.
- Publica cada legenda estabilizada também em uma API local via WebSocket,
  pronta para ser consumida por um site, um overlay, uma live ou outro app.

## Por onde começar

A aplicação já funciona ponta a ponta mesmo sem modelo treinado: o MVP vem
com um classificador "mock" que exercita todo o pipeline, de modo que você
pode abrir a webcam, ver as mãos sendo detectadas, ver a janela temporal
enchendo e ver legendas aparecendo estabilizadas.

Quando você treinar o seu próprio modelo (o pipeline de treino está incluído
no projeto), basta apontar a configuração para o arquivo do modelo e o
mesmo sistema passa a usar o backend real — a arquitetura foi desenhada
justamente para isso, classificador é uma peça trocável.

## Estrutura do projeto

- `src/models` — tipos de dados usados em todo o sistema (landmarks,
  predição, confiança, idioma).
- `src/interface` — os contratos (Protocols) que separam o domínio dos
  detalhes de implementação.
- `src/implementations` — implementações concretas: captura via OpenCV,
  extração via MediaPipe, classificador mock, classificador Keras, serviços
  temporais, pipeline principal, configuração, logging.
- `src/server` — API local com FastAPI e broadcaster WebSocket.
- `src/ui` — interface desktop em CustomTkinter com preview, legenda,
  medidor de confiança e painel de configurações.
- `src/bootstrap.py` e `src/main.py` — composition root e entry point
  do aplicativo.
- `training/` — pipeline offline para coletar amostras, preparar datasets,
  treinar e avaliar um modelo temporal real.
- `scripts/` — atalhos para rodar o aplicativo desktop ou em modo headless.
- `artifacts/` — onde os modelos treinados e o vocabulário de rótulos
  vivem.
- `docs/architecture.md` — notas de arquitetura mais detalhadas.
- `config.yaml` — todos os parâmetros de runtime (câmera, tamanho da
  janela temporal, limiares, idioma, porta da API).

## Próximos passos para virar produto

O MVP foi desenhado como base, não como produto final. Os próximos passos
naturais são construir um dataset real com múltiplos usuários, substituir
o classificador mock por um modelo temporal treinado (uma LSTM baseline
já está incluída no pipeline de treino, com espaço para evoluir para
Transformer ou modelos seq2seq), adicionar canais de velocidade e
aceleração nas features, introduzir segmentação de início e fim de sinal
para trabalhar com fala sinalizada contínua, e finalmente migrar de
classificação de sinais isolados para tradução de frases completas. Tudo
isso pode ser feito sem romper a arquitetura, porque cada etapa fica atrás
de uma interface bem definida.
