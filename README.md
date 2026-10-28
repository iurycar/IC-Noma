# Noma - Iniciação Científica

Repositório disponibilizado para expor o código-fonte da pesquisa de Inteligência Artificial para controlar inimigos em um jogo desenvolvido com Unity. O projeto combina redes neurais, aprendizado supervisionado e aprendizado por reforço para selecionar ações do inimigo.

## Estrutura

- `.cs/` — scripts C# para integração com a Unity:
  - `IA/` — rede neural, aprendizado supervisionado, Q-Learning e persistência de pesos;
  - `Enemy/` — comportamento e tomada de decisão do inimigo.
- `.py/` — versões experimentais em Python para treinamento e análise.
- `data/` — datasets, pesos treinados e métricas exportadas.

Os pesos e métricas gerados podem ser carregados pela IA durante a execução do jogo.