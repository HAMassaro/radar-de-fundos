# Radar de Fundos

Código do TCC "Sistema de recomendação de fundos de investimento baseado em clusterização por aprendizado de máquina com dados públicos da CVM", de Hugo Antonio Massaro (Engenharia de Computação, UTFPR, Câmpus Apucarana, 2026). Orientador: Prof. Fábio Irigon Pereira.

Interface publicada: https://radar-de-fundos-gules.vercel.app

## Dados

Informes diários de fundos (dez/2020 a dez/2025), `cad_fi.csv`, `registro_fundo.csv` e `registro_classe.csv`, do Portal de Dados Abertos da CVM (https://dados.cvm.gov.br), baixados em setembro de 2026. Os arquivos brutos (cerca de 3 GB) não estão no repositório: o ETL espera encontrá-los em `raw/` e `registro_fundo_classe/`. O banco SQLite é criado em `~/tcc_scratch/fundos_real.db`.

Como a CVM atualiza os arquivos, dados baixados em outra data podem dar números um pouco diferentes. Os resultados usados na monografia estão em `reports_real/`.

## Instalação

    pip install -r requirements.txt

Python 3.10.

## Execução

| Ordem | Script | O que faz |
|---|---|---|
| 1 | `etl_real.py` | unifica os formatos antes e depois da RCVM 175 e grava o banco SQLite |
| 2 | `pipeline_real.py` | elegibilidade, features, K-Means (k = 3) e relatórios em `reports_real/` |
| 3 | `analises_estatisticas.py` | medianas, delta de Cliff, correlações, tercis, PCA, estabilidade |
| 4 | `analises_backtest.py 2024` e `analises_backtest.py 2025` | backtests e comparações |
| 5 | `analise_janela.py` | sensibilidade com janela de 504 pregões |
| 6 | `comparacao_algoritmos.py` | Ward, misturas gaussianas e DBSCAN |
| 7 | `figuras_e_painel.py` | figuras e `painel_dados.json` usado pela interface |

Análises complementares: `teste_redundancia.py`, `teste_pc1_tercis.py`, `teste_corte_mesmo_tamanho.py`, `sorteios_estratificados.py`, `validacao_fora_amostra.py`, `retorno_concentrado.py`, `backtest_publico_geral.py`.

API: `uvicorn api_real:app`. Interface estática: `web/index.html`.

## Parâmetros principais

Janela de 756 pregões; winsorização de 1% por cauda na volatilidade e na assimetria; exclusão de séries com variação diária acima de 50%; K-Means com k-means++, 10 inicializações e semente 42; backtests com corte em 31/12/2023 (avaliado em 2024) e em 31/12/2024 (avaliado em 2025).

## Licença

CC BY-NC 4.0.
