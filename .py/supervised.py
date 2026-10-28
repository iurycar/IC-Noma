from dataclasses import dataclass
import matplotlib.pyplot as plt
from nn import NeuralNetwork
from pathlib import Path
import numpy as np
import csv

BASE_DIR = Path(__file__).parent

@dataclass
class AmostraTreino:
    """
    Classe que representa uma amostra de treino.
    A amostra de treino é composta por um estado, uma ação e uma recompensa.
    """
    entrada: np.ndarray
    saida_esperada: np.ndarray

class SupervisedLearning(NeuralNetwork):
    """
    Classe que representa o aprendizado supervisionado.
    O aprendizado supervisionado é um tipo de aprendizado de máquina onde o modelo é treinado com dados rotulados.
    A classe SupervisedLearning herda da classe NeuralNetwork, permitindo que a rede neural seja treinada com dados rotulados.
    """
    def __init__(self, 
                 caminho_arquivo: str = None, 
                 entrada_tamanho: int = 30, 
                 oculta_tamanho: int = 64, 
                 oculta_quantidade: int = 2,
                 saida_tamanho: int = 13
        ):
        super().__init__(
            caminho_pesos_pre_treinados=caminho_arquivo, 
            entrada_tamanho=entrada_tamanho, 
            oculta_tamanho=oculta_tamanho, 
            oculta_quantidade=oculta_quantidade, 
            saida_tamanho=saida_tamanho
            )

    def carregar_amostras_csv(self, caminho_csv: str, alvo_inimigo: bool = True) -> list[AmostraTreino]:
        """Carrega e converte as amostras do CSV para matrizes de entrada e vetores One-Hot."""

        amostras: list[AmostraTreino] = []
        arquivo = Path(caminho_csv)

        if not arquivo.exists():
            print(f"Arquivo {caminho_csv} não encontrado.")
            return amostras

        with open(caminho_csv, mode='r', encoding='utf-8') as f:
            primeira_linha = f.readline()

            delimitador = ';' if ';' in primeira_linha else ','
            f.seek(0)  # Retorna ao início do arquivo para leitura completa

            leitor = csv.reader(f, delimiter=delimitador)
            _cabecalho = next(leitor, None)  # Pula o cabeçalho

            for linha in leitor:
                if not linha or len(linha) < 33:
                    continue  # Ignora linhas vazias ou incompletas

                # Vetor de entrada com as 30 variáveis de estado
                entradas = []
                for i in range(self.entrada_tamanho):
                    entradas.append(float(linha[i]))

                entrada_matriz = np.array(entradas, dtype=np.float32).reshape(-1, 1)

                # InimigoAcaoId está na coluna 30; JogadorAcaoId está na coluna 32
                if alvo_inimigo:
                    indice_acao = int(linha[30])
                else:
                    indice_acao = int(linha[32])

                saida_one_hot = np.zeros((self.saida_tamanho, 1), dtype=np.float32)
                if 0 <= indice_acao < self.saida_tamanho:
                    saida_one_hot[indice_acao, 0] = 1.0

                amostras.append(
                    AmostraTreino(
                        entrada=entrada_matriz, 
                        saida_esperada=saida_one_hot
                    )
                )

        print(f"Dataset carregado com {len(amostras)} amostras válidas do arquivo {caminho_csv}.")
        return amostras

    def treinar(self, dataset: list[AmostraTreino], epochs = 1000, taxa_aprendizado: float = 0.01, plotar_grafico: bool = True) -> tuple[list[float], list[float]]:
        """
        Treina a rede neural usando aprendizado supervisionado.
        Parameters:
            amostras (list[AmostraTreino]): Lista de amostras de treino.
            epochs (int): Número de épocas para o treinamento.
            taxa_aprendizado (float): Taxa de aprendizado.
        """

        if not dataset:
            print("Dataset vazio ou nulo. Não é possível treinar a rede neural.")
            return [], []

        total_amostras = len(dataset)
        historico_erros = []
        historico_acuracia = []

        print("\n" + "=" * 65)
        print(f"Iniciando Treinamento: {epochs} épocas | {total_amostras} amostras")
        print("=" * 65)

        passo_atualizacao = max(1, total_amostras // 50)

        for epoch in range(epochs):
            soma_erro_quadratico = 0.0
            acertos = 0

            # Embaralhamento das amostras para cada época (Fisher-Yates Shuffle)
            indices = np.random.permutation(total_amostras)

            for idx in indices:
                amostra = dataset[idx]

                saida_real = self.forward_pass(amostra.entrada, aplicar_softmax=True)
                self.backward_pass(amostra.saida_esperada, taxa_aprendizado)

                # Cálculo do erro quadrático
                erro = amostra.saida_esperada - saida_real
                soma_erro_quadratico += float(np.sum(erro ** 2))

                # Cálculo da acurácia (classe prevista vs classe esperada)
                classe_prevista = int(np.argmax(saida_real))
                classe_alvo = int(np.argmax(amostra.saida_esperada))

                if classe_prevista == classe_alvo:
                    acertos += 1

            erro_medio = soma_erro_quadratico / (total_amostras * self.saida_tamanho)
            acuracia = (acertos / total_amostras) * 100.0

            historico_erros.append(erro_medio)
            historico_acuracia.append(acuracia)

            self._renderizar_barra_progresso(epoch + 1, epochs, erro_medio, acuracia)

        print("\n" + "=" * 65)
        print("Treinamento finalizado com sucesso!")

        if plotar_grafico:
            self.exibir_graficos(historico_erros, historico_acuracia)

        return historico_erros, historico_acuracia

    def _renderizar_barra_progresso(
            self,
            epoch_atual: int,
            epoch_total: int,
            erro: float,
            acuracia: float,
            tamanho_barra: int = 30
        ):
        """Renderiza uma barra de carregamento na mesma linha do terminal via carriage return."""

        fracao = epoch_atual / epoch_total
        preenchido = int(tamanho_barra * fracao)
        barra = '█' * preenchido + '-' * (tamanho_barra - preenchido)

        print(
            f"\rEpoch [{epoch_atual:>4}/{epoch_total}] |{barra}| "
            f"Erro: {erro:.5f} | Acurácia: {acuracia:5.1f}%", 
            end='', 
            flush=True
        )

    def exibir_graficos(self, historico_erros: list[float], historico_acuracia: list[float]):
        epochs = range(1, len(historico_erros) + 1)

        plt.figure(figsize=(12, 5))

        # Subplot 1: Erro médio quadrático
        plt.subplot(1, 2, 1)
        plt.plot(epochs, historico_erros, color="crimson", linewidth=1.5, label="Erro Médio")
        plt.title("Convergência da Perda (Loss)")
        plt.xlabel("Épocas")
        plt.ylabel("Erro Médio Quadrático")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        # Subplot 2: Acurácia
        plt.subplot(1, 2, 2)
        plt.plot(epochs, historico_acuracia, color="teal", linewidth=1.5, label="Acurácia (%)")
        plt.title("Acurácia de Imitação das Ações")
        plt.xlabel("Épocas")
        plt.ylabel("Acurácia (%)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    caminho_csv = f"{BASE_DIR}/data/dataset_supervisionado_dois_jogadores.csv"
    caminho_pesos = f"{BASE_DIR}/data/pesos_comportamento_rl.json"
    caminho_saida = f"{BASE_DIR}/data/pesos_treinados_supervisionado.json"

    rede = SupervisedLearning(
        caminho_arquivo=None,
        entrada_tamanho=30,
        oculta_tamanho=20,
        oculta_quantidade=4,
        saida_tamanho=13
    )

    dados = rede.carregar_amostras_csv(caminho_csv, alvo_inimigo=True)

    if dados:
        rede.treinar(
            dataset=dados,
            epochs=100,
            taxa_aprendizado=0.001,
            plotar_grafico=True
        )

        rede.salvar_pesos_json(caminho_saida)
        print(f"Pesos salvos e validados com sucesso em: {caminho_saida}")