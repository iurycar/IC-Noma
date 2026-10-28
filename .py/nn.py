import numpy as np
import json
import os

"""
    Arquivo: nn.py
    Este arquivo contém a implementação de uma rede neural simples.
    A rede neural é composta por uma camada de entrada, uma camada oculta e uma camada de saída.
    A função de ativação ReLU é usada na camada oculta, e a função de ativação Softmax é usada na camada de saída.
    A rede neural pode ser inicializada com pesos aleatórios ou carregada a partir de um arquivo JSON contendo os pesos e bias (vieses).
    A função de ativação ReLU é usada para introduzir não linearidade na rede neural
    e a função de ativação Softmax é usada para gerar uma distribuição de probabilidade sobre as ações possíveis na camada de saída.
    O arquivo "data/data.yaml" é usado para armazenar os pesos e vieses da rede neural, permitindo que a rede seja carregada posteriormente com os mesmos parâmetros.

    Função da rede neural:
        f(x) = B0 + Sum(Wk * hk(Xk)) para k = 1 até n

    Onde:
        - f(x) é a saída da rede neural
        - B0 é o viés (bias) da camada
        - Wk são os pesos (weights) da camada
        - hk(Xk) é a função de ativação aplicada à entrada Xk

    Função da rede neural com mais de uma camada oculta:
        f(x) = B0 + Sum(Wk * Ak) para k = 1 até n

    Onde:
        - Ak é a saída da camada oculta k, que é calculada como:
        Ak = hk(Xk) = hk(Sum(Wjk * Xj) + Bj)    
"""

class NeuralNetwork:
    def __init__(self, 
                 caminho_pesos_pre_treinados: str = None,
                 entrada_tamanho: int = 3, 
                 oculta_tamanho: int = 2, 
                 oculta_quantidade: int = 1,
                 saida_tamanho: int = 3
        ):
        
        self.entrada_tamanho = entrada_tamanho
        self.oculta_tamanho = oculta_tamanho
        self.oculta_quantidade = oculta_quantidade
        self.saida_tamanho = saida_tamanho

        self.pesos = []
        self.bias = []
        self.z = []
        self.a = []
        self.estado = None

        if not caminho_pesos_pre_treinados:
            print("Inicializando pesos e bias da rede neural com valores aleatórios.")
            self.inicializar()
        else:
            print(f"Carregando pesos e bias da rede neural a partir do arquivo: {caminho_pesos_pre_treinados}")
            self.carregar_pesos_json(caminho_pesos_pre_treinados)


    def forward_pass(self, estado: np.ndarray, aplicar_softmax: bool = True) -> np.ndarray:
        """
        Realiza a passagem para frente (forward pass) na rede neural.
        Parameters:
            estado (np.ndarray): O estado de entrada para a rede neural.
            aplicar_softmax (bool): Indica se a função Softmax deve ser aplicada na saída.
        Returns:
            np.ndarray: A saída da rede neural após a passagem para frente.
        """
        
        if estado.ndim == 1:
            estado = estado.reshape(-1, 1)  # Converte para vetor coluna

        self.estado = estado.astype(np.float32)
        self.z.clear()
        self.a.clear()

        entrada_atual = self.estado
        num_camadas = len(self.pesos)

        for camada in range(num_camadas):
            z_atual = np.dot(self.pesos[camada], entrada_atual) + self.bias[camada]
            self.z.append(z_atual)

            is_ultima_camada = (camada == num_camadas - 1)
            if not is_ultima_camada:
                a_atual = self.relu(z_atual)
            elif aplicar_softmax:
                a_atual = self.softmax(z_atual)
            else:
                a_atual = z_atual  # Sem ativação na última camada se softmax não for aplicado

            self.a.append(a_atual)
            entrada_atual = a_atual

        return self.a[-1]  # Retorna a saída da última camada


    def backward_pass(self, saida_esperada: np.ndarray, taxa_aprendizado: float = 0.1):
        """
        Realiza a passagem para trás (backward pass) na rede neural, atualizando os pesos e bias.
        Parameters:
            saida_esperada (np.ndarray): A saída esperada para calcular o erro.
            taxa_aprendizado (float): A taxa de aprendizado para atualização dos pesos.
        """

        if saida_esperada.ndim == 1:
            saida_esperada = saida_esperada.reshape(-1, 1)  # Converte para vetor coluna

        num_camadas = len(self.pesos)
        if num_camadas == 0 or len(self.a) != num_camadas:
            return  # Não há camadas ou a passagem para frente não foi realizada

        # Gradiente da camada de saída
        delta = self.a[num_camadas - 1] - saida_esperada # Erro da camada de saída
        delta = np.clip(delta, -2.0, 2.0)  # Limita o delta para evitar explosão de gradiente

        for camada in range(num_camadas - 1, -1, -1):
            if camada > 0:
                a_anterior = self.a[camada - 1]
            else:
                a_anterior = self.estado

            delta_w = np.dot(delta, a_anterior.T)
            delta_b = delta.copy()

            delta_w = np.clip(delta_w, -5.0, 5.0)
            delta_b = np.clip(delta_b, -5.0, 5.0)

            if camada > 0:
                delta_a = np.dot(self.pesos[camada].T, delta)
                delta = delta_a * self.relu_derivative(self.z[camada - 1])
                delta = np.clip(delta, -2.0, 2.0)

            self.pesos[camada] -= taxa_aprendizado * delta_w
            self.bias[camada] -= taxa_aprendizado * delta_b

    def inicializar(self):
        self.pesos.clear()
        self.bias.clear()

        # Sem camadas ocultas / Entrada diretamente para a saída
        if self.oculta_quantidade <= 0:
            self.pesos.append(self.gerar_pesos(self.saida_tamanho, self.entrada_tamanho))
            self.bias.append(self.gerar_bias(self.saida_tamanho))
            return

        # Entrada -> Primeira camada oculta
        self.pesos.append(self.gerar_pesos(self.oculta_tamanho, self.entrada_tamanho))
        self.bias.append(self.gerar_bias(self.oculta_tamanho))

        # Camadas ocultas intermediárias
        for _ in range(self.oculta_quantidade - 1):
            self.pesos.append(self.gerar_pesos(self.oculta_tamanho, self.oculta_tamanho))
            self.bias.append(self.gerar_bias(self.oculta_tamanho))

        # Última camada oculta -> Saída
        self.pesos.append(self.gerar_pesos(self.saida_tamanho, self.oculta_tamanho))
        self.bias.append(self.gerar_bias(self.saida_tamanho))

    def gerar_pesos(self, linhas: int, colunas: int) -> np.ndarray:
        """Gera uma matriz de pesos aleatórios com valores entre -1 e 1."""
        return np.random.uniform(-1, 1, (linhas, colunas)).astype(np.float32)

    def gerar_bias(self, tamanho: int) -> np.ndarray:
        """Gera um vetor de bias (biases) com valores aleatórios para a camada da rede neural."""
        return np.zeros((tamanho, 1), dtype=np.float32)


    # Função de ativação ReLU, para a saída gerar valores não negativos.
    def relu(self, x) -> np.ndarray:
        """Função de ativação ReLU (Rectified Linear Unit)"""
        return np.maximum(0.0, x)

    def relu_derivative(self, x) -> np.ndarray:
        """Derivada da função de ativação ReLU"""
        return np.where(x > 0.0, 1.0, 0.0)

    # Função de ativação Softmax, para a saída gerar uma distribuição de probabilidade sobre as ações possíveis.
    def softmax(self, x) -> np.ndarray:
        """Função de ativação Softmax"""
        max_valor = np.max(x)
        exp_ajustada = np.exp(x - max_valor)  # Subtraindo o valor máximo para estabilidade numérica
        soma_exp = np.sum(exp_ajustada)

        if soma_exp > 0.0:
            return exp_ajustada / soma_exp
        return exp_ajustada  # Retorna o vetor ajustado se a soma for zero para evitar divisão por zero


    def carregar_pesos_json(self, caminho_arquivo: str):
        """
        Carrega os pesos e bias da rede neural a partir de um arquivo JSON.
        Parameters:
            caminho_arquivo (str): O caminho para o arquivo JSON contendo os pesos e bias.
        """
        if not os.path.exists(caminho_arquivo):
            raise FileNotFoundError(f"O arquivo {caminho_arquivo} não foi encontrado.")

        with open(caminho_arquivo, 'r', encoding='utf-8') as file:
            dados = json.load(file)

        self.entrada_tamanho = dados['entradaTamanho']
        self.oculta_quantidade = dados['ocultaQuantidade']
        self.oculta_tamanho = dados['ocultaTamanho']
        self.saida_tamanho = dados['saidaTamanho']

        self.pesos.clear()
        self.bias.clear()

        for camada in range(len(dados['pesos'])):
            matriz_linhas = dados['pesos'][camada]['linhas']
            
            linhas_array = [linha['valores'] for linha in matriz_linhas]
            self.pesos.append(np.array(linhas_array, dtype=np.float32))

            bias_valores = dados['bias'][camada]['valores']
            self.bias.append(np.array(bias_valores, dtype=np.float32).reshape(-1, 1))


    def salvar_pesos_json(self, caminho_arquivo: str):
        """
        Salva os pesos e bias da rede neural em um arquivo JSON.
        Parameters:
            caminho_arquivo (str): O caminho para o arquivo JSON onde os pesos e bias serão salvos.
        """
        dto = {
            "entradaTamanho": self.entrada_tamanho,
            "ocultaTamanho": self.oculta_tamanho,
            "ocultaQuantidade": self.oculta_quantidade,
            "saidaTamanho": self.saida_tamanho,
            "pesos": [],
            "bias": []
        }

        for camada_w in self.pesos:
            lista_linhas = []
            for i in range(camada_w.shape[0]):
                linha_valores = [float(val) for val in camada_w[i, :]]
                lista_linhas.append({"valores": linha_valores})
            
            dto["pesos"].append({"linhas": lista_linhas})

        for camada_b in self.bias:
            valores_bias = [float(val) for val in camada_b.flatten()]
            dto["bias"].append({"valores": valores_bias})

        with open(caminho_arquivo, "w", encoding="utf-8") as f:
            json.dump(dto, f, indent=4)