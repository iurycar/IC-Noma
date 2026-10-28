from nn import NeuralNetwork
from pathlib import Path
import numpy as np

"""
    Arquivo rl.py
    Este arquivo contém a implementação de uma rede neural simples para aprendizado por reforço (RL).
"""

diretorio = Path(__file__).parent

class ReinforcementLearning(NeuralNetwork):
    def __init__(self, caminho_arquivo: str = None, entrada_tamanho: int = 3, oculta_tamanho: int = 2, saida_tamanho: int = 3):
        super().__init__(caminho_arquivo, entrada_tamanho, oculta_tamanho, saida_tamanho)
        
    def forward_pass(self, estado):
        """
        Realiza a passagem para frente (forward pass) da rede neural.
        Parameters:
            estado (np.ndarray): O estado atual do ambiente, representado como um vetor de entrada.
        Returns:
            np.ndarray: A saída da rede neural.
        """

        self.estado = estado
        
        # Camada 1 (Oculta)
        # O que faz np.dot? Ele realiza a multiplicação matricial entre os pesos W1 e o vetor de entrada estado, 
        # e depois adiciona o viés b1.
        self.Z1 = np.dot(self.W1, self.estado) + self.b1
    
        # A função de ativação ReLU é aplicada à saída da primeira camada (Z1) para introduzir não linearidade
        # na rede neural.
        self.A1 = self.relu(self.Z1)

        # Camada 2 (Saída)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.softmax(self.Z2)

        return self.A2

    def backward_pass(self, saida_esperada: np.ndarray, taxa_aprendizado: float = 0.1):
        """
        Realiza a retropropagação (backpropagation) da rede neural para ajustar os pesos e vieses com base no erro.
        Parameters:
            saida_esperada (np.ndarray): A saída esperada da rede neural.
            taxa_aprendizado (float): A taxa de aprendizado.
        """

        # Gradiente da Camada de Saída (Camada 2)
        # Com Softmax e Cross-Entropy, o gradiente da camada de saída é calculado como a diferença entre a saída da rede neural (A2) e a saída esperada (saida_esperada).
        delta_Z2 = self.A2 - saida_esperada

        # Cálculo dos gradientes para os pesos e vieses da camada de saída    
        delta_W2 = np.dot(delta_Z2, self.A1.T)
        delta_b2 = np.sum(delta_Z2, axis=1, keepdims=True)

        # Gradiente da Camada Oculta (Camada 1)
        delta_A1 = np.dot(self.W2.T, delta_Z2)

        # Cálculo do gradiente da camada oculta usando a multiplicação da derivada da função de ativação ReLU
        delta_Z1 = delta_A1 * self.relu_derivative(self.Z1)

        delta_W1 = np.dot(delta_Z1, self.estado.T)
        delta_b1 = np.sum(delta_Z1, axis=1, keepdims=True)

        self.W2 -= taxa_aprendizado * delta_W2
        self.b2 -= taxa_aprendizado * delta_b2
        self.W1 -= taxa_aprendizado * delta_W1
        self.b1 -= taxa_aprendizado * delta_b1

    def treinamento(self, estado: np.ndarray, saida_esperada: np.ndarray, epochs: int, taxa_aprendizado: float = 0.1):
        """
        Realiza o treinamento da rede neural, incluindo a passagem para frente e a retropropagação.
        Parameters:
            estado (np.ndarray): O estado atual do ambiente, representado como um vetor de entrada.
            saida_esperada (np.ndarray): A saida esperada da rede neural.
            epochs (int): O número de épocas de treinamento.
            taxa_aprendizado (float): A taxa de aprendizado.
        """
        for epoch in range(epochs):    
            saida_real = self.forward_pass(estado)
            self.backward_pass(saida_esperada, taxa_aprendizado)

            if epoch % 4000 == 0:
                erro = self.error_measure(saida_esperada, saida_real)
                print(f"Epoch {epoch}, Erro: {erro}")

        print("Treinamento concluído.")
        print("Pesos e vieses finais:")
        print(self.get_pesos_e_vieses())

        print("Saída final da rede neural:")
        print(self.forward_pass(estado))


    def error_measure(self, saida_esperada: np.ndarray, saida_real: np.ndarray) -> float:
        """
        Calcula a medida de erro entre a saída esperada e a saída real da rede neural
        Parameters:
            saida_esperada (np.ndarray): A saída esperada da rede neural.
            saida_real (np.ndarray): A saída real da rede neural.
        Returns:
            float: A medida de erro calculada.
        """

        # Calcula o erro como a diferença entre a saída esperada e a saída real
        error = (saida_esperada - saida_real) ** 2

        return error


    def q_learning():
        pass

    def reinforce():
        pass

    def policy_gradient():
        pass

    def e_greedy():
        pass

if __name__ == "__main__":

    caminho_arquivo = diretorio / "data" / "data.yaml"

    rl = ReinforcementLearning(caminho_arquivo=str(caminho_arquivo))

    # Entrada: Distância = 0.8 (Longe), Vida = 0.2 (Quase morto)
    estado_atual = np.array([[0.8], [0.2]])

    saida_esperada = np.array([[0.9], [0.1]])  # Exemplo de saída esperada

    rl.treinamento(estado_atual, saida_esperada, epochs=10000, taxa_aprendizado=0.01)
