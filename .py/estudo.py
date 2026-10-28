# Arquivo de estudo para Reiforcement Learning (Aprendizado por Reforço)
# Link: https://www.youtube.com/watch?v=2uXUK_4_JH4

"""
Os principais componentes do Aprendizado por Reforço são:
1. Agente: O agente é o tomador de decisões que interage com o ambiente
2. Ambiente: O ambiente é o mundo com o qual o agente interage
3. Ação: A ação é o que o agente pode fazer no ambiente
4. Recompensa: A recompensa é o feedback que o agente recebe do ambiente após realizar uma ação

Os elementos principais do RL são:

- Política (Policy): A política é a estratégia do agente para lidar com o mundo. Mapeando entre o que o agente 
    compreende do ambiente (estados) e suas ações. A política pode ser determinística (uma ação específica para cada 
    estado) ou estocástica (uma distribuição de probabilidade sobre as ações possíveis para cada estado).

- Retorno (Return): O retorno é a soma das recompensas futuras que o agente espera receber, ponderadas por um fator
    de desconto. O retorno ajuda o agente a avaliar a qualidade de suas ações e a tomar decisões informadas sobre 
    quais ações seguir para maximizar a recompensa acumulada ao longo do tempo. O ideal é caminho de maior recompensa
    a longo prazo, mesmo que imediatamente repreensivo.
    
- Função de valor (Value Function): A função de valor cumulativa das recompensas de várias interações (longo prazo) 
    que o agente espera receber a partir de um estado específico, seguindo uma política específica.

- Modelo (Model): O modelo é uma representação do ambiente que o agente pode usar para prever as consequências de 
    suas ações.
"""



# Rt = SUM(Y^k*rt+k+1) para k=0 até T
# O valor de Gama (Y) deve ser menor que 1 e maior que 0

def retorno(Y: float, rt: list, T: int) -> float:
    """
    Função para calcular o retorno Rt dado um fator de desconto Y, uma lista de recompensas rt e um horizonte de tempo T.
    """

    Rt: float = 0
    k: int = 0

    while k <= T:
        Rt += (Y**k) * rt[k+1]
        k += 1

    return Rt

def função_de_valor():
    pass

if __name__ == "__main__":
    Y = 0.9
    rt = [0, 1, 2, 3, 4, 5]
    T = 4

    Rt = retorno(Y, rt, T)
    print("Rt:", Rt)