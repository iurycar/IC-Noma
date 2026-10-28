using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Classe que implementa o Aprendizado por Reforço (Reinforcement Learning).
/// Usando Algoritmo Q-Learning ou Algoritmo de REINFORCE.
/// 
/// Q-Learning é um algoritmo de aprendizado por reforço baseado em valores, enquanto REINFORCE é um algoritmo baseado em políticas.
/// 
/// Q-Learning aprende uma função de valor de ação (Q) que estima a qualidade de uma ação em um determinado estado, 
/// enquanto REINFORCE aprende diretamente uma política que mapeia estados para ações.
/// 
/// A escolha entre Q-Learning e REINFORCE depende do problema específico, da complexidade do ambiente e da 
/// disponibilidade de informações sobre o modelo do ambiente. Q-Learning é mais adequado para problemas com espaço de 
/// estado discreto e conhecido.
/// </summary>

public class ReinforcementLearning : NeuralNetwork
{
    [Header("Hiperparâmetros de RL")]
    public float Gamma = 0.9f; // Fator de desconto para recompensas futuras
    public float Epsilon = 1.0f; // Taxa de exploração para a política epsilon-greedy
    public float EpsilonMin = 0.1f; // Valor mínimo de epsilon para exploração
    public float EpsilonDecay = 0.995f; // Fator de decaimento para epsilon após cada episódio

    // Estrutura para armazenar uma transição (estado, ação, recompensa, próximo estado)
    public struct Transicao
    {
        public Matriz Estado; // Estado atual
        public int Acao; // Ação tomada
        public float Recompensa; // Recompensa recebida
        public Matriz ProximoEstado; // Próximo estado após a ação
        public bool Terminado; // Indica se o episódio terminou
    }

    public ReinforcementLearning(string caminhoArquivo = null, int entradaTamanho = 3, int ocultaTamanho = 2, int ocultaQuantidade = 1, int saidaTamanho = 3)
        : base(caminhoArquivo, entradaTamanho, ocultaTamanho, ocultaQuantidade, saidaTamanho)
    {
    }

    /// <summary>
    /// Escolhe uma ação com base no estado atual usando a política epsilon-greedy (E-greedy).
    /// </summary>
    public int EscolherAcaoEGreedy(Matriz estado)
    {
        // Se um valor aleatório for menor que Epsilon, escolhe uma ação aleatória (exploração)
        if (UnityEngine.Random.value < Epsilon)
        {
            return UnityEngine.Random.Range(0, SaidaTamanho);
        }

        // Caso contrário, escolhe a ação com o maior valor Q (explotação)
        Matriz qValores = ForwardPass(estado, aplicarSoftmax: false); // Passagem para frente para obter os valores Q
        int melhorAcao = 0; // Inicializa a melhor ação como a primeira ação
        float melhorValor = qValores.Dados[0, 0];

        // Passar por todas as ações possíveis para encontrar a ação com o maior valor Q
        for (int i = 1; i < SaidaTamanho; i++)
        {            
            if (qValores.Dados[i, 0] > melhorValor)
            {
                melhorValor = qValores.Dados[i, 0]; // Atualiza o melhor valor Q
                melhorAcao = i;
            }
        }

        return melhorAcao; // Retorna a ação com o maior valor Q
    }

    // Função para decair o valor de epsilon ao longo do tempo, incentivando a exploração no início e a explotação no final
    public void DecairEpsilon()
    {
        if (Epsilon > EpsilonMin)
        {
            // Decai epsilon multiplicando pelo fator de decaimento, mas não abaixo do mínimo
            Epsilon = Mathf.Max(EpsilonMin, Epsilon * EpsilonDecay); 
        }
    }

    /// <summary>
    /// Treina a rede neural usando uma transição (estado, ação, recompensa, próximo estado) e 
    /// ajusta os pesos com base na diferença entre o valor Q atual e o valor Q alvo.
    /// </summary>
    public float TreinarPassoQLearning(Transicao transicao, float taxaAprendizado = 0.005f)
    {
        Matriz qAtual = ForwardPass(transicao.Estado, aplicarSoftmax: false); // Passagem para frente para obter os valores Q atuais
        Matriz qAlvo = qAtual.Clonar(); // Cria uma cópia dos valores Q atuais para atualizar o valor da ação tomada

        float novoQParaAcao;

        if (transicao.Terminado)
        {
            // Se o episódio terminou, o valor Q alvo é apenas a recompensa recebida
            novoQParaAcao = transicao.Recompensa;
        }
        else
        {
            Matriz qProximo = ForwardPass(transicao.ProximoEstado, aplicarSoftmax: false); // Passagem para frente para obter os valores Q do próximo estado
            float maxQProximo = qProximo.Dados[0, 0];

            // Encontrar o valor máximo Q do próximo estado
            for (int i = 1; i < SaidaTamanho; i++)
            {   
                if (qProximo.Dados[i, 0] > maxQProximo)
                {
                    maxQProximo = qProximo.Dados[i, 0]; // Atualiza o valor máximo Q do próximo estado
                }
            }
            
            novoQParaAcao = transicao.Recompensa + (Gamma * maxQProximo); // Calcula o valor Q alvo usando a equação de Bellman
        }

        // Limita o valor Q alvo para evitar valores extremos que possam desestabilizar o treinamento
        novoQParaAcao = Mathf.Clamp(novoQParaAcao, -20.0f, 20.0f);

        float qAntigoAcao = qAtual.Dados[transicao.Acao, 0]; // Valor Q antigo para a ação tomada
        float tdLoss = Mathf.Pow(novoQParaAcao - qAntigoAcao, 2f); // Calcula a perda de diferença temporal (TD Loss)

        qAlvo.Dados[transicao.Acao, 0] = novoQParaAcao; // Atualiza o valor Q alvo para a ação tomada

        ForwardPass(transicao.Estado, aplicarSoftmax: false); // Passagem para frente novamente para atualizar os valores Q atuais
        BackwardPass(qAlvo, taxaAprendizado); // Retropropagação para ajustar os pesos da rede neural com base no valor Q alvo

        return tdLoss; // Retorna a perda de diferença temporal para monitoramento do treinamento
    }

    /// <summary>
    /// Treina a rede neural usando uma trajetória completa de transições (episódio)
    /// e ajusta os pesos com base nos retornos normalizados (G) para cada passo.
    /// </summary>
    public void TreinarEpisodioReinforce(List<Transicao> trajetoria, float taxaAprendizado = 0.1f)
    {
        if (trajetoria == null || trajetoria.Count == 0)
        {
            Debug.LogWarning("A trajetória está vazia. Nenhum treinamento será realizado.");
            return;
        }

        int passos = trajetoria.Count; // Número de passos na trajetória
        float[] retornos = new float[passos]; // Array para armazenar os retornos (G) para cada passo
        float gAcumulado = 0.0f; // Inicializa o retorno acumulado

        for (int t = passos - 1; t >= 0; t--)
        {
            gAcumulado = trajetoria[t].Recompensa + (Gamma * gAcumulado); // Calcula o retorno acumulado G
            retornos[t] = gAcumulado; // Armazena o retorno acumulado para o passo t
        }

        float media = 0.0f; // Inicializa a média dos retornos

        for (int i = 0; i < passos; i++)
        {
            media += retornos[i]; // Soma todos os retornos
        }
        media /= passos; // Calcula a média dos retornos

        float variancia = 0.0f; // Inicializa a variância dos retornos
        for (int i = 0; i < passos; i++)
        {
            variancia += Mathf.Pow(retornos[i] - media, 2f); // Soma os quadrados das diferenças em relação à média
        }
        
        float desvioPadrao = Mathf.Sqrt(variancia / passos) + 1e-7f; // Calcula o desvio padrão dos retornos

        for (int i = 0; i < passos; i++)
        {
            retornos[i] = (retornos[i] - media) / desvioPadrao; // Normaliza os retornos para estabilizar o treinamento
        }

        for (int t = 0; t < passos; t++)
        {
            Transicao transicao = trajetoria[t]; // Obtém a transição atual
            Matriz probabilidades = ForwardPass(transicao.Estado, aplicarSoftmax: false); // Passagem para frente para obter as probabilidades das ações

            Matriz alvo = probabilidades.Clonar(); // Cria uma cópia das probabilidades para atualizar o alvo
            float probAcao = probabilidades.Dados[transicao.Acao, 0]; // Obtém a probabilidade da ação tomada

            // Atualiza a probabilidade da ação tomada com base no retorno normalizado
            alvo.Dados[transicao.Acao, 0] = Mathf.Clamp(probAcao + (retornos[t] * 0.1f), 0.001f, 0.999f); 

            BackwardPass(alvo, taxaAprendizado); // Retropropagação para ajustar os pesos da rede neural com base no alvo atualizado
        }
    }
}