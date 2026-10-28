using System;
using System.IO;
using System.Globalization;
using System.Collections.Generic;
using UnityEngine;

public class SupervisedLearning : NeuralNetwork
{
    public struct AmostraTreino
    {
        public Matriz Entrada;
        public Matriz SaidaEsperada;
    }

    // Construtor da classe SupervisedLearning que herda de NeuralNetwork
    public SupervisedLearning(string caminhoArquivo = null, int entradaTamanho = 3, int ocultaTamanho = 2, int ocultaQuantidade = 1, int saidaTamanho = 3)
        : base(caminhoArquivo, entradaTamanho, ocultaTamanho, ocultaQuantidade, saidaTamanho)
    {
    }

    /// <summary>
    /// Carrega amostras de treino a partir de um arquivo CSV.
    /// </summary>
    /// <param name="caminhoCSV">Caminho do arquivo CSV contendo as amostras de treino.</param>
    /// <param name="alvoInimigo"> Indica se a saída esperada deve ser baseada na ação do inimigo (true) ou do jogador (false).</param>
    /// <returns>Lista de amostras de treino carregadas.</returns>
    public List<AmostraTreino> CarregarAmostrasTreino(string caminhoCSV, bool alvoInimigo = true)
    {
        var amostras = new List<AmostraTreino>();

        if (!File.Exists(caminhoCSV))
        {
            Debug.LogError($"Arquivo não encontrado: {caminhoCSV}");
            return amostras;
        }

        string[] linhas = File.ReadAllLines(caminhoCSV);
        var inv = CultureInfo.InvariantCulture; // Usar CultureInfo.InvariantCulture para garantir a leitura correta de números com ponto decimal

        for (int i = 1; i < linhas.Length; i++)
        {
            // Ignora linhas vazias
            string linha = linhas[i].Trim();
            if (string.IsNullOrEmpty(linha)) continue;

            // Divide a linha em colunas usando vírgula como separador
            char separador = linha.Contains(";") ? ';' : ',';
            string[] colunas = linha.Split(separador);

            // Cria o vetor de entrada a partir das colunas do arquivo CSV
            float[] entradaVetor = new float[EntradaTamanho];
            for (int e = 0; e < EntradaTamanho; e++)
            {
                entradaVetor[e] = float.Parse(colunas[e], inv);
            }

            // Pega a saída esperada (InimigoAcaoId índice 30, PlayerAcaoId índice 32)
            int indiceAcao = alvoInimigo ? int.Parse(colunas[30], inv) : int.Parse(colunas[32], inv);

            // Monta o vetor One-Hot, ou seja, um vetor com todos os elementos zero, exceto o índice da ação que será 1
            Matriz saidaOneHot = Matriz.Zeros(SaidaTamanho, 1);
            if (indiceAcao >= 0 && indiceAcao < SaidaTamanho)
            {
                saidaOneHot.Dados[indiceAcao, 0] = 1.0f;
            }

            amostras.Add(new AmostraTreino
            {
                Entrada = Matriz.DeVetorColuna(entradaVetor),
                SaidaEsperada = saidaOneHot
            });
        }

        Debug.Log($"Dataset carregado com {amostras.Count} amostras válidas.");
        return amostras;
    }


    /// <summary>
    /// Treina a rede neural usando o dataset fornecido.
    /// </summary>
    /// <param name="dataset">Lista de amostras de treino.</param>
    /// <param name="epochs">Número de épocas de treinamento.</param>
    /// <param name="taxaAprendizado">Taxa de aprendizado para o ajuste dos pesos da rede neural.</param>
    public void Treinar(List<AmostraTreino> dataset, int epochs, float taxaAprendizado = 0.1f)
    {
        if (dataset == null || dataset.Count == 0)
        {
            Debug.LogError("Dataset vazio ou nulo. Não é possível treinar a rede neural.");
            return;
        }

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float somaErroQuadratico = 0.0f;

            // Embaralha o dataset para cada época de treinamento
            // Usando o algoritmo de Fisher-Yates para embaralhar a lista de amostras
            for (int i = dataset.Count - 1; i > 0; i--)
            {
                int random = UnityEngine.Random.Range(0, i + 1);
                var temp = dataset[i];
                dataset[i] = dataset[random];
                dataset[random] = temp;
            }

            for (int i = 0; i < dataset.Count; i++)
            {
                Matriz saidaReal = ForwardPass(dataset[i].Entrada, aplicarSoftmax: true);
                BackwardPass(dataset[i].SaidaEsperada, taxaAprendizado);

                for (int saida = 0; saida < SaidaTamanho; saida++)
                {
                    float erro = dataset[i].SaidaEsperada.Dados[saida, 0] - saidaReal.Dados[saida, 0];
                    somaErroQuadratico += erro * erro;
                }
            }

            if ((epoch + 1) % 50 == 0 || epoch == 0)
            {
                float erroMedio = somaErroQuadratico / (dataset.Count * SaidaTamanho);
                Debug.Log($"Época {epoch + 1}/{epochs} - Erro médio: {erroMedio:F5}");
            }
        }
    }
}