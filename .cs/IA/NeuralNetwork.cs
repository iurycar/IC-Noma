using System.Collections.Generic;
using System.IO;
using System;
using UnityEngine;

[Serializable]
public class LinhaMatriz
{
    public float[] valores; // Representa uma linha da matriz como um array de floats
}

[Serializable]
public class MatrizContainer
{
    public List<LinhaMatriz> linhas = new List<LinhaMatriz>(); // Lista de linhas representando os pesos 1
}

[Serializable]
public class VetorContainer
{
    public float[] valores; // Representa um vetor como um array de floats
}

[Serializable]
public class RedeParametrosDTO
{
    public int entradaTamanho;
    public int ocultaTamanho;
    public int ocultaQuantidade;
    public int saidaTamanho;
    public List<MatrizContainer> pesos = new List<MatrizContainer>(); // Contêiner para os pesos da rede neural
    public List<VetorContainer> bias = new List<VetorContainer>(); // Contêiner para os bias da rede neural
}

public class NeuralNetwork
{
    public int EntradaTamanho { get; protected set; } // Tamanho da entrada da rede neural
    public int OcultaTamanho { get; protected set; } // Tamanho da camada oculta da rede neural
    public int OcultaQuantidade { get; protected set; } // Quantidade de camadas ocultas na rede neural
    public int SaidaTamanho { get; protected set; } // Tamanho da saída da rede neural

    public List<Matriz> Pesos { get; protected set; } // Lista de matrizes representando os pesos da rede neural
    public List<Matriz> Bias { get; protected set; } // Lista de matrizes representando os bias da rede neural

    public Matriz Estado { get; protected set; } // Estado da rede neural
    public List<Matriz> Z { get; protected set; } // Lista de matrizes representando os valores Z (pré-ativação) da rede neural
    public List<Matriz> A { get; protected set; } // Lista de matrizes representando os valores A (ativação) da rede neural

    /// <summary>
    /// Construtor da rede neural. Inicializa os pesos e bias da rede neural.
    /// <param name="caminhoArquivo">Caminho do arquivo JSON para carregar os pesos e bias da rede neural. Se for nulo, inicializa pesos aleatórios.</param>
    /// <param name="entradaTamanho">Número de neurônios na camada de entrada.</param>
    /// <param name="ocultaTamanho">Número de neurônios na camada oculta.</param>
    /// <param name="ocultaQuantidade">Número de camadas ocultas.</param>
    /// <param name="saidaTamanho">Número de neurônios na camada de saída.</param>
    /// </summary>
    public NeuralNetwork(string caminhoArquivo = null, int entradaTamanho = 3, int ocultaTamanho = 2, int ocultaQuantidade = 1, int saidaTamanho = 3)
    {
        EntradaTamanho = entradaTamanho;
        OcultaTamanho = ocultaTamanho;
        OcultaQuantidade = ocultaQuantidade;
        SaidaTamanho = saidaTamanho;

        Pesos = new List<Matriz>();
        Bias = new List<Matriz>();
        Z = new List<Matriz>();
        A = new List<Matriz>();

        if (string.IsNullOrEmpty(caminhoArquivo))
        {
            InicializarPesos(entradaTamanho, ocultaTamanho, ocultaQuantidade, saidaTamanho);
        }
        else
        {
            CarregarPesos(caminhoArquivo);
        }
    }

    /// <summary>
    /// Passagem para frente: calcula as ativações de cada camada.
    /// </summary>
    public virtual Matriz ForwardPass(Matriz estado, bool aplicarSoftmax = true)
    {
        Estado = estado;
        Z.Clear();
        A.Clear();

        Matriz entradaAtual = estado;
        int numCamadas = Pesos.Count;

        for (int camada = 0; camada < numCamadas; camada++)
        {
            // Lógica para cada camada oculta
            Matriz zAtual = Matriz.Somar(Matriz.Multiplicar(Pesos[camada], entradaAtual), Bias[camada]);
            Z.Add(zAtual);

            Matriz aAtual;
            bool isUltimaCamada = (camada == numCamadas - 1);

            if (!isUltimaCamada)
            {
                aAtual = Relu(zAtual); // Ativação ReLU para camadas ocultas
            }
            else
            {
                aAtual = aplicarSoftmax ? Softmax(zAtual) : zAtual.Clonar(); // Softmax ou identidade para a camada de saída
            }

            A.Add(aAtual);
            entradaAtual = aAtual; // A saída da camada atual se torna a entrada para a próxima camada
        }

        return A[A.Count - 1]; // Retorna a ativação da última camada (saída)
    }


    /// <summary>
    /// Retropropagação: calcula os gradientes e ajusta os pesos e vieses.
    /// </summary>
    public virtual void BackwardPass(Matriz saidaEsperada, float taxaAprendizado = 0.1f)
    {
        // Gradiente da camada de saída
        int numCamadas = Pesos.Count;
        if (numCamadas == 0 || A.Count != numCamadas) return;

        Matriz delta = Matriz.Subtrair(A[numCamadas - 1], saidaEsperada);

        for (int linha = 0; linha < delta.Linhas; linha++)
        {
            for (int coluna = 0; coluna < delta.Colunas; coluna++)
            {
                // Limita o gradiente para evitar explosão de gradientes
                delta.Dados[linha, coluna] = Mathf.Clamp(delta.Dados[linha, coluna], -2.0f, 2.0f); 
            }
        }

        for (int camada = numCamadas - 1; camada >= 0; camada--)
        {
            Matriz ativacaoAnterior = (camada > 0) ? A[camada - 1] : Estado;

            Matriz deltaW = Matriz.Multiplicar(delta, ativacaoAnterior.Transposta());
            Matriz deltaB = delta.Clonar();

            if (camada > 0)
            {
                Matriz deltaA = Matriz.Multiplicar(Pesos[camada].Transposta(), delta);
                delta = Matriz.MultiplicarElemento(deltaA, ReluDerivative(Z[camada - 1]));
            }

            Pesos[camada] = Matriz.Subtrair(Pesos[camada], Matriz.Escalar(deltaW, taxaAprendizado));
            Bias[camada] = Matriz.Subtrair(Bias[camada], Matriz.Escalar(deltaB, taxaAprendizado));
        }
    }


    public Matriz Relu(Matriz x)
    {
        Matriz resultado = new Matriz(x.Linhas, x.Colunas);

        for (int linha = 0; linha < x.Linhas; linha++)
        {
            for (int coluna = 0; coluna < x.Colunas; coluna++)
            {
                resultado.Dados[linha, coluna] = Mathf.Max(0f, x.Dados[linha, coluna]); // Aplica a função ReLU: retorna o valor se for positivo, caso contrário retorna 0
            }
        }

        return resultado;
    }

    // Método para calcular a derivada da função ReLU
    public Matriz ReluDerivative(Matriz x)
    {
        Matriz resultado = new Matriz(x.Linhas, x.Colunas);

        for (int linha = 0; linha < x.Linhas; linha++)
        {
            for (int coluna = 0; coluna < x.Colunas; coluna++)
            {
                if (x.Dados[linha, coluna] > 0.0f)
                    resultado.Dados[linha, coluna] = 1.0f; // Derivada da função ReLU é 1 para valores positivos
                else
                    resultado.Dados[linha, coluna] = 0.0f; // Derivada da função ReLU é 0 para valores negativos ou iguais a zero
            }
        }

        return resultado;
    }

    // Método para calcular a função Softmax
    public Matriz Softmax(Matriz x)
    {
        Matriz resultado = new Matriz(x.Linhas, x.Colunas);
        float maxValor = float.MinValue;

        for (int linha = 0; linha < x.Linhas; linha++)
        {
            if (x.Dados[linha, 0] > maxValor)
                maxValor = x.Dados[linha, 0]; // Encontra o valor máximo para estabilidade numérica
        }

        float somaExp = 0f;
        for (int linha = 0; linha < x.Linhas; linha++)
        {
            resultado.Dados[linha, 0] = Mathf.Exp(x.Dados[linha, 0] - maxValor); // Calcula a exponencial do valor ajustado
            somaExp += resultado.Dados[linha, 0]; // Soma das exponenciais para normalização
        }
        
        if (somaExp > 0f)
        {
            for (int linha = 0; linha < x.Linhas; linha++)
                resultado.Dados[linha, 0] /= somaExp; // Normaliza os valores para obter probabilidades
        }

        return resultado;
    }


    private void InicializarPesos(int entrada, int oculta, int qtdOculta, int saida)
    {
        Pesos.Clear();
        Bias.Clear();

        // Caso direto entrada -> saída (sem camadas ocultas)
        if (qtdOculta <= 0)
        {
            Pesos.Add(Matriz.Aleatoria(saida, entrada));
            Bias.Add(Matriz.Zeros(saida, 1));
            return;
        }
        
        // Camada 1 - Entrada -> Primeira Oculta
        Pesos.Add(Matriz.Aleatoria(oculta, entrada));
        Bias.Add(Matriz.Zeros(oculta, 1));

        // Camadas ocultas intermediárias - Oculta -> Oculta
        for (int i = 0; i < qtdOculta - 1; i++)
        {
            Pesos.Add(Matriz.Aleatoria(oculta, oculta));
            Bias.Add(Matriz.Zeros(oculta, 1));
        }

        // Última camada - Última Oculta -> Saída
        Pesos.Add(Matriz.Aleatoria(saida, oculta));
        Bias.Add(Matriz.Zeros(saida, 1));
    }

    // Método para carregar os pesos e bias da rede neural a partir de um arquivo JSON
    public void CarregarPesos(string caminhoArquivo)
    {
        if (!File.Exists(caminhoArquivo))
        {
            throw new FileNotFoundException($"O arquivo {caminhoArquivo} não foi encontrado.");
        }

        string jsonTexto = File.ReadAllText(caminhoArquivo);
        RedeParametrosDTO dados = JsonUtility.FromJson<RedeParametrosDTO>(jsonTexto);

        EntradaTamanho = dados.entradaTamanho;
        OcultaTamanho = dados.ocultaTamanho;
        OcultaQuantidade = dados.ocultaQuantidade;
        SaidaTamanho = dados.saidaTamanho;

        Pesos.Clear();
        Bias.Clear();

        for (int camada = 0; camada < dados.pesos.Count; camada++)
        {
            Pesos.Add(ConverterListaParaMatriz(dados.pesos[camada].linhas));
            Bias.Add(ConverterVetorParaMatrizColuna(dados.bias[camada].valores));
        }
    }

    public void SalvarPesosParaJson(string caminhoArquivo)
    {
        var dto = new RedeParametrosDTO
        {
            entradaTamanho = EntradaTamanho,
            ocultaTamanho = OcultaTamanho,
            ocultaQuantidade = OcultaQuantidade,
            saidaTamanho = SaidaTamanho,
        };

        for (int camada = 0; camada < Pesos.Count; camada++)
        {
            dto.pesos.Add(new MatrizContainer { linhas = MatrizParaLinhas(Pesos[camada]) });
            dto.bias.Add(new VetorContainer { valores = MatrizParaVetor(Bias[camada]) });
        }

        string json = JsonUtility.ToJson(dto, prettyPrint: true);
        File.WriteAllText(caminhoArquivo, json);
    }

    // Método auxiliar para converter lista de linhas de floats em Matriz
    private Matriz ConverterListaParaMatriz(List<LinhaMatriz> linhas)
    {
        int numLinhas = linhas.Count;
        int numColunas = linhas[0].valores.Length;
        Matriz matriz = new Matriz(numLinhas, numColunas);

        for (int linha = 0; linha < numLinhas; linha++)
        {
            for (int camada = 0; camada < numColunas; camada++)
            {
                matriz.Dados[linha, camada] = linhas[linha].valores[camada];
            }
        }

        return matriz;
    }

    // Método auxiliar para converter vetor de floats em Matriz de uma única coluna
    private Matriz ConverterVetorParaMatrizColuna(float[] vetor)
    {
        Matriz coluna = new Matriz(vetor.Length, 1);
        for (int linha = 0; linha < vetor.Length; linha++)
        {
            coluna.Dados[linha, 0] = vetor[linha];
        }
        return coluna;
    }

    // Método auxiliar para converter Matriz em lista de linhas de floats
    private List<LinhaMatriz> MatrizParaLinhas(Matriz matriz)
    {
        var linhas = new List<LinhaMatriz>();

        for (int linha = 0; linha < matriz.Linhas; linha++)
        {
            float[] valoresLinha = new float[matriz.Colunas]; // Cria um array para armazenar os valores da linha atual

            for (int coluna = 0; coluna < matriz.Colunas; coluna++)
            {
                valoresLinha[coluna] = matriz.Dados[linha, coluna]; // Armazena no array
            }
            linhas.Add(new LinhaMatriz { valores = valoresLinha }); // Adiciona a linha à lista de linhas
        }

        return linhas;
    }

    // Método auxiliar para converter Matriz de uma única coluna em vetor de floats
    private float[] MatrizParaVetor(Matriz coluna)
    {
        float[] vetor = new float[coluna.Linhas];

        for (int linha = 0; linha < coluna.Linhas; linha++)
        {
            vetor[linha] = coluna.Dados[linha, 0]; // Converte o valor para float e armazena no vetor
        }

        return vetor;
    }
}