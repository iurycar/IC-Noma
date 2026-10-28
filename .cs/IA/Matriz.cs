using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

[Serializable]
public class Matriz
{
    public int Linhas { get; }
    public int Colunas { get; }
    public float[,] Dados { get; }

    public Matriz(int linhas, int colunas)
    {
        Linhas = linhas;
        Colunas = colunas;
        Dados = new float[linhas, colunas];
    }

    // Método para multiplicação de matrizes
    public static Matriz Multiplicar(Matriz a, Matriz b)
    {
        if (a.Colunas != b.Linhas)
        {
            throw new InvalidOperationException($"Dimensões incompatíveis: ({a.Linhas}x{a.Colunas}) e ({b.Linhas}x{b.Colunas})");
        }

        Matriz resultado = new Matriz(a.Linhas, b.Colunas);

        for (int i = 0; i < a.Linhas; i++)
        {
            for (int j = 0; j < b.Colunas; j++)
            {
                float soma = 0f;
                for (int k = 0; k < a.Colunas; k++)
                {
                    soma += a.Dados[i, k] * b.Dados[k, j];
                }

                resultado.Dados[i, j] = soma;
            }
        }

        return resultado;
    }

    // Método para soma de matrizes
    public static Matriz Somar(Matriz a, Matriz b)
    {
        Matriz resultado = new Matriz(a.Linhas, a.Colunas);

        for (int i = 0; i < a.Linhas; i++)
        {
            for (int j = 0; j < a.Colunas; j++)
            {
                resultado.Dados[i, j] = a.Dados[i, j] + b.Dados[i, j];
            }
        }

        return resultado;
    }

    // Método para subtração de matrizes
    public static Matriz Subtrair(Matriz a, Matriz b)
    {
        Matriz resultado = new Matriz(a.Linhas, a.Colunas);

        for (int i = 0; i < a.Linhas; i++)
        {
            for (int j = 0; j < a.Colunas; j++)
            {
                resultado.Dados[i, j] = a.Dados[i, j] - b.Dados[i, j];
            }
        }

        return resultado;
    }

    // Método para multiplicação elemento a elemento (Hadamard product)
    public static Matriz MultiplicarElemento(Matriz a, Matriz b)
    {
        Matriz resultado = new Matriz(a.Linhas, a.Colunas);

        for (int i = 0; i < a.Linhas; i++)
        {
            for (int j = 0; j < a.Colunas; j++)
            {
                resultado.Dados[i, j] = a.Dados[i, j] * b.Dados[i, j];
            }
        }

        return resultado;
    }

    // Método para multiplicação de uma matriz por um escalar
    public static Matriz Escalar(Matriz a, float escalar)
    {
        Matriz resultado = new Matriz(a.Linhas, a.Colunas);

        for (int i = 0; i < a.Linhas; i++)
        {
            for (int j = 0; j < a.Colunas; j++)
            {
                resultado.Dados[i, j] = a.Dados[i, j] * escalar;
            }
        }

        return resultado;
    }

    // Método para transposição de uma matriz
    public Matriz Transposta()
    {
        Matriz resultado = new Matriz(Colunas, Linhas);

        for (int i = 0; i < Linhas; i++)
        {
            for (int j = 0; j < Colunas; j++)
            {
                resultado.Dados[j, i] = Dados[i, j];
            }
        }

        return resultado;
    }

    public Matriz Clonar()
    {
        Matriz clone = new Matriz(Linhas, Colunas);

        Array.Copy(Dados, clone.Dados, Dados.Length);

        return clone;
    }

    public static Matriz Aleatoria(int linhas, int colunas)
    {
        Matriz resultado = new Matriz(linhas, colunas);

        for (int i = 0; i < linhas; i++)
        {
            for (int j = 0; j < colunas; j++)
            {
                resultado.Dados[i, j] = UnityEngine.Random.Range(-1f, 1f);
            }
        }

        return resultado;
    }

    public static Matriz Zeros(int linhas, int colunas)
    {
        return new Matriz(linhas, colunas);
    }

    public static Matriz DeVetorColuna(float[] vetor)
    {
        int tamanho = vetor.Length;
        Matriz resultado = new Matriz(tamanho, 1);

        for (int i = 0; i < tamanho; i++)
        {
            resultado.Dados[i, 0] = vetor[i];
        }

        return resultado;
    }
}