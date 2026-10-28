using UnityEngine;
using System.IO;
using System.Collections;
using System.Text;
using System.Globalization;
using TMPro;

public class EnemyAI : MonoBehaviour
{
    public enum Acao
    {
        MoverMaisX = 0,
        MoverMenosX = 1,
        MoverMaisY = 2,
        MoverMenosY = 3,
        MoverDiagonalMaisXMaisY = 4,
        MoverDiagonalMaisXMenosY = 5,
        MoverDiagonalMenosXMaisY = 6,
        MoverDiagonalMenosXMenosY = 7,
        AtacarCorpoACorpo = 8,
        AtacarEmArea = 9,
        AtacarDistancia = 10,
        AtacarEspecial = 11,
        Curar = 12,
        //AtacarFogoFatuo = 12,
        //Defender = 13,
        Max = 13
    }

    [Header("Ação Atual")]
    [SerializeField] private Acao _acaoAtual = Acao.MoverMaisX;

    [Header("Texto de Visualização")]
    [SerializeField] private TMP_Text _textoAcaoAtual;
    [SerializeField] private TMP_Text _textoRecompensaAtual;

    [Header("Movimento")]
    [SerializeField] private float _velocidadePadrao = 0.8f;
    [SerializeField] private float _raioDeteccao = 50f;
    [SerializeField] private float _intervaloDecisao = 2.0f;
    [SerializeField] private float _duracaoAcaoMovimento = 0.4f;
    [SerializeField] private float _duracaoAcaoAtaque = 0.8f;

    [Header("Curar e Defender")]
    [SerializeField] private int _quantidadeCura = 15;
    private bool _curou = false;

    [Header("Debug & Telemetria")]
    [SerializeField] private bool _ativarLogs = true;
    [SerializeField] private float _intervaloLogQValores = 1.0f;
    private float _proximoLogQTempo;

    [Header("Métricas de Desempenho RL")]
    [SerializeField] private bool _exportarMetricasCSV = true;
    [SerializeField] private float _intervaloRelatorioMetricas = 20f;
    private float _proximoRelatorioTempo;
    private string _caminhoArquivoCSV = "";

    // Métricas do Episódio / Janela
    private float _recompensaEpisodioTotal = 0f;
    private int _passosEpisodioTotal = 0;
    private float _somaLossEpisodio = 0f;
    private int _passosTreinoEpisodio = 0;
    private int _qtdExploracoes = 0;
    private int _qtdExplotacoes = 0;
    private float _tempoInicioEpisodio = 0f;

    private float _recompensaJanelaAtual = 0f;
    private int _passosJanelaAtual = 0;
    private float _somaLossJanela = 0f;

    [Header("Machine Learning")]
    private bool _iaHabilitada = true;
    private ReinforcementLearning _brain;
    private Matriz _ultimoEstadoAmbiente;
    private int _ultimaAcao = -1;
    private float _distanciaAnterior;
    private Vector2 _posicaoFisicaAnterior;

    // Acumula recompensas de eventos físicos (como colisões, ataques bem-sucedidos, etc.).
    private float _recompensaAcumuladaEventos = 0f;

    private EnemyPathfinding _inimigoEncontrarCaminho;
    private EnemyCombat _enemyCombat;
    private EnemyHealth _enemyHealth;
    private Animator _inimigoAnimator;

    private float _velocidadeAtual;
    private float _proximaDecisaoTempo;

    private void Awake()
    {
        VerificarModoOperacao();

        if (!_iaHabilitada) return;

        _inimigoEncontrarCaminho = GetComponent<EnemyPathfinding>();
        _enemyCombat = GetComponent<EnemyCombat>();
        _enemyHealth = GetComponent<EnemyHealth>();
        _inimigoAnimator = GetComponent<Animator>();

        if (_textoAcaoAtual == null)
        {
            GameObject obj = GameObject.Find("Acao");
            if (obj != null)
            {
                _textoAcaoAtual = obj.GetComponent<TMP_Text>();
            }
        }

        if (_textoRecompensaAtual == null)
        {
            GameObject obj = GameObject.Find("Recompensa");
            if (obj != null)
            {
                _textoRecompensaAtual = obj.GetComponent<TMP_Text>();
            }
        }

        _velocidadeAtual = _velocidadePadrao;

        InicializarLoggerCSV();
    }

    private void Start()
    {
        VerificarModoOperacao();

        if (!_iaHabilitada) return;

        string caminho = SaveManager.Instance.ObterCaminhoPesos;

        // Cria o objeto "Cerebro", responsável por decidir as ações do inimigo com base no estado do ambiente
        // Usando Reinforcement Learning (Aprendizado por Reforço) com Q-Learning
        _brain = new ReinforcementLearning(
            caminhoArquivo: (caminho != null && File.Exists(caminho)) ? caminho : null,
            entradaTamanho: 30,
            ocultaTamanho: 64,
            ocultaQuantidade: 4,
            saidaTamanho: (int)Acao.Max
        );

        /*if (_ativarLogs)
        {
            Debug.Log($"<color=cyan>[RL Init]</color> Rede instanciada com sucesso!" +
                      $"\n- W1: {_brain.Pesos[0].Dados.Linhas}x{_brain.Pesos[0].Dados.Colunas}" +
                      $"\n- W2: {_brain.Pesos[1].Dados.Linhas}x{_brain.Pesos[1].Dados.Colunas}" +
                      $"\n- ..." +
                      $"\n- Epsilon inicial: {_brain.Epsilon}");
        }*/

        _tempoInicioEpisodio = Time.time;
        _proximoRelatorioTempo = Time.time + _intervaloRelatorioMetricas;

        StartCoroutine(CicloComportamento());
    }

    private void Update()
    {
        if (!_iaHabilitada || (_enemyHealth != null && _enemyHealth.EstaMorto)) 
            return;

        if (Time.time >= _proximoRelatorioTempo)
        {
            _proximoRelatorioTempo = Time.time + _intervaloRelatorioMetricas;
            ExibirRelatorioMetricas(finalDeEpisodio: false);
        }

        if (Time.time < _proximaDecisaoTempo) return;

        _proximaDecisaoTempo = Time.time + _intervaloDecisao;
        TomarDecisao();
    }


    private void VerificarModoOperacao()
    {
        if (GameMaster.Instance != null && GameMaster.Instance.SegundoJogador)
        {
            _iaHabilitada = false;
            enabled = false;
            PararMovimento();
            return;
        }

        if (GameMaster.Instance != null)
        {
            _iaHabilitada = GameMaster.Instance.LigarIAInimigos;
            if (!_iaHabilitada)
            {
                enabled = false;
                PararMovimento();
            }
        }

        if (GameMaster.Instance != null && GameMaster.Instance.HabilitarDebug)
        {
            _ativarLogs = true;
        }
    }


    public void RegistrarRecompensaEvento(float recompensa)
    {
        _recompensaAcumuladaEventos += recompensa;
    }

    private void TomarDecisao()
    {
        if (_enemyCombat != null && _enemyCombat.EstaAtacando())
        {
            return;
        }

        if (PlayerController.Instance == null)
        {
            ExecutarPatrulha();
            return;
        }

        Knockback knockback = GetComponent<Knockback>();
        if (knockback != null && knockback.EstarSendoEmpurrado)
        {
            return;
        }

        float distanciaParaJogador = Vector2.Distance(transform.position, PlayerController.Instance.GetPosicao());

        // Fora da área de detecção, patrulha aleatoriamente
        if (distanciaParaJogador > _raioDeteccao)
        {
            ExecutarPatrulha();
            return;
        }

        // Aplica o aprendizado por reforço da ação do passo anterior
        if (_ultimoEstadoAmbiente != null && _ultimaAcao >= 0)
        {
            AplicarRecompensaEPassoRL(distanciaParaJogador);
        }

        // Coleta o estado detalhado do ambiente atual
        _ultimoEstadoAmbiente = SaveManager.Instance != null 
            ? SaveManager.Instance.ColetarSituacaoAtual() 
            : Matriz.Zeros(30, 1);

        _distanciaAnterior = distanciaParaJogador;
        _posicaoFisicaAnterior = transform.position;

        if (Random.value < _brain.Epsilon)
        {
            _ultimaAcao = Random.Range(0, _brain.SaidaTamanho);
            _qtdExploracoes++;
        }
        else
        {
            Matriz qValores = _brain.ForwardPass(_ultimoEstadoAmbiente, aplicarSoftmax: false);
            int melhorAcao = 0;
            float melhorValor = qValores.Dados[0, 0];

            for (int i = 1; i < _brain.SaidaTamanho; i++)
            {
                if (qValores.Dados[i, 0] > melhorValor)
                {
                    melhorValor = qValores.Dados[i, 0];
                    melhorAcao = i;
                }
            }
            _ultimaAcao = melhorAcao;
            _qtdExplotacoes++;
        }

        // Escolhe a microação pela política Epsilon-Greedy (E-greedy)
        // _ultimaAcao = _brain.EscolherAcaoEGreedy(_ultimoEstadoAmbiente);
        _acaoAtual = (Acao)_ultimaAcao;
        _passosEpisodioTotal++;

        // Log periódico dos valores Q para inspecionar saturação ou NaNs
        if (_ativarLogs && Time.time >= _proximoLogQTempo)
        {
            _proximoLogQTempo = Time.time + _intervaloLogQValores;
            LogDiagnosticoQValores();
        }

        ExecutarAcao(_acaoAtual);
    }


    private void ExecutarAcao(Acao acao)
    {
        Vector2 direcao = Vector2.zero;

        switch (acao)
        {
            // Movimento nos eixos X e Y
            case Acao.MoverMaisX:
                direcao = Vector2.right;
                break;
            case Acao.MoverMenosX:
                direcao = Vector2.left;
                break;
            case Acao.MoverMaisY:
                direcao = Vector2.up;
                break;
            case Acao.MoverMenosY:
                direcao = Vector2.down;
                break;
            
            // Movimentos diagonais
            case Acao.MoverDiagonalMaisXMaisY:
                direcao = new Vector2(1f, 1f).normalized;
                break;
            case Acao.MoverDiagonalMaisXMenosY:
                direcao = new Vector2(1f, -1f).normalized;
                break;
            case Acao.MoverDiagonalMenosXMaisY:
                direcao = new Vector2(-1f, 1f).normalized;
                break;
            case Acao.MoverDiagonalMenosXMenosY:
                direcao = new Vector2(-1f, -1f).normalized;
                break;

            // Ataques e Ações Estáticas
            case Acao.AtacarCorpoACorpo:
            case Acao.AtacarEmArea:
            case Acao.AtacarDistancia:
            case Acao.AtacarEspecial:
            //case Acao.AtacarFogoFatuo:
                PararMovimento();

                if (_enemyCombat != null)
                {
                    bool sucesso = _enemyCombat.Atacar(tipoAtaque: (int)acao);

                    if (!sucesso)
                    {
                        float loss = _brain.TreinarPassoQLearning(new ReinforcementLearning.Transicao
                        {
                            Estado = _ultimoEstadoAmbiente,
                            Acao = _ultimaAcao,
                            Recompensa = -0.4f, // Penalidade por falhar no ataque
                            ProximoEstado = _ultimoEstadoAmbiente,
                            Terminado = false
                        }, taxaAprendizado: 0.005f);

                        RegistrarMetricaPasso(-0.4f, loss);
                        _proximaDecisaoTempo = Time.time + 0.4f;
                    }
                    else
                    {
                        _proximaDecisaoTempo = Time.time + _duracaoAcaoAtaque;;
                    }
                }
                return;

            case Acao.Curar:
                PararMovimento();

                if (_enemyHealth != null)
                {
                    bool sucesso = _enemyHealth.Curar(quantidade: _quantidadeCura);
                    _proximaDecisaoTempo = Time.time + 2.5f;
                    
                    if (sucesso)
                    {
                        _curou = true;
                    }
                    else
                    {
                        _curou = false;
                    }
                }

                break;
            //case Acao.Defender:
            
                //return;

        }

        if (direcao != Vector2.zero && _inimigoEncontrarCaminho != null)
        {
            _velocidadeAtual = _velocidadePadrao;
            AtualizarAnimacao(direcao);
            _inimigoEncontrarCaminho.MoverPara(direcao, _velocidadeAtual);
            _proximaDecisaoTempo = Time.time + _duracaoAcaoMovimento;
        }
    }

    private void AplicarRecompensaEPassoRL(float distanciaParaJogador)
    {
        float recompensa = 0f;
        
        // Diferença positiva = encurtou a distância (aproximou); negativa = aumentou (afastou)
        float aproximacao = _distanciaAnterior - distanciaParaJogador;
        float deslocamentoReal = Vector2.Distance((Vector2)transform.position, _posicaoFisicaAnterior);

        // Contextos extraídos do estado
        float vidaInimigo = _ultimoEstadoAmbiente.Dados[3, 0];
        float vidaCritica = _ultimoEstadoAmbiente.Dados[4, 0];
        float vidaCriticaJogador = _ultimoEstadoAmbiente.Dados[11, 0];

        // Recompensa por se mover
        if (_ultimaAcao >= (int)Acao.MoverMaisX && _ultimaAcao <= (int)Acao.MoverDiagonalMenosXMenosY)
        {
            // Se não saiu do lugar (ficou preso em parede ou obstáculo)
            if (deslocamentoReal < 0.03f)
            {
                recompensa = -0.4f;
            }
            else
            {
                // Inimigo com vida crítica -> DEVE RECUAR (afastar)
                if (vidaCritica == 1.0f)
                {
                    // aproximacao < 0 significa que se afastou do perigo
                    recompensa = aproximacao < 0f ? 0.60f : -0.40f;
                }
                // Jogador com vida crítica -> AVANÇAR AGRESSIVAMENTE
                else if (vidaCriticaJogador == 1.0f)
                {
                    recompensa = aproximacao > 0f ? 1.0f : -0.30f;
                }
                // Comportamento Neutro Padrão -> Aproximar do jogador
                else
                {
                    recompensa = aproximacao > 0f ? 1.0f : -0.10f;
                }

                // Penalidade extra se ficou orbitando/sem mudar de distância efetiva
                if (Mathf.Abs(aproximacao) < 0.01f)
                {
                    recompensa -= 0.15f;
                }
            }
        }

        // Recompensa por atacar
        else if (_ultimaAcao == (int)Acao.AtacarCorpoACorpo)
        {
            // Se a ação for atacar corpo a corpo e estiver próximo, premia a ação; caso contrário, penaliza 
            recompensa = distanciaParaJogador <= 3.0f ? 1.5f : -0.4f;
        }
        else if (_ultimaAcao == (int)Acao.AtacarEmArea)
        {
            recompensa = distanciaParaJogador <= 4.0f ? 1.0f : -0.4f;
        }
        else if (_ultimaAcao == (int)Acao.AtacarDistancia || _ultimaAcao == (int)Acao.AtacarEspecial)
        {
            recompensa = distanciaParaJogador >= 5.0f ? 0.4f : -0.8f;
        }

        // Recompensa por curar
        else if (_ultimaAcao == (int)Acao.Curar)
        {
            // Se curou com vida baixa, e longe do jogador, recompensa alta; se curou com vida alta, recompensa menor; se não curou, penaliza
            if (vidaInimigo < 0.5f && distanciaParaJogador > 5f && _curou)
            {
                recompensa = 0.9f;
            }
            else if (vidaInimigo >= 0.5f && distanciaParaJogador > 3f && _curou)
            {
                recompensa = 0.4f;
            }
            else if (!_curou)
            {
                recompensa = -0.4f;
            }
            else
            {
                recompensa = -0.8f; // Penalidade por curar quando não era necessário
            }
        }
        
        /*else if (_ultimaAcao == (int)Acao.Defender)
        {
            recompensa = distanciaParaJogador <= 2.5f ? 0.4f : -0.3f;
        }*/

        if (_recompensaAcumuladaEventos != 0f)
        {
            if (_ativarLogs)
            {
                Debug.Log($"<color=yellow>[RL Evento]</color> Recompensa acumulada de eventos físicos: {_recompensaAcumuladaEventos:F2}");
            }
        }

        recompensa += _recompensaAcumuladaEventos;
        _recompensaAcumuladaEventos = 0f;

        _curou = false; // Reseta a flag de cura para o próximo passo

        // Próximo estado do ambiente após a ação
        Matriz proximoEstado = SaveManager.Instance != null
            ? SaveManager.Instance.ColetarSituacaoAtual()
            : Matriz.Zeros(30, 1);

        bool morto = _enemyHealth != null && _enemyHealth.EstaMorto;

        var transicao = new ReinforcementLearning.Transicao
        {
            Estado = _ultimoEstadoAmbiente,
            Acao = _ultimaAcao,
            Recompensa = recompensa,
            ProximoEstado = proximoEstado,
            Terminado = morto
        };


        float loss = _brain.TreinarPassoQLearning(transicao, taxaAprendizado: 0.005f);
        _brain.DecairEpsilon();

        RegistrarMetricaPasso(recompensa, loss);
        
        // IMPORTANTE!
        // SALVA O ESTADO E O INPUT DO JOGADOR PARA O DATASET SUPERVISIONADO
        if (SaveManager.Instance != null)
        {
            SaveManager.Instance.GravarAmostraSupervisionada(_ultimoEstadoAmbiente);
        }

        if (_ativarLogs && Mathf.Abs(recompensa) >= 0.4f)
        {
            Debug.Log($"<color=lime>[RL Passo]</color> Ação: {(Acao)_ultimaAcao} | Reward: {recompensa:F2} | Dist: {distanciaParaJogador:F2} | Epsilon: {_brain.Epsilon:F3}");
        }

        if (_textoAcaoAtual != null)
        {
            _textoAcaoAtual.text = $"Ação: {(Acao)_ultimaAcao}";
        }

        if (_textoRecompensaAtual != null)
        {
            _textoRecompensaAtual.text = $"Reward: {recompensa:F2}";
        }

        if (morto)
        {
            ExibirRelatorioMetricas(finalDeEpisodio: true);
        }
    }

    private void RegistrarMetricaPasso(float recompensa, float loss)
    {
        _recompensaJanelaAtual += recompensa;
        _somaLossJanela += loss;
        _passosJanelaAtual++;

        _recompensaEpisodioTotal += recompensa;
        _somaLossEpisodio += loss;
        _passosTreinoEpisodio++;
    }

    private void ExibirRelatorioMetricas(bool finalDeEpisodio)
    {
        float duracao = Time.time - _tempoInicioEpisodio;
        
        // Se for intervalo regular, usa a média da janela; se for morte, do episódio inteiro
        float lossExibicao = finalDeEpisodio 
            ? (_passosTreinoEpisodio > 0 ? _somaLossEpisodio / _passosTreinoEpisodio : 0f)
            : (_passosJanelaAtual > 0 ? _somaLossJanela / _passosJanelaAtual : 0f);

        float recompensaExibicao = finalDeEpisodio ? _recompensaEpisodioTotal : _recompensaJanelaAtual;
        int passosExibicao = finalDeEpisodio ? _passosEpisodioTotal : _passosJanelaAtual;

        int totalAcoes = _qtdExploracoes + _qtdExplotacoes;
        float taxaExploracao = totalAcoes > 0 ? ((float)_qtdExploracoes / totalAcoes) * 100f : 0f;

        string tipo = finalDeEpisodio ? "FIM DE EPISÓDIO (MORTE)" : "INTERVALO (20s)";

        Debug.Log($"<color={(finalDeEpisodio ? "orange" : "cyan")}><b>[RL - {tipo}]</b></color>\n" +
                $"• Recompensa no Período: <b>{recompensaExibicao:F2}</b>\n" +
                $"• Passos no Período: {passosExibicao}\n" +
                $"• Média Recompensa/Passo: {(passosExibicao > 0 ? recompensaExibicao / passosExibicao : 0f):F3}\n" +
                $"• TD Loss Médio: <b>{lossExibicao:F4}</b>\n" +
                $"• Epsilon: {_brain.Epsilon:F3} (Exploração: {taxaExploracao:F1}%)");

        if (_exportarMetricasCSV)
        {
            GravarLinhaCSV(duracao, _recompensaEpisodioTotal, recompensaExibicao, passosExibicao, lossExibicao, _brain.Epsilon, taxaExploracao, finalDeEpisodio);
        }

        // Limpa a janela periódica
        _recompensaJanelaAtual = 0f;
        _somaLossJanela = 0f;
        _passosJanelaAtual = 0;

        if (finalDeEpisodio)
        {
            _recompensaEpisodioTotal = 0f;
            _passosEpisodioTotal = 0;
            _somaLossEpisodio = 0f;
            _passosTreinoEpisodio = 0;
            _qtdExploracoes = 0;
            _qtdExplotacoes = 0;
            _tempoInicioEpisodio = Time.time;
        }
    }

    private void InicializarLoggerCSV()
    {
        if (!_exportarMetricasCSV) return;

        _caminhoArquivoCSV = Path.Combine(Application.persistentDataPath, "rl_metricas.CSV");

        if (!File.Exists(_caminhoArquivoCSV))
        {
            string cabecalho = "TempoSegundos;Tipo;RecompensaTotal;RecompensaPorJanela;Passos;TDLossMedio;Epsilon;TaxaExploracaoPct\n";
            File.WriteAllText(_caminhoArquivoCSV, cabecalho, Encoding.UTF8);
        }
    }

    private void GravarLinhaCSV(float duracao, float recompensaTotal, float recompensaPorJanela, int passos, float loss, float epsilon, float taxaExploracao, bool finalEpisodio)
    {
        if (string.IsNullOrEmpty(_caminhoArquivoCSV)) return;

        var inv = System.Globalization.CultureInfo.InvariantCulture;
        string tipo = finalEpisodio ? "Morte" : "Intervalo";

        string linha = string.Format(inv, "{0:F1};{1};{2:F2};{3:F2};{4};{5:F4};{6:F3};{7:F1}%\n",
            Time.time,
            tipo,
            recompensaTotal,
            recompensaPorJanela,
            passos,
            loss,
            epsilon,
            taxaExploracao);

        File.AppendAllText(_caminhoArquivoCSV, linha, Encoding.UTF8);
    }

    private void OnCollisionEnter2D(Collision2D collision)
    {
        if (!_iaHabilitada) return;

        // Penaliza por colisão com parede ou móvel
        if (collision.gameObject.CompareTag("Parede") || collision.gameObject.CompareTag("Movel"))
        {
            RegistrarRecompensaEvento(-0.5f);
        }
    }

    private void OnCollisionStay2D(Collision2D collision)
    {
        if (!_iaHabilitada) return;

        // Penaliza por permanecer colidindo com parede ou móvel
        if (collision.gameObject.CompareTag("Parede") || collision.gameObject.CompareTag("Movel"))
        {
            RegistrarRecompensaEvento(-0.1f);
        }
    }

    private void LogDiagnosticoQValores()
    {
        Matriz q = _brain.ForwardPass(_ultimoEstadoAmbiente, aplicarSoftmax: false);
        string diagnostico = $"<color=grey>[Diagnóstico Q]</color> Epsilon: {_brain.Epsilon:F3} | Ação Escolhida: {_acaoAtual}\n";

        for (int i = 0; i < (int)Acao.Max; i++)
        {
            diagnostico += $"[{((Acao)i)}: {q.Dados[i, 0]:F3}] ";
            if ((i + 1) % 5 == 0) diagnostico += "\n";
        }

        // Checagem de segurança contra explosão numérica
        if (float.IsNaN(q.Dados[0, 0]) || float.IsInfinity(q.Dados[0, 0]))
        {
            Debug.LogError("<color=red>[CRÍTICO RL]</color> Os valores Q retornaram NaN ou Infinity! A taxa de aprendizado está muito alta ou houve divisão por zero.");
        }
        else
        {
            Debug.Log(diagnostico);
        }
    }

    private IEnumerator CicloComportamento()
    {
        while (true)
        {
            if (_enemyHealth != null && _enemyHealth.EstaMorto)
            {
                PararMovimento();
                yield break; // Sai do loop se o inimigo estiver morto
            }

            yield return null;
        }
    }

    private void ExecutarPatrulha()
    {
        if (_inimigoEncontrarCaminho == null)
        {
            return;
        }

        Vector2 direcaoPatrulha = GetPatrulhaBularPosicao();

        _velocidadeAtual = _velocidadePadrao;

        AtualizarAnimacao(direcaoPatrulha);

        _inimigoEncontrarCaminho.MoverPara(direcaoPatrulha, _velocidadeAtual);
    }

    private void AtualizarAnimacao(Vector2 direcao)
    {
        if (_enemyCombat != null && _enemyCombat.EstaAtacando())
        {
            // Se o inimigo estiver atacando, não atualize a animação de movimento
            return;
        }

        if (_inimigoAnimator == null) return;
        _inimigoAnimator.SetFloat("UltimoX", direcao.x);
        _inimigoAnimator.SetFloat("UltimoY", direcao.y);
        _inimigoAnimator.SetFloat("DestinoX", direcao.x);
        _inimigoAnimator.SetFloat("DestinoY", direcao.y);
        _inimigoAnimator.SetInteger("idTransition", direcao != Vector2.zero ? 1 : 0);
    }

    private void PararMovimento()
    {
        if (_inimigoEncontrarCaminho != null)
        {
            _inimigoEncontrarCaminho.MoverPara(Vector2.zero, 0f);
        }

        if (_inimigoAnimator != null)
        {
            _inimigoAnimator.SetInteger("idTransition", 0);
        }
    }

    private Vector2 GetPatrulhaBularPosicao()
    {
        return new Vector2(
            Random.Range(-0.2f, 0.2f),
            Random.Range(-0.2f, 0.2f)
        ).normalized;
    }

    private void OnApplicationQuit()
    {
        if (_brain != null && SaveManager.Instance != null)
        {
            _brain.SalvarPesosParaJson(SaveManager.Instance.ObterCaminhoPesos);
        }
    }
}