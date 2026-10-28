using System.IO;
using System.Text;
using System.Globalization;
using UnityEngine;

public class SaveManager : MonoBehaviour
{
    public static SaveManager Instance { get; private set; }

    [Header("Jogador Componentes")]
    [SerializeField] private GameObject _player;
    [SerializeField] private PlayerController _playerController;
    [SerializeField] private PlayerHealth _playerHealth;
    [SerializeField] private PlayerMovement _playerMovement;
    [SerializeField] private PlayerInput _playerInput;
    [SerializeField] private PlayerHand _playerHand;
    [SerializeField] private PlayerDash _playerDash;
    [SerializeField] private ActiveWeapon _playerCombat;

    [Header("Inimigo Componentes")]
    [SerializeField] private GameObject _enemy;
    [SerializeField] private EnemyAI _enemyAI;
    [SerializeField] private EnemyController _enemyController;
    [SerializeField] private EnemyInput _enemyInput;
    [SerializeField] private EnemyHealth _enemyHealth;
    [SerializeField] private EnemyCombat _enemyCombat;
    [SerializeField] private EnemyPathfinding _enemyPathfinding;
    [SerializeField] private GameObject _enemyEspecial;

    [SerializeField] private GameObject _snake;
    [SerializeField] private Snake _snakeScript;
    [SerializeField] private SnakeMovimento _snakeMoveScript;

    [Header("Arena parâmetros")]
    [SerializeField] private Vector2 _centroArena = Vector2.zero;
    [SerializeField] private Vector2 _tamanhoArena = new Vector2(16f, 9f);

    [Header("Caminhos de Arquivo")]
    [SerializeField] private string _caminhoArquivoEntradaSaida;
    [SerializeField] private string _caminhoArquivoPesos;
    [SerializeField] private string _caminhoCSVSupervisionado;

    [Header("Taxa de Amostragem Manual")]
    [SerializeField] private float _intervaloGravacaoManual = 0.1f; // Grava 10 vezes por segundo se o segundo jogador estiver ativo
    private float _proximaGravacaoManualTempo = 0f;

    private enum TipoObjeto
    {
        Vazio = 0,
        Espatula = 1,
        Estilingue = 2,
        Max = 3
    }

    public string ObterCaminhoPesos => _caminhoArquivoPesos;
    public string ObterCaminhoEntradaSaida => _caminhoArquivoEntradaSaida;

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            Debug.Log("SaveManager: Instância criada.");
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Debug.Log("SaveManager: Instância duplicada destruída.");
            Destroy(gameObject);
            return;
        }

        LocalizarReferencias();

        _caminhoArquivoPesos = Path.Combine(Application.persistentDataPath, "pesos_comportamento_rl.json");
        _caminhoArquivoEntradaSaida = Path.Combine(Application.persistentDataPath, "entrada_saida_rl.csv");
    }

    private void Start()
    {
        InicializarCSVSupervisionado();
    }

    private void Update()
    {
        // Se o modo de segundo jogador estiver ativo, gerencia a coleta regular de dados
        if (GameMaster.Instance != null && GameMaster.Instance.SegundoJogador)
        {
            if (Time.time >= _proximaGravacaoManualTempo)
            {
                _proximaGravacaoManualTempo = Time.time + _intervaloGravacaoManual;
                ColetarEGravacaoPassoManual();
            }
        }
    }

    public void LocalizarReferencias()
    {
        if (_player == null) _player = GameObject.FindGameObjectWithTag("Player");
        if (_player != null)
        {
            _playerController = _player.GetComponent<PlayerController>();
            _playerHealth = _player.GetComponent<PlayerHealth>();
            _playerMovement = _player.GetComponent<PlayerMovement>();
            _playerInput = _player.GetComponent<PlayerInput>();
            _playerCombat = _player.GetComponentInChildren<ActiveWeapon>();
            _playerHand = _player.GetComponentInChildren<PlayerHand>();
            _playerDash = _player.GetComponent<PlayerDash>();
        }

        if (_enemy == null) _enemy = GameObject.FindGameObjectWithTag("Inimigo");
        if (_enemy != null)
        {
            _enemyAI = _enemy.GetComponent<EnemyAI>();
            _enemyController = _enemy.GetComponent<EnemyController>();
            _enemyInput = _enemy.GetComponent<EnemyInput>();
            _enemyHealth = _enemy.GetComponent<EnemyHealth>();
            _enemyCombat = _enemy.GetComponent<EnemyCombat>();
            _enemyPathfinding = _enemy.GetComponent<EnemyPathfinding>();
            _enemyEspecial = GameObject.FindGameObjectWithTag("InimigoEspecial");
        }

        if (_snake == null) _snake = GameObject.FindGameObjectWithTag("Snake");
        if (_snake != null)
        {
            _snakeScript = _snake.GetComponent<Snake>();
            _snakeMoveScript = _snake.GetComponent<SnakeMovimento>();
        }
    }

    public Matriz ColetarSituacaoAtual()
    {
        if (_player == null || _enemy == null)
        {
            LocalizarReferencias();
        }

        bool temJogador = _player != null;
        bool temInimigo = _enemy != null;

        Vector2 posicaoJogador = temJogador ? (Vector2)_player.transform.position : Vector2.zero;
        Vector2 posicaoInimigo = temInimigo ? (Vector2)_enemy.transform.position : Vector2.zero;

        float distanciaNorm = 0f;
        float dirXParaJogador = 0f;
        float dirYParaJogador = 0f;

        if (temJogador && temInimigo)
        {
            Vector2 vetorParaJogador = posicaoJogador - posicaoInimigo;
            float distanciaReal = vetorParaJogador.magnitude;

            distanciaNorm = Mathf.Clamp01(distanciaReal / 20f);
            if (distanciaReal > 0.001f)
            {
                Vector2 direcao = vetorParaJogador / distanciaReal;
                dirXParaJogador = direcao.x;
                dirYParaJogador = direcao.y;
            }
        }

        float arenaMinX = _centroArena.x - _tamanhoArena.x / 2f;
        float arenaMinY = _centroArena.y - _tamanhoArena.y / 2f;

        float posXJogadorNorm = temJogador ? Mathf.Clamp01((posicaoJogador.x - arenaMinX) / _tamanhoArena.x) : 0.5f;
        float posYJogadorNorm = temJogador ? Mathf.Clamp01((posicaoJogador.y - arenaMinY) / _tamanhoArena.y) : 0.5f;

        float posXInimigoNorm = temInimigo ? Mathf.Clamp01((posicaoInimigo.x - arenaMinX) / _tamanhoArena.x) : 0.5f;
        float posYInimigoNorm = temInimigo ? Mathf.Clamp01((posicaoInimigo.y - arenaMinY) / _tamanhoArena.y) : 0.5f;

        // ------------------------------------------------------
        // Estado do Inimigo
        float vidaInimigoNorm = 0f;
        float inimigoVidaCritica = 0f;
        float podeAtacar = 0f;
        float podeEspecial = 0f;
        float podeArea = 0f;
        float estaAtacando = 0f;
        float inimigoResistenciaNorm = 0f;
        float velInimigoNorm = 0f;
        float inimigoPodeCurar = 0f;

        if (temInimigo)
        {
            float vidaInimigo = _enemyHealth != null ? _enemyHealth.ObterVidaAtual() : 0f;
            float vidaMaxInimigo = _enemyHealth != null ? Mathf.Max(1f, _enemyHealth.ObterVidaMaxima()) : 1f;
            vidaInimigoNorm = Mathf.Clamp01(vidaInimigo / vidaMaxInimigo);
            inimigoVidaCritica = vidaInimigoNorm <= 0.3f ? 1.0f : 0.0f;

            float resistenciaInimigo = _enemyHealth != null ? _enemyHealth.ObterResistencia() : 0f;
            float resistenciaMaxInimigo = _enemyHealth != null ? Mathf.Max(1f, _enemyHealth.ObterResistenciaMax()) : 1f;
            inimigoResistenciaNorm = Mathf.Clamp01(resistenciaInimigo / resistenciaMaxInimigo);

            inimigoPodeCurar = (_enemyHealth != null && _enemyHealth.ObterPodeCurar()) ? 1.0f : 0.0f;

            podeAtacar = (_enemyCombat != null && _enemyCombat.ObterPodeAtacar()) ? 1.0f : 0.0f;
            podeEspecial = (_enemyCombat != null && _enemyCombat.ObterPodeAtaqueEspecial()) ? 1.0f : 0.0f;
            podeArea = (_enemyCombat != null && _enemyCombat.ObterPodeAtaqueEmArea()) ? 1.0f : 0.0f;
            estaAtacando = (_enemyCombat != null && _enemyCombat.ObterEstaAtacando()) ? 1.0f : 0.0f;
            velInimigoNorm = _enemyPathfinding != null ? Mathf.Clamp01(_enemyPathfinding.ObterVelocidadeAtual / 10f) : 0f;
        }

        // ------------------------------------------------------
        // Estado do jogador
        float vidaJogadorNorm = 0f;
        float vidaJogadorCritica = 0f;
        float jogadorPodeDash = 0f;
        float jogadorEstaAtacando = 0f;
        float velJogadorNorm = 0f;
        float jogadorInputX = 0f;
        float jogadorInputY = 0f;
        float jogadorEstaComArma = 0f;
        float jogadorTipoArma = 0f;
        float jogadorPodeCurar = 0f;

        if (temJogador)
        {
            float vidaJogador = _playerHealth != null ? (float)_playerHealth.ObterVidaAtual : 0f;
            float vidaMaxJogador = _playerHealth != null ? Mathf.Max(1f, (float)_playerHealth.ObterVidaMaxima) : 1f;
            vidaJogadorNorm = Mathf.Clamp01(vidaJogador / vidaMaxJogador);
            vidaJogadorCritica = vidaJogadorNorm <= 0.3f ? 1.0f : 0.0f;

            jogadorPodeCurar = (_playerHealth != null && _playerHealth.TemPotes()) ? 1.0f : 0.0f;
            jogadorPodeDash = (_playerDash != null && _playerDash.PodeDash()) ? 1.0f : 0.0f;
            jogadorEstaAtacando = (_playerCombat != null && _playerCombat.EstaAtacando()) ? 1.0f : 0.0f;
            velJogadorNorm = _playerMovement != null ? Mathf.Clamp01(_playerMovement.ObterVelocidadeAtual / 10f) : 0f;
            jogadorInputX = _playerController != null ? Mathf.Clamp(_playerController.GetUltimoX(), -1f, 1f) : 0f;
            jogadorInputY = _playerController != null ? Mathf.Clamp(_playerController.GetUltimoY(), -1f, 1f) : 0f;

            TipoObjeto tipoArmaDetectada = TipoObjeto.Vazio;

            if (ActiveWeapon.Instance != null && ActiveWeapon.Instance.ArmaAtiva != null)
            {
                MonoBehaviour arma = ActiveWeapon.Instance.ArmaAtiva;

                if (arma is Spatula)
                {
                    tipoArmaDetectada = TipoObjeto.Espatula;
                }
                else if (arma is Estilingue)
                {
                    tipoArmaDetectada = TipoObjeto.Estilingue;
                }
            }

            jogadorEstaComArma = tipoArmaDetectada != TipoObjeto.Vazio ? 1.0f : 0.0f;
            
            // Normaliza o tipo da arma entre 0.0 e 1.0 (ex: Vazio = 0.0, Espátula = 0.33, Estilingue = 0.66)
            jogadorTipoArma = (float)tipoArmaDetectada / (float)TipoObjeto.Max;
        }

        // ------------------------------------------------------
        // Estado da Cobra
        float cobraPresente = _snake != null ? 1.0f : 0.0f;
        float distanciaCobraNorm = 1.0f;
        Vector2 dirParaCobra = Vector2.zero;

        if (_snake != null)
        {
            Vector3 origem = temInimigo ? _enemy.transform.position : (temJogador ? _player.transform.position : Vector3.zero);
            Vector2 vetorCobra = (Vector2)_snake.transform.position - (Vector2)origem;

            float distanciaCobra = vetorCobra.magnitude;
            distanciaCobraNorm = Mathf.Clamp01(distanciaCobra / 15f);

            if (distanciaCobra > 0.001f)
            {
                dirParaCobra = vetorCobra / distanciaCobra;
            }
        }

        // Vetor com 30 características numéricas
        float[] entradas = new float[]
        {
            distanciaNorm,          // [0]
            dirXParaJogador,        // [1]
            dirYParaJogador,        // [2]
            vidaInimigoNorm,        // [3]
            inimigoVidaCritica,     // [4]
            podeAtacar,             // [5]
            podeEspecial,           // [6]
            podeArea,               // [7]
            estaAtacando,           // [8]
            inimigoResistenciaNorm, // [9]
            vidaJogadorNorm,        // [10]
            vidaJogadorCritica,     // [11]
            jogadorPodeDash,        // [12]
            jogadorEstaAtacando,    // [13]
            jogadorInputX,          // [14]
            jogadorInputY,          // [15]
            posXInimigoNorm,        // [16]
            posYInimigoNorm,        // [17]
            posXJogadorNorm,        // [18]
            posYJogadorNorm,        // [19]
            cobraPresente,          // [20]
            distanciaCobraNorm,     // [21]
            dirParaCobra.x,         // [22]
            dirParaCobra.y,         // [23]
            velInimigoNorm,         // [24]
            velJogadorNorm,         // [25]
            jogadorEstaComArma,     // [26]
            jogadorTipoArma,        // [27]
            inimigoPodeCurar,       // [28]
            jogadorPodeCurar        // [29]
        };

        return Matriz.DeVetorColuna(entradas);
    }

    private void InicializarCSVSupervisionado()
    {
        if (string.IsNullOrEmpty(_caminhoCSVSupervisionado))
        {
            _caminhoCSVSupervisionado = Path.Combine(Application.persistentDataPath, "dataset_supervisionado_dois_jogadores.csv");
        }

        if (!File.Exists(_caminhoCSVSupervisionado))
        {
            var csvContent = new StringBuilder();
            
            // 30 Variáveis de Estado (X)
            csvContent.Append("DistanciaNorm;DirXParaJogador;DirYParaJogador;");
            csvContent.Append("VidaInimigoNorm;InimigoVidaCritica;PodeAtacar;PodeEspecial;PodeArea;EstaAtacando;InimigoResistenciaNorm;");
            csvContent.Append("VidaJogadorNorm;VidaJogadorCritica;JogadorPodeDash;JogadorEstaAtacando;JogadorInputX;JogadorInputY;");
            csvContent.Append("PosXInimigoNorm;PosYInimigoNorm;PosXJogadorNorm;PosYJogadorNorm;");
            csvContent.Append("CobraPresente;DistanciaCobraNorm;DirParaCobraX;DirParaCobraY;");
            csvContent.Append("VelInimigoNorm;VelJogadorNorm;JogadorEstaComArma;JogadorTipoArma;");
            csvContent.Append("InimigoPodeCurar;JogadorPodeCurar;");
            
            // Saídas Supervisionadas / Ações Tomadas (Y)
            // Inimigo é o alvo principal a ser aprendido pela rede
            csvContent.Append("InimigoAcaoId;InimigoAcaoNome;JogadorAcaoId;JogadorAcaoNome\n");

            File.WriteAllText(_caminhoCSVSupervisionado, csvContent.ToString(), Encoding.UTF8);
        }
    }

    private void ColetarEGravacaoPassoManual()
    {
        if (_enemyHealth != null && _enemyHealth.EstaMorto) return;
        if (_playerHealth != null && _playerHealth.EstaMorto) return;

        Matriz estado = ColetarSituacaoAtual();
        GravarAmostraSupervisionada(estado);
    }

    /// <summary>
    /// Salva o estado atual e as decisões tomadas por ambos os jogadores.
    /// </summary>
    public void GravarAmostraSupervisionada(Matriz estadoAmbiente)
    {
        if (string.IsNullOrEmpty(_caminhoCSVSupervisionado) || estadoAmbiente == null) return;

        if (_enemyInput == null && _enemy != null)
        {
            _enemyInput = _enemy.GetComponent<EnemyInput>();
        }

        // Coleta ação do Inimigo (Segundo Jogador)
        int acaoInimigo = _enemyInput != null ? (int)_enemyInput.ObterUltimoBotaoPressionadoInimigo() : 0;
        string nomeAcaoInimigo = _enemyInput != null ? _enemyInput.ObterUltimoBotaoPressionadoInimigo().ToString() : "Nenhum";

        // Coleta ação do Jogador 1
        int acaoJogador = _playerInput != null ? (int)_playerInput.ObterUltimoBotaoPressionado() : 0;
        string nomeAcaoJogador = _playerInput != null ? _playerInput.ObterUltimoBotaoPressionado().ToString() : "Nenhum";

        // Se ambos estiverem ociosos (nenhum botão pressionado), é ignorado para não poluir o dataset
        if (acaoInimigo == 0 && acaoJogador == 0) return;

        var csv = new StringBuilder();
        var inv = CultureInfo.InvariantCulture;

        // Grava as 30 entradas do ambiente (X)
        for (int i = 0; i < estadoAmbiente.Linhas; i++)
        {
            csv.Append(string.Format(inv, "{0:F4};", estadoAmbiente.Dados[i, 0]));
        }

        // Grava as saídas correspondentes (Y)
        csv.Append(string.Format(inv, "{0};{1};{2};{3}\n", acaoInimigo, nomeAcaoInimigo, acaoJogador, nomeAcaoJogador));

        File.AppendAllText(_caminhoCSVSupervisionado, csv.ToString(), Encoding.UTF8);
    }
}