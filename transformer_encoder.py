import numpy as np
import pandas as pd

# Passo 1: Preparação dos Dados

def preparar_dados():
    # 1. Criar um pequeno DataFrame no pandas simulando um vocabulário
    vocabulario = {
        "o": 0,
        "banco": 1,
        "bloqueou": 2,
        "cartao": 3,
        "meu": 4,
        "ontem": 5
    }
    df_vocab = pd.DataFrame(list(vocabulario.items()), columns=['Palavra', 'ID'])
    print("Vocabulário Simulado:\n", df_vocab)

    # 2. Definir uma frase de entrada e convertê-la em uma lista de IDs
    frase = "o banco bloqueou meu cartao ontem"
    palavras = frase.split()
    ids_entrada = [vocabulario[p] for p in palavras]
    print(f"\nFrase de entrada: '{frase}'")
    print(f"IDs de entrada: {ids_entrada}")

    # 3. Inicializar uma 'Tabela de Embeddings' simulada
    vocab_size = len(vocabulario)
    d_model = 64 # Conforme sugestão para processamento em CPU
    
    # Usamos uma inicialização menor (Xavier/He) para evitar valores muito altos que causam instabilidade
    embedding_table = np.random.randn(vocab_size, d_model) * 0.1
    
    # Converter IDs para embeddings
    embeddings_entrada = np.array([embedding_table[i] for i in ids_entrada])
    
    # 4. O tensor de entrada final (X) deve ter o formato (Batch_size, Sequence_Length, d_model)
    X = np.expand_dims(embeddings_entrada, axis=0)
    
    print(f"\nFormato do tensor de entrada (X): {X.shape} (Batch, Seq_Len, d_model)")
    return X, d_model

# Passo 2: O Motor Matemático

def softmax_estavel(x):
    """
    Implementação da função Softmax numericamente estável.
    Subtraímos o valor máximo de cada linha antes de calcular o exponencial
    para evitar o erro de overflow (np.exp de números muito grandes).
    """
    # x shape: (Batch, Seq, Seq)
    # Subtrair o máximo ao longo do último eixo
    max_x = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - max_x)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(X, d_model):
    """Implementação do Scaled Dot-Product Attention."""
    d_k = d_model
    
    # 1. Inicializar matrizes de pesos aleatórias (escala reduzida para estabilidade)
    Wq = np.random.randn(d_model, d_k) * 0.1
    Wk = np.random.randn(d_model, d_k) * 0.1
    Wv = np.random.randn(d_model, d_model) * 0.1
    
    # 2. Gerar Q, K e V a partir de X
    Q = np.matmul(X, Wq) 
    K = np.matmul(X, Wk) 
    V = np.matmul(X, Wv) 
    
    # 3. Calcular produto escalar entre Q e a transposta de K
    K_T = K.transpose(0, 2, 1)
    scores = np.matmul(Q, K_T) 
    
    # 4. Fazer o scaling (divisão) por sqrt(d_k) - ESSENCIAL PARA ESTABILIDADE
    scaled_scores = scores / np.sqrt(d_k)
    
    # 5. Aplicar Softmax Estável
    attention_weights = softmax_estavel(scaled_scores)
    
    # 6. Multiplicar o resultado pelas matrizes de valor V
    output = np.matmul(attention_weights, V) 
    
    return output

def layer_norm(X, epsilon=1e-6):
    """Implementação da Layer Normalization."""
    # Normalização de camada opera na dimensão das features (último eixo)
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    
    # Normalizar: (X - mean) / sqrt(var + epsilon)
    return (X - mean) / np.sqrt(var + epsilon)

def feed_forward_network(X, d_model, d_ff=256):
    """Implementação da Feed-Forward Network (FFN)."""
    # 1. Primeira transformação linear (expansão)
    W1 = np.random.randn(d_model, d_ff) * 0.1
    b1 = np.zeros((1, d_ff))
    
    # 2. Função de ativação ReLU (max(0, x))
    h1 = np.maximum(0, np.matmul(X, W1) + b1)
    
    # 3. Segunda transformação linear (contração)
    W2 = np.random.randn(d_ff, d_model) * 0.1
    b2 = np.zeros((1, d_model))
    
    output = np.matmul(h1, W2) + b2
    return output

# Passo 3: Empilhando tudo

def encoder_block(X, d_model):
    """Executa o fluxo de uma única camada do Encoder."""
    # 1. X_att = SelfAttention(X)
    X_att = scaled_dot_product_attention(X, d_model)
    
    # 2. X_norm1 = LayerNorm(X + X_att) (Conexão Residual + Normalização)
    X_norm1 = layer_norm(X + X_att)
    
    # 3. X_ffn = FFN(X_norm1)
    X_ffn = feed_forward_network(X_norm1, d_model)
    
    # 4. X_out = LayerNorm(X_norm1 + X_ffn) (Conexão Residual + Normalização)
    X_out = layer_norm(X_norm1 + X_ffn)
    
    return X_out

def transformer_encoder_scratch():
    print("--- Iniciando Construção do Transformer Encoder From Scratch ---\n")
    
    # Preparação dos dados
    X, d_model = preparar_dados()
    
    # Loop para N=6 camadas idênticas
    N = 6
    print(f"\nPassando por N={N} camadas do Encoder...")
    
    current_X = X
    for i in range(N):
        current_X = encoder_block(current_X, d_model)
        print(f"Camada {i+1} concluída. Shape: {current_X.shape}")
    
    # Validação de Sanidade
    print("\n--- Validação de Sanidade ---")
    print(f"Shape Inicial: {X.shape}")
    print(f"Shape Final (Vetor Z): {current_X.shape}")
    
    if X.shape == current_X.shape:
        print("Sucesso! As dimensões foram preservadas.")
        
        # Verificar se há NaNs na saída
        if np.isnan(current_X).any():
            print("AVISO: Foram detectados valores NaN na saída.")
        else:
            print("Estabilidade Numérica: OK (Sem NaNs).")
    else:
        print("Erro: As dimensões mudaram durante o processamento.")
    
    print("\nRepresentação contínua densa (Z) - Primeiros 5 valores do primeiro token:")
    print(current_X[0, 0, :5])

if __name__ == "__main__":
    transformer_encoder_scratch()