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
    embedding_table = np.random.randn(vocab_size, d_model)
    
    # Converter IDs para embeddings
    embeddings_entrada = np.array([embedding_table[i] for i in ids_entrada])
    
    # 4. O tensor de entrada final (X) deve ter o formato (Batch_size, Sequence_Length, d_model)
    # Adicionando a dimensão de Batch (Batch_size = 1)
    X = np.expand_dims(embeddings_entrada, axis=0)
    
    print(f"\nFormato do tensor de entrada (X): {X.shape} (Batch, Seq_Len, d_model)")
    return X, d_model

# Passo 2: Implementação da Multi-Head Attention

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Inicializar pesos para Q, K, V
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model)
    
    def split_heads(self, x):
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, d_k)
    
    def scaled_dot_product_attention(self, Q, K, V):
        # Q, K, V: (batch, num_heads, seq_len, d_k)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        output = np.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V)
        
        # Concatenar heads
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.d_model)
        
        # Aplicar W_o
        output = np.matmul(attn_output, self.W_o)
        return output, attn_weights

# Passo 3: Implementação da Feed-Forward Network

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        hidden = np.matmul(x, self.W1) + self.b1
        hidden = np.maximum(hidden, 0)  # ReLU
        output = np.matmul(hidden, self.W2) + self.b2
        return output