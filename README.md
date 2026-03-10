# Transformer Encoder From Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ckGcLo-yYV0cG1hFAJsQpCdifgHvr8dw?usp=sharing)

ou https://colab.research.google.com/drive/1ckGcLo-yYV0cG1hFAJsQpCdifgHvr8dw?usp=sharing

Este projeto implementa um Transformer Encoder do zero usando apenas NumPy e Pandas. O objetivo é demonstrar o funcionamento interno do mecanismo de atenção e das camadas do Transformer, sem depender de bibliotecas de deep learning como TensorFlow ou PyTorch.

## Estrutura do Projeto

- `transformer_encoder.py`: Contém a implementação completa do Transformer Encoder, incluindo:
  - Preparação de dados simulados
  - Multi-Head Attention
  - Feed-Forward Network
  - Encoder Layer
  - Transformer Encoder completo

## Como Usar

1. Execute o script principal:
   ```bash
   python transformer_encoder.py
   ```

2. O script irá:
   - Preparar um vocabulário simulado
   - Converter uma frase de exemplo em embeddings
   - Processar através do Transformer Encoder
   - Exibir os resultados

## Dependências

- NumPy
- Pandas

Instale as dependências com:
```bash
pip install numpy pandas
```

## Exemplo de Saída

O script imprime o vocabulário, a frase de entrada, os IDs correspondentes e o formato do tensor de entrada.

## Conceitos Implementados

- **Embeddings**: Conversão de palavras em vetores numéricos
- **Multi-Head Attention**: Mecanismo de atenção com múltiplas cabeças
- **Feed-Forward Network**: Rede neural feed-forward com ativação ReLU
- **Residual Connections**: Conexões residuais para melhorar o treinamento
- **Layer Normalization**: Normalização das camadas (simulada)

```bash
--- Iniciando Construção do Transformer Encoder From Scratch ---

Vocabulário Simulado:
     Palavra  ID
0         o   0
1     banco   1
2  bloqueou   2
3    cartao   3
4       meu   4
5     ontem   5

Frase de entrada: 'o banco bloqueou meu cartao ontem'
IDs de entrada: [0, 1, 2, 4, 3, 5]

Formato do tensor de entrada (X): (1, 6, 64) (Batch, Seq_Len, d_model)

Passando por N=6 camadas do Encoder...
Camada 1 concluída. Shape: (1, 6, 64)
Camada 2 concluída. Shape: (1, 6, 64)
Camada 3 concluída. Shape: (1, 6, 64)
Camada 4 concluída. Shape: (1, 6, 64)
Camada 5 concluída. Shape: (1, 6, 64)
Camada 6 concluída. Shape: (1, 6, 64)

--- Validação de Sanidade ---
Shape Inicial: (1, 6, 64)
Shape Final (Vetor Z): (1, 6, 64)
Sucesso! As dimensões foram preservadas.
Estabilidade Numérica: OK (Sem NaNs).

Representação contínua densa (Z) - Primeiros 5 valores do primeiro token:
[ 0.33893649  0.46089679 -0.00154602  0.382796   -0.88802493]
```


## Limitações

Esta é uma implementação educacional simplificada:
- Não inclui positional encoding
- LayerNorm é simulada
- Não há treinamento real
- Apenas para demonstração conceitual

## Créditos

•
Implementação e Lógica: Este projeto foi desenvolvido para a disciplina de Tópicos em Inteligência Artificial.

•
Documentação e README: Este documento README.md foi gerado com o auxílio do Gemini (Google), que também auxiliou na organização de outros detalhes menos críticos do projeto para garantir clareza e profissionalismo.

## Nota sobre Integridade Acadêmica

O uso de IA Generativa (Gemini) foi restrito ao suporte na estruturação do código, esclarecimento de sintaxe de bibliotecas matemáticas (numpy) e geração de documentação. A lógica matemática central e a arquitetura do Transformer foram implementadas seguindo rigorosamente as especificações do enunciado do laboratório.

## Versão

v1.0




## Referências

Baseado no paper "Attention is All You Need" de Vaswani et al. (2017).