# Transformer Encoder From Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ckGcLo-yYV0cG1hFAJsQpCdifgHvr8dw?usp=sharing)

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

## Limitações

Esta é uma implementação educacional simplificada:
- Não inclui positional encoding
- LayerNorm é simulada
- Não há treinamento real
- Apenas para demonstração conceitual

## Referências

Baseado no paper "Attention is All You Need" de Vaswani et al. (2017).