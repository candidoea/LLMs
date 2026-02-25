Excelente iniciativa! Ao incluir ambos os modelos, o seu repositório torna-se um laboratório completo de **Arquiteturas de Atenção**, mostrando que você domina as duas principais variações do artigo original de 2017: o **Decoder-only** (GPT) e o **Encoder-Decoder** (NMT).

Aqui está a proposta de um `README.md` robusto e unificado para o seu repositório:

---

# 🧠 Laboratório de LLMs: Arquiteturas Transformer

Este repositório contém implementações académicas e visuais das duas principais variações da arquitetura **Transformer**. O objetivo é demonstrar a aplicação prática de mecanismos de atenção em tarefas de modelagem de linguagem e tradução automática.

## 🚀 Modelos Implementados

### 1. GPT Académico (Decoder-only)

**Foco:** Modelagem de linguagem autoregressiva.
Este notebook implementa um modelo inspirado no GPT, onde o foco é a predição do próximo token baseado num contexto prévio.

* **Mecanismo:** *Masked Self-Attention* (Causal).
* **Diferenciais:**
* Uso de tokens especiais `[SEP]` para delimitação de contexto.
* Cálculo de **Perplexity** para avaliação de performance.
* Visualização do **Espaço Semântico (PCA)**: Redução de embeddings 64D para 3D para análise de agrupamentos de palavras.



### 2. Transformer NMT (Encoder-Decoder)

**Foco:** Tradução Automática Neuronal (Português -> Inglês).
Implementação da arquitetura clássica *Seq2Seq* para transformar sequências de um domínio noutro.

* **Mecanismo:** *Cross-Attention* (o Decoder consulta o Encoder).
* **Diferenciais:**
* **Regularização Ativa:** Implementação de Dropout (20%) e Data Augmentation para combater o *overfitting*.
* **Interface Interativa:** Função de inferência palavra por palavra com suporte a input do utilizador.
* **Visualização de Fibras 3D:** Mapeamento em tempo real das conexões de atenção entre os dois idiomas.



---

## 🔬 Fundamentos Matemáticos

### Atenção por Produto Escalar Escalonado

Ambos os modelos baseiam-se na fórmula fundamental:


$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

* No **GPT**, $M$ é uma máscara triangular que impede o modelo de "olhar para o futuro".
* No **NMT**, a *Cross-Attention* utiliza $Q$ do Decoder e $K, V$ do Encoder para realizar o alinhamento semântico.

### Regularização e Generalização

Para evitar que os modelos apenas "decorem" os datasets de treino, aplicamos:

* **Standardization Personalizada:** Limpeza rigorosa de strings via Regex no Grafo do TensorFlow.
* **Teacher Forcing:** Utilizado durante o treino para acelerar a convergência, fornecendo o alvo real como input para o passo seguinte.

---

## 📊 Visualização Avançada

O diferencial deste repositório é a capacidade de "abrir a caixa preta" do modelo:

1. **Top-10 Probability Flow:** Gráficos interativos (Plotly) que mostram as 10 palavras mais prováveis na saída da Softmax.
2. **Panorâmica 3D:** Visualização ajustada (aspect ratio 3:1) para observar o fluxo de dados em sequências longas sem cortes.

---

## 🛠️ Tecnologias Utilizadas

* **TensorFlow / Keras:** Construção das camadas customizadas e loops de treino.
* **Pandas / Numpy:** Manipulação e aumento de dados sintéticos.
* **Plotly:** Visualização dinâmica e interativa das matrizes de atenção.
* **Scikit-Learn:** Redução de dimensionalidade para análise de Embeddings.

---

### Como utilizar este repositório:

1. Comece pelo `GPT_Academico_Final_Corrigido` para entender a predição de texto.
2. Avance para o `Transformer NMT` para compreender como dois Transformers comunicam entre si através da Cross-Attention.

---
