## 🧠 Arquitetura Transformer Autoregressiva — Descrição Técnica

Este projeto implementa um **Transformer decoder-only autoregressivo** para modelagem de linguagem.
Dada uma sequência de tokens, o modelo aprende a estimar a distribuição condicional do próximo token:

$$P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

O treinamento maximiza a log-verossimilhança dos dados (equivalente à minimização da cross-entropy).

---

## 📥 Representação de Entrada

Cada token discreto (x_1) é mapeado para um vetor contínuo por uma matriz de embeddings:

$$e_i = E[x_i], \quad E \in \mathbb{R}^{|V| \times d_{model}}$$

Para incorporar ordem sequencial, adicionamos embeddings posicionais:

$$z_i^{(0)} = e_i + p_i$$

onde (p_i) representa a posição do token.

---

## 🔁 Bloco Transformer

O modelo é composto por (L) blocos idênticos. Cada bloco contém:

1. **Masked Multi-Head Self-Attention**
2. **Feed-Forward Network**
3. **Conexões residuais**
4. **Layer Normalization**

Para a camada (l):

$$z^{(l)} = \text{TransformerBlock}(z^{(l-1)})$$

---

## 🎯 Self-Attention com Máscara Causal

Para uma sequência de comprimento (T), definimos:

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$$

onde:

* $X \in \mathbb{R}^{T \times d_{model}}$
* $W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_k}$

A atenção escalada:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

A **máscara causal** (M) impede acesso ao futuro:

$$M_{ij} = \begin{cases} 0 & \text{se } j \le i \\ -\infty & \text{se } j > i \end{cases}$$

---

## 🧩 Multi-Head Attention

A atenção é aplicada em paralelo em (h) cabeças:

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$$

$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W_O$$

---

## 🧮 Feed-Forward Position-Wise

Aplicado independentemente em cada posição:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

---

## 🔄 Residual + Normalization

Cada subcamada usa residual connection:

$$y = \text{LayerNorm}(x + \text{Sublayer}(x))$$

---

## 📤 Camada de Saída

A representação final $H = z^{(L)}$ é projetada no vocabulário:

$$\text{logits} = HW_{out} + b$$

Probabilidades:

$$P(x_{t+1} \mid x_{\le t}) = \text{softmax}(\text{logits}_t)$$

---

## 🎓 Função de Perda

Treinamento por máxima verossimilhança:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, \dots, x_{t-1})$$

---

## 📐 Complexidade

Self-attention densa:

$$O(T^2 \cdot d_{model})$$

---

## ✅ Propriedades do Modelo

* Modelagem autoregressiva explícita
* Dependências globais via atenção
* Paralelização total no treinamento
* Fatoração causal garantida pela máscara triangular

---
