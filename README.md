## 🧠 Arquitetura Transformer Autoregressiva — Descrição Técnica

Este projeto implementa um **Transformer decoder-only autoregressivo** para modelagem de linguagem.
Dada uma sequência de tokens, o modelo aprende a estimar a distribuição condicional do próximo token:

O treinamento maximiza a log-verossimilhança dos dados (equivalente à minimização da cross-entropy).

---

## 📥 Representação de Entrada

Cada token discreto () é mapeado para um vetor contínuo por uma matriz de embeddings:

Para incorporar ordem sequencial, adicionamos embeddings posicionais:

onde () representa a posição do token.

---

## 🔁 Bloco Transformer

O modelo é composto por () blocos idênticos. Cada bloco contém:

1. **Masked Multi-Head Self-Attention**
2. **Feed-Forward Network**
3. **Conexões residuais**
4. **Layer Normalization**

Para a camada ():

---

## 🎯 Self-Attention com Máscara Causal

Para uma sequência de comprimento (), definimos:

onde:

* 
* 

A atenção escalada:

A **máscara causal** () impede acesso ao futuro:

Isso garante fatoração autoregressiva da distribuição conjunta.

---

## 🧩 Multi-Head Attention

A atenção é aplicada em paralelo em () cabeças:

---

## 🧮 Feed-Forward Position-Wise

Aplicado independentemente em cada posição:

Normalmente: 

---

## 🔄 Residual + Normalization

Cada subcamada usa residual connection:

---

## 📤 Camada de Saída

A representação final  é projetada no vocabulário:

Probabilidades:

---

## 🎓 Função de Perda

Treinamento por máxima verossimilhança:

Equivalente à **cross-entropy token-wise**.

---

## ✍️ Geração Autoregressiva

Durante inferência:

1. Entrada inicial ()
2. Computar 
3. Amostrar ou escolher `argmax`
4. Concatenar e repetir

---

## 📐 Complexidade

Self-attention densa:

onde () é o comprimento da sequência.

---

## ✅ Propriedades do Modelo

* Modelagem autoregressiva explícita
* Dependências globais via atenção
* Paralelização total no treinamento
* Fatoração causal garantida pela máscara triangular

---
