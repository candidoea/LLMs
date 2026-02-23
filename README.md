## 🧠 Arquitetura Transformer Autoregressiva — Descrição Técnica

Este projeto implementa um **Transformer decoder-only autoregressivo** para modelagem de linguagem.
Dada uma sequência de tokens, o modelo aprende a estimar a distribuição condicional do próximo token:

<img width="168" height="18" alt="image" src="https://github.com/user-attachments/assets/144c058b-89fe-4495-931d-f5d42070d4d3" />


O treinamento maximiza a log-verossimilhança dos dados (equivalente à minimização da cross-entropy).

---

## 📥 Representação de Entrada

Cada token discreto (x_i) é mapeado para um vetor contínuo por uma matriz de embeddings:

<img width="222" height="21" alt="image" src="https://github.com/user-attachments/assets/9de14097-9cc9-470a-9aa3-60083dd6a8de" />


Para incorporar ordem sequencial, adicionamos embeddings posicionais:

<img width="104" height="23" alt="image" src="https://github.com/user-attachments/assets/8e002768-5c87-4464-80f5-3f1556cb1c0d" />


onde (p_i) representa a posição do token.

---

## 🔁 Bloco Transformer

O modelo é composto por (L) blocos idênticos.
Cada bloco contém:

1. **Masked Multi-Head Self-Attention**
2. **Feed-Forward Network**
3. **Conexões residuais**
4. **Layer Normalization**

Para a camada (l):

<img width="244" height="21" alt="image" src="https://github.com/user-attachments/assets/d7f15d45-664f-4e95-86a0-9700194c0d57" />


---

## 🎯 Self-Attention com Máscara Causal

Para uma sequência de comprimento (T), definimos:

<img width="305" height="19" alt="image" src="https://github.com/user-attachments/assets/b9924958-6bce-438f-9377-17c757d28246" />


onde:

<img width="128" height="20" alt="image" src="https://github.com/user-attachments/assets/90669914-9d81-4b3a-a00c-4af66602f2ee" />

<img width="216" height="21" alt="image" src="https://github.com/user-attachments/assets/41919b05-6c74-4438-b27d-ca0b721f4048" />


A atenção escalada:

<img width="362" height="44" alt="image" src="https://github.com/user-attachments/assets/b9ddfb9f-f204-44a4-87a9-27c497ae1e98" />


A **máscara causal** (M) impede acesso ao futuro:

<img width="178" height="65" alt="image" src="https://github.com/user-attachments/assets/8a093149-35d6-4e77-8e22-900e8025d5f7" />


Isso garante fatoração autoregressiva da distribuição conjunta.

---

## 🧩 Multi-Head Attention

A atenção é aplicada em paralelo em (h) cabeças:

<img width="221" height="18" alt="image" src="https://github.com/user-attachments/assets/cdfd10c2-0d4d-4b83-9beb-c87f14db4c97" />


<img width="531" height="18" alt="image" src="https://github.com/user-attachments/assets/2eab5b78-2b0a-49af-a8e7-2ae870298e81" />


---

## 🧮 Feed-Forward Position-Wise

Aplicado independentemente em cada posição:

<img width="275" height="18" alt="image" src="https://github.com/user-attachments/assets/8fa3af35-3b63-4dbf-94a2-a3d632755c0a" />


Normalmente:

<img width="94" height="18" alt="image" src="https://github.com/user-attachments/assets/892b1824-c2f9-4543-b5f7-b809fb2ee407" />


---

## 🔄 Residual + Normalization

Cada subcamada usa residual connection:

<img width="349" height="19" alt="image" src="https://github.com/user-attachments/assets/eeb435ad-13bd-4387-aced-0613c9b99125" />


Isso melhora estabilidade do gradiente e convergência.

---

## 📤 Camada de Saída

A representação final:

<img width="65" height="17" alt="image" src="https://github.com/user-attachments/assets/18b3de0e-b8de-4a75-8dfc-0e1b886adb90" />


é projetada no vocabulário:

<img width="213" height="20" alt="image" src="https://github.com/user-attachments/assets/80dc0ba6-72d9-4b82-9d70-5e6ebd3af4dc" />


Probabilidades:

<img width="242" height="18" alt="image" src="https://github.com/user-attachments/assets/c33e89a4-d0ae-4bb0-914b-fe64ede863e3" />


---

## 🎓 Função de Perda

Treinamento por máxima verossimilhança:

<img width="243" height="51" alt="image" src="https://github.com/user-attachments/assets/c7806fdb-a83e-46e2-8cd6-e2a38ffce0e2" />


Equivalente à **cross-entropy token-wise**.

---

## ✍️ Geração Autoregressiva

Durante inferência:

1. Entrada inicial (x_{1:t})
2. Computar (P(x_{t+1}))
3. Amostrar ou escolher argmax
4. Concatenar e repetir

---

## 📐 Complexidade

Self-attention densa:

<img width="102" height="20" alt="image" src="https://github.com/user-attachments/assets/07713cf2-d48b-4093-bb53-44f823b65ef4" />


onde (T) é o comprimento da sequência.

---

## ✅ Propriedades do Modelo

* Modelagem autoregressiva explícita
* Dependências globais via atenção
* Paralelização total no treinamento
* Escalabilidade por profundidade e largura
* Fatoração causal garantida pela máscara triangular

---
