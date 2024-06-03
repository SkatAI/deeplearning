
# attention

https://chatgpt.com/c/e01a4580-cd66-4a48-88dc-9b99b0852ec5

Le mécanisme d'auto-attention fonctionne en permettant à chaque élément d'une séquence de calculer une "attention" ou une pondération par rapport à tous les autres éléments de la séquence, de manière à intégrer ces informations contextuelles de manière plus pertinente. Voici comment il fonctionne en détail :

1. **Calcul des Représentations** : Chaque élément de la séquence est transformé en trois vecteurs appelés **requête** (query), **clé** (key), et **valeur** (value) à l'aide de matrices de poids. Pour un élément \( x_i \), les vecteurs sont calculés comme :
   \[
   Q_i = W_q x_i, \quad K_i = W_k x_i, \quad V_i = W_v x_i
   \]
   où \( W_q \), \( W_k \), et \( W_v \) sont des matrices de poids apprises.

2. **Calcul des Scores d'Attention** : Les scores d'attention sont calculés en prenant le produit scalaire entre les vecteurs de requête d'un élément et les vecteurs de clé de tous les éléments de la séquence. Ceci est généralement normalisé en utilisant une fonction softmax :
   \[
   \text{score}(Q_i, K_j) = \frac{Q_i \cdot K_j}{\sqrt{d_k}}
   \]
   où \( d_k \) est la dimension des vecteurs de clé, et la fonction softmax transforme ces scores en probabilités.

3. **Pondération des Valeurs** : Les vecteurs de valeur sont ensuite pondérés par ces scores d'attention pour chaque élément de la séquence. Cela permet à chaque élément de la séquence d'incorporer des informations contextuelles de tous les autres éléments :
   \[
   \text{attention}_i = \sum_j \text{softmax}(\text{score}(Q_i, K_j)) \cdot V_j
   \]

4. **Agrégation des Résultats** : Les vecteurs pondérés sont finalement agrégés pour former la représentation contextuelle de chaque élément dans la séquence.

Cette approche permet aux modèles basés sur l'auto-attention, comme les transformers, de capturer efficacement les relations à longue portée et les dépendances dans les données séquentielles, en parallèle et de manière computationnellement efficace.

### Références :
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)