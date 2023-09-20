## Algoritmos evolucionaros implementados

- Genetic Algorithm
- Evolution Strategy Multimember plus
- Bees Algorithm
- Ant Colony Optimization
<!-- - Greedy -->

## Sobre o projeto

<!-- O projeto testa cada algoritmo (exceto o greedy) várias vezes e compara os resultados com gráficos de fitness x generation. -->
O projeto testa cada algoritmo várias vezes e compara os resultados com gráficos de fitness x generation.

<img src="https://github.com/VictorG-028/Comp-Evolutiva-1VA/blob/d26cab570c78b914ea6db0b1da74c4ca59442712/plot_imgs/best_sample_fitness_over_generations.png" alt="comparison chart" width="640" height="384">

## Sobre o problema combinatorial

O problema é contrar uma build (combinação de 5 artefatos) de personagem de RPG.
Uma **Build** equivale a um **Indivíduo**.
Os artefatos são guardados em um dataset em formato de texto e foram extraidos do inventário de um jogo usando as ferramentas [Genshin Optimizer](https://frzyc.github.io/genshin-optimizer/) e [Genshin Scanner Amenoma](https://github.com/daydreaming666/Amenoma/tree/main).
Foi utilizado uma curva gaussiana customizada no algoritmo Evolutionary Strategy multimember plus.


## Bibliotecas utilizadas

- Numpy
- Pandas
- Matplotlib

<!-- Lembrar caso precise rodar esse projeto:

crair arquivo main ou run_algorithm.py) que chama funções definidas nos algoritmos
exportar funções nos arquivos dos algoritmos para serem chamadas no main
colocar opção de rodar como experimento/sample/trial ou como opção definitiva
colocar as calsses base dentro de src
colocar/virificar output de imagens para pasta plot_imgs fora de src
 -->