from ast import In
from typing import Callable

from Artifact import *
from Build import *

import numpy as np
import pandas as pd

from codetiming import Timer




# Cria as builds (indivíduos) da população 
def create_population(database: list[Artifact], 
                      pop_size: int = 1, 
                      mutation_rate: float = 2.5
                      ) -> tuple[np.ndarray, dict[str, np.ndarray]]: 

    # Separa a base de dados por artefato
    flowers  = list(filter(lambda a: a.type_ == 'Flower',  database))
    feathers = list(filter(lambda a: a.type_ == 'Feather', database))
    sands    = list(filter(lambda a: a.type_ == 'Sand',    database))
    goblets  = list(filter(lambda a: a.type_ == 'Goblet',  database))
    circlets = list(filter(lambda a: a.type_ == 'Circlet', database))

    def set_mutation_rate(artifact: Artifact):
        artifact.mutation_rate = mutation_rate
        return artifact

    flowers  = np.array(list(map(set_mutation_rate,  flowers)))
    feathers = np.array(list(map(set_mutation_rate, feathers)))
    sands    = np.array(list(map(set_mutation_rate, sands)))
    goblets  = np.array(list(map(set_mutation_rate, goblets)))
    circlets = np.array(list(map(set_mutation_rate, circlets)))

    flowers = np.random.choice(flowers, pop_size, replace=False)
    feathers = np.random.choice(feathers, pop_size, replace=False)
    sands = np.random.choice(sands, pop_size, replace=False)
    goblets = np.random.choice(goblets, pop_size, replace=False)
    circlets = np.random.choice(circlets, pop_size, replace=False)

    cromossomes = np.array([])
    for i in np.arange(pop_size):
        b = Build(flowers[i], feathers[i], sands[i], goblets[i], circlets[i])
        cromossomes = np.append(cromossomes, b, axis = None)
    
    bag_of = {
        "flower": flowers,
        "feather": feathers,
        "sand": sands,
        "goblet": goblets,
        "circlet": circlets,
    }

    return cromossomes, bag_of



def fitness_1(build: Build) -> float:

    sheet = build.get_artifact_sheet()

    ##############

    # Weapon_Crit_DMG = 66.2
    # Char_Crit_DMG = 38.4
    # Char_Base_ATK = 106.51
    # Weapon_Base_ATK = 608
    # Base_ATK = Weapon_Base_ATK + Char_Base_ATK
    # Weapon_Passive_1_HP = 20.0
    # Char_HP = 15552
    Total_HP = 15552 * np.divide(120.0 + sheet["HP%"], 100.0) + sheet["HP"]
    Elemental_Skill_ATK = np.min([np.multiply(0.06256, Total_HP), 2858.04]) # facade coeficient = 0.063 ; true coeficient = 0.06256
    Total_Weapon_Passive_ATK = np.divide(1.8 * Total_HP, 100.0)
    # Enemy_Pyro_Res = 10
    # Enemy_Level = 90
    # Char_Level = 90
    # Enemy_DEF_Multiplier = 0.5 # (Char._Level + 100)/(Char._Level + 100 + Enemy_Level + 100)
    # Active_Talent_Passive = 33.0
    # Active_set_bonus = 22.5 # 15% (2-piece bonus) + 7.5 (4-piece bonus)
    # Total_Pyro_DMG_Bonus = 33.0 + Active_set_bonus + sheet["Pyro"]
    Total_DMG_Bonus = np.divide(55.5 + sheet["Pyro"], 100.0)

    ##############

    Total_Crit_DMG = np.divide(154.6 + sheet["CD"], 100.0)

    Total_Crit_Rate = np.divide(np.min([5.0 + sheet["CR"], 100]), 100.0)

    Total_ATK = Elemental_Skill_ATK + Total_Weapon_Passive_ATK + np.multiply(714.51, np.divide(100 + sheet["ATK%"], 100.0)) + sheet["ATK"]

    ##############

    return np.multiply(np.multiply(np.multiply(np.multiply(2.4255, Total_ATK), 1 + np.multiply(Total_Crit_Rate, Total_Crit_DMG)), 1 + Total_DMG_Bonus), 0.45) # facade coeficient = 2.426 ; true coeficient = 2.4255



def fitness_2(build: Build) -> float:

    sheet = build.get_artifact_sheet()

    # Weapon_Passive = 46.9
    # Char_HP = 14695

    # Total_Shield_Strength = 0.25 # + Artifacts
    Total_HP = 14695 * np.divide(146.9 + sheet["HP%"], 100.0) + sheet["HP"]

    return np.multiply(1.875, (0.23*Total_HP + 2711.5))
    


# N-point crossover -> Escolhe N pontos de corte e gera 2 novos indivíduos
def crossover(b1: Build, b2: Build, points: int = 2) -> tuple[Build, Build]:

    # Link que ajuda a entender o código
    # https://medium.com/@samiran.bera/crossover-operator-the-heart-of-genetic-algorithm-6c0fdcb405c0

    possible_cuts = [0, 1, 2, 3, 4]
    random_cut = np.random.choice(possible_cuts, size = points, replace = False)

    #           <--0         <--1           <--2        <--3          <--4
    cromossome_1 = [b1["flower"], b1["feather"], b1["sand"], b1["goblet"], b1["circlet"]]
    cromossome_2 = [b2["flower"], b2["feather"], b2["sand"], b2["goblet"], b2["circlet"]]

    for cut in np.sort(random_cut):
        temp = cromossome_1[:cut] + cromossome_2[cut:]
        cromossome_2 = cromossome_2[:cut] + cromossome_1[cut:]
        cromossome_1 = temp

    b1 = Build(cromossome_1[0], cromossome_1[1],cromossome_1[2],cromossome_1[3],cromossome_1[4])
    b2 = Build(cromossome_2[0], cromossome_2[1],cromossome_2[2],cromossome_2[3],cromossome_2[4])

    return (b1, b2)



# Gassian Pertubation -> Substitui cada gene por outro gerado aleatoriamente usando a chance de Gauss
def mutation(b: Build, bag_of: dict[str, np.ndarray]) -> Build:
    
    genes_types = ["flower", "feather", "sand", "goblet", "circlet"]

    for gene_type in genes_types: # Faz um pertubaçao em cada gene

        while True: # Para quando ocorre uma pertubação bem sucedida
            artifacts = bag_of[gene_type]
            new_artifact = np.random.choice(artifacts, size = 1, replace = False)[0]

            number_diferent_atributes = b[gene_type].diference(new_artifact)

            # Escolhe se vai manter a pertubação com base no número de diferenças entre 2 artefatos
            # + diferença = maior probabilidade
            """ Formula de Gauss -> Probabilidade em função da diferença
            σ = mutation_rate
            Δx = number_diferent_atributes
            E = média da gaussiana = 5
            P(Δx) = 1/(σ \sqrt(2 np.pi)) * np.e ^ -( Δx - E )^2 / (2 * σ^2)

            Link do desenho da curva -> https://www.desmos.com/calculator/5bys1g8whk
            """
            # Observação -> \ é uma quebra de linha para deixar mais facil de ler

            probability = 2 * \
                np.divide(
                    1.0, b[gene_type].mutation_rate * np.sqrt(2*np.pi)
                ) * \
                np.power(
                    np.e, -np.divide(np.power(number_diferent_atributes - 5, 2), 
                    (2 * np.power(b[gene_type].mutation_rate, 2)))
                )


            if np.random.rand(1) <= probability:
                b[gene_type] = new_artifact
                break

    return b



"""
fitness -> Função fitness que será usada
target_fitness -> objetivo de parada da função fitness
"""
@Timer(name="decorator", 
       text="Tempo da busca: {:.4f} segundos")
def ES_multimember_plus(fitness: Callable, 
                        target_fitness: int, 
                        seed: int):
    
    # Parâmetros
    RANDOM_SEED = seed
    POP_SIZE = 300
    MAX_GENERATIONS = 10000
    MAX_NO_CHANGE_GENERATIONS = 750
    CROSSOVER_RATE = 0.50
    MUTATION_RATE = 2.5 # Desvio padrão
    N_POINT = 2 # 2-point crossover

    MU: int = POP_SIZE     # μ -> número de indivíduos selecionados para gerar os λ filhos
    LAMBDA: int = POP_SIZE # λ -> Número de filhos gerados

    # Configuração inicial
    np.random.seed(seed=RANDOM_SEED)
    artifacts = Artifact.read_database()
    population, bag_of_artifacts = create_population(artifacts, POP_SIZE, MUTATION_RATE)

    generation_fitness = np.zeros(shape=[POP_SIZE])
    best_fitness: float = -np.inf
    best_individual: Build = None
    best_index: int = None

    # Calcula o fitness da geração atual
    for i in np.arange(POP_SIZE):
        generation_fitness[i] = fitness(population[i])

        if generation_fitness[i] > best_fitness:
            best_fitness = generation_fitness[i]
            best_individual = population[i]
            best_index = i

    generation_count = 0 # Controla a quantidade máxima de gerações
    no_change_count = 0 # Controla a quantidade de gerações sem mudança
    logs: list[str] = [] # controla as informações de output

    df = pd.DataFrame()
    df["Cromossomes"] = population
    df["Fitness"] = generation_fitness

    # Loop principal
    while True:

        # print(generation_count)
        if not (best_fitness < target_fitness):
            stop_by = "Reach fitness target"
            break
        if not (generation_count < MAX_GENERATIONS):
            stop_by = "Reach MAX_GENERATIONS"
            break
        if not (no_change_count < MAX_NO_CHANGE_GENERATIONS):
            stop_by = "Reach MAX_NO_CHANGE_GENERATIONS"
            break

        # Seleciona aleatoriamente e uniformimente os μ parents
        selected = np.random.choice(df["Cromossomes"], size = MU, replace = False)

        # Guarda os λ novos elementos
        new_cromossomes = [] 

        # Forma pares para aplicar crossover
        for i in np.arange(2, MU+2, 2):
            
            pair = selected[i-2:i]

            if np.random.rand(1) <= CROSSOVER_RATE:
                pair = crossover(pair[0], pair[1], N_POINT)

            # Acrescenta no array de filhos
            new_cromossomes.append(pair[0])
            new_cromossomes.append(pair[1])

        # Aplica mutação nos filhos
        for i, cromossome in enumerate(new_cromossomes):
            new_cromossomes[i] = mutation(cromossome, bag_of_artifacts)


        # Contem μ + λ cromossomos
        selection = pd.DataFrame()
        selection["Cromossomes"] = np.concatenate((df["Cromossomes"], np.array(new_cromossomes)), axis = None) 

        # Calcula o fitness da geração atual
        generation_fitness = np.zeros(shape=[MU + LAMBDA])
        for i in np.arange(POP_SIZE):
            generation_fitness[i] = fitness(selection["Cromossomes"][i])

            if generation_fitness[i] > best_fitness:
                best_fitness = generation_fitness[i]
                best_individual = selection["Cromossomes"][i]
                best_index = i
                no_change_count = 0 # Reseta o contador de mudanças
        selection["Fitness"] = generation_fitness

        # Ordena os cromossomos por fitness (Elitismo)
        selection.sort_values(by='Fitness', ascending=False, inplace=True)
        selection.reset_index(drop=True, inplace=True)

        # Atualiza geração de forma elitista
        df["Cromossomes"] = selection["Cromossomes"][0:MU]
        df["Fitness"] = selection["Fitness"][0:MU] # Guarda o fitness dos sobreviventes

        generation_count += 1
        no_change_count += 1
        
        # Log das informações da geração
        logs.append(f'best_fitness:{best_fitness}|best_individual:{best_individual}|sum:{df["Fitness"].sum()}|mean:{df["Fitness"].mean()}\n')

    return {
        'best_fitness': best_fitness,
        'best_individual': best_individual,
        'best_index': best_index,
        'stop_by': stop_by, 
        'logs': logs
    }

print("ÍNÍCIO")
if __name__ == '__main__':
    INPUTS = ["Hu Tao + Báculo de Homa R1 + Habilidade Elemental + Ataque Carregado", "Zhongli + Borla Preta R5 + Dano de Absorção do Escudo"]
    file_names = ["first", "second"]
    functions = [fitness_1, fitness_2]
    targets = [30000, 50000] # 30k dano e 50k absorção

    seeds = [seed for seed in range(0, 30, 1)]

    for function, target, file_name, input_ in zip(functions, targets, file_names, INPUTS):
        
        for seed in seeds:
            result = ES_multimember_plus(function, target, seed)

            print("Resultados")
            print(f"Combinação: {input_}")
            print(f"Melhor output da build: { result['best_fitness'] }")
            print(f"Melhor build: { result['best_individual'] }")
            print(f"Index da build: { result['best_index'] }")
            print(f"Stop by: { result['stop_by'] }")
            # print(f"Log: { result['logs'] }")
            print("=------------=")

            # Escreve num arquivo os resultados
            with open(f"ES_{file_name}_{seed}.txt", "w") as f:
                for log in result['logs']:
                    f.writelines(log)
print("FIM")



