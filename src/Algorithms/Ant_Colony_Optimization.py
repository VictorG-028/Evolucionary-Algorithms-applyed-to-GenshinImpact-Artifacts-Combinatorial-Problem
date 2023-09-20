from typing import Callable

from Base_Classes.Artifact import *
from Base_Classes.Build import *

import numpy as np
# from numpy.random import rand as np_random_rand
# from numpy import max as np_max, argmax as np_argmax
import pandas as pd

from codetiming import Timer



# Cria um dicionário com 5 listas de artefato 
def read_database(database: list[Artifact], pop_size: int) -> tuple[list[Build], dict[str, np.ndarray]]:

    # database -> 300 artefatos de cada tipo, 1500 (300 * 5) no total

    # Separa a base de dados por artefato 
    flowers  = np.array(list(filter(lambda a: a.type_ == 'Flower',  database)))
    feathers = np.array(list(filter(lambda a: a.type_ == 'Feather', database)))
    sands    = np.array(list(filter(lambda a: a.type_ == 'Sand',    database)))
    goblets  = np.array(list(filter(lambda a: a.type_ == 'Goblet',  database)))
    circlets = np.array(list(filter(lambda a: a.type_ == 'Circlet', database)))

    # flowers = np.random.choice(flowers, len(flowers), replace=False)
    # feathers = np.random.choice(feathers, len(feathers), replace=False)
    # sands = np.random.choice(sands, len(sands), replace=False)
    # goblets = np.random.choice(goblets, len(goblets), replace=False)
    # circlets = np.random.choice(circlets, len(circlets), replace=False)

    bag_of = {
        "flower": flowers,
        "feather": feathers,
        "sand": sands,
        "goblet": goblets,
        "circlet": circlets,
    }

    return bag_of



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
    # Artifact_Active_Set_Bonus = 22.5 # = 15% (2-piece bonus) + 7.5 (4-piece bonus)
    # Total_Pyro_DMG_Bonus = 33.0 + Active_Set_Bonus + sheet["Pyro"]
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



def get_artifact_value(a: Artifact, useless_stats: list[str]) -> np.int8:
    return sum(map(
            lambda stat: 1 if stat.type_ not in useless_stats else 0, 
            [a.main_stat, *a.sub_stats]
            ))



"""
fitness -> Função fitness que será usada
target_fitness -> Um dos objetivos de parada
"""
@Timer(name="decorator", text="Tempo da busca: {:.4f} segundos")
def ACO(fitness: Callable, 
        target_fitness: int, 
        useless_stats: list[str],
        seed: int):
    
    # Parâmetros
    RANDOM_SEED = seed
    MAX_GENERATIONS = 10000
    MAX_NO_CHANGE_GENERATIONS = 1000
    POP_SIZE = 300
    INITIAL_FEROMONE = 0.5
    ALPHA = 1
    BETA = 1
    VAPE_RATE = 0.05
    PATHS_TO_CHOSSE = 5

    # Configuração inicial
    np.random.seed(seed=RANDOM_SEED)
    artifacts = Artifact.read_database()
    bag_of_artifacts = read_database(artifacts, POP_SIZE)

    generation_count = 0  # Controla a quantidade máxima de gerações
    no_change_count  = 0  # Controla a quantidade de gerações sem mudança
    logs: list[str]  = [] # controla as informações de output

    # Vetorizadores
    vectorized_fitness = np.vectorize(fitness)
    vectorized_apply_alpha = np.vectorize(lambda value: np.power(value, ALPHA))
    vectorized_apply_beta = np.vectorize(lambda value: np.power(value, BETA))

    # Matrizes
    ants = np.zeros(shape=(POP_SIZE, PATHS_TO_CHOSSE), dtype=Artifact) # Cada linha é uma formiga e cada coluna é uma escolha de artefato
    pheromone = np.full(shape=(POP_SIZE, PATHS_TO_CHOSSE), fill_value=INITIAL_FEROMONE, dtype=np.float32)
    artifact_value = np.zeros(shape=(POP_SIZE, PATHS_TO_CHOSSE), dtype=int) # Guarda o "custo" de escolher um artefato
    builds = np.zeros(shape=(POP_SIZE, 1), dtype=Build) # Matrix auxiliar para calcular o fitness
    generation_fitness = np.zeros(shape=(POP_SIZE, 1), dtype=np.float32)
    choices = np.full(shape=(POP_SIZE, PATHS_TO_CHOSSE), fill_value = -1, dtype=int)
    delta_pheromone = np.zeros(shape=(POP_SIZE, PATHS_TO_CHOSSE), dtype=np.float32)

    # Preenche os custos
    for i in range(300):
        # map(lambda a: get_artifact_value(a, useless_stats), bag_of_artifacts["flower"])
        artifact_value[i][0] = get_artifact_value(bag_of_artifacts["flower"][i] , useless_stats)
        artifact_value[i][1] = get_artifact_value(bag_of_artifacts["feather"][i], useless_stats)
        artifact_value[i][2] = get_artifact_value(bag_of_artifacts["sand"][i]   , useless_stats)
        artifact_value[i][3] = get_artifact_value(bag_of_artifacts["goblet"][i] , useless_stats)
        artifact_value[i][4] = get_artifact_value(bag_of_artifacts["circlet"][i], useless_stats)

    # Converte bag_of_artifacts para np.array() para ser mais conveniente no loop principal
    flowers  = np.array(bag_of_artifacts["flower"])
    feathers = np.array(bag_of_artifacts["feather"])
    sands    = np.array(bag_of_artifacts["sand"])
    goblets  = np.array(bag_of_artifacts["goblet"])
    circlets = np.array(bag_of_artifacts["circlet"])

    # Guarda o melhor indivíduo
    best_fitness_so_far = 0
    best_fitness = 0
    best_index = -1
    best_individual = None

    # Loop principal
    while True:

        # Critérios de parada
        if not (best_fitness < target_fitness):
            stop_by = "Reach fitness target"
            break
        if not (generation_count < MAX_GENERATIONS):
            stop_by = "Reach MAX_GENERATIONS"
            break
        if not (no_change_count < MAX_NO_CHANGE_GENERATIONS):
            stop_by = "Reach MAX_NO_CHANGE_GENERATIONS"
            break

        # Calcula a probabilidade das escolhas
        pheromone_alpha = vectorized_apply_alpha(pheromone)
        artifact_value_beta = vectorized_apply_beta(artifact_value)
        numerator = np.multiply(pheromone_alpha[:, :], artifact_value_beta[:, :])
        denominator = np.zeros(shape=(1, 5))
        # np.apply_along_axis(lambda x: print(x), 0, zip(pheromone_alpha, artifact_value_beta))
        for column in np.arange(5):
            denominator[0, column] = np.sum(np.multiply(pheromone_alpha[:, column], artifact_value_beta[:, column]))
        probability = np.divide(numerator, denominator) # Matrix (300, 5)

        # Dataframe com a probabilidade cumulativa para fazer escolhas com o método da roleta
        # temp_df = pd.DataFrame()
        # temp_df["0"] = probability[:, 0].cumsum()
        # temp_df["1"] = probability[:, 1].cumsum()
        # temp_df["2"] = probability[:, 2].cumsum()
        # temp_df["3"] = probability[:, 3].cumsum()
        # temp_df["4"] = probability[:, 4].cumsum()
        
        # Faz 5 escolhas para cada formiga
        # for k in np.arange(POP_SIZE):
        #     for choice in np.arange(PATHS_TO_CHOSSE):
        #         random_num = np.random.rand(1)[0]
        #         temp = temp_df[str(choice)]
        #         temp = temp.append(pd.Series(random_num))
        #         temp.sort_values(ascending = True, inplace = True)
        #         temp.reset_index(drop = True, inplace = True)
        #         temp = np.where(temp.values == random_num)[0][0]
        #         choices[k, choice] = temp

        #     ants[k, 0] = bag_of_artifacts["flower"][temp]
        #     ants[k, 1] = bag_of_artifacts["feather"][temp]
        #     ants[k, 2] = bag_of_artifacts["sand"][temp]
        #     ants[k, 3] = bag_of_artifacts["goblet"][temp]
        #     ants[k, 4] = bag_of_artifacts["circlet"][temp]

        # Faz 5 escolhas para cada formiga (total de 100 = 300 * 5 escolhas)
        choices[:, 0] = np.random.choice(np.arange(POP_SIZE), POP_SIZE, replace = True, p = probability[:, 0]) # Escolhe a flor de cada formiga
        choices[:, 1] = np.random.choice(np.arange(POP_SIZE), POP_SIZE, replace = True, p = probability[:, 1]) # Escolhe a pena de cada formiga
        choices[:, 2] = np.random.choice(np.arange(POP_SIZE), POP_SIZE, replace = True, p = probability[:, 2]) # Escolhe a ampulheta de cada formiga
        choices[:, 3] = np.random.choice(np.arange(POP_SIZE), POP_SIZE, replace = True, p = probability[:, 3]) # Escolhe o cálice de cada formiga
        choices[:, 4] = np.random.choice(np.arange(POP_SIZE), POP_SIZE, replace = True, p = probability[:, 4]) # Escolhe a tiara de cada formiga
        
        ants[:, 0] = np.take(flowers , choices[:, 0])
        ants[:, 1] = np.take(feathers, choices[:, 1])
        ants[:, 2] = np.take(sands   , choices[:, 2])
        ants[:, 3] = np.take(goblets , choices[:, 3])
        ants[:, 4] = np.take(circlets, choices[:, 4])

        # Transformas os caminhos escolhidos em builds e calcula o fitness
        builds = np.apply_along_axis(lambda row: Build(*row), 1, ants)
        generation_fitness = vectorized_fitness(builds)

        # Atualiza o melhor indivíduo encontrado
        best_fitness = np.max(generation_fitness)
        if best_fitness > best_fitness_so_far:
            best_fitness_so_far = best_fitness
            best_index = np.argmin(generation_fitness)
            best_individual = builds[best_index]
            no_change_count = 0

        # Update matrix feromônio
        for i in np.arange(300):
            delta_pheromone[i, 0] = artifact_value[choices[i, 0], 0]/5 * (generation_fitness[i] / best_fitness)
            delta_pheromone[i, 1] = artifact_value[choices[i, 1], 1]/5 * (generation_fitness[i] / best_fitness)
            delta_pheromone[i, 2] = artifact_value[choices[i, 2], 2]/5 * (generation_fitness[i] / best_fitness)
            delta_pheromone[i, 3] = artifact_value[choices[i, 3], 3]/5 * (generation_fitness[i] / best_fitness)
            delta_pheromone[i, 4] = artifact_value[choices[i, 4], 4]/5 * (generation_fitness[i] / best_fitness)
        pheromone = (1 - VAPE_RATE) * pheromone + delta_pheromone

        generation_count += 1
        no_change_count += 1

        # Log das informações da geração
        sum_fitness = generation_fitness.sum()
        logs.append(f'best_fitness:{best_fitness}|best_individual:{best_individual}|sum:{sum_fitness}|mean:{sum_fitness / POP_SIZE}\n')

    return {
        'best_fitness': best_fitness_so_far,
        'best_individual': best_individual,
        'best_index': best_index,
        'stop_by': stop_by,
        'logs': logs
    }

    
print("ÍNÍCIO")
if __name__ == '__main__':
    INPUTS = ["Hu Tao + Báculo de Homa R1 + Habilidade Elemental + Ataque Carregado", "Zhongli + Borla Preta R5 + Dano de Absorção do Escudo"]
    USELESS_STATS = [["DEF", "DEF%", "ER", "Hydro", "Cryo", "Electro", "Geo", "Anemo", "Healing"], ["DEF", "DEF%", "Hydro", "Cryo", "Electro", "Pyro", "Anemo", "Healing"]]
    file_names = ["first", "second"]
    functions = [fitness_1, fitness_2]
    targets = [30000, 50000] # 30k dano e 50k absorção de escudo

    seeds = [seed for seed in range(0, 30, 1)]

    for function, target, file_name, useless_stats, input_ in \
        zip(functions, targets, file_names, USELESS_STATS, INPUTS):
            
        for seed in seeds:
            result = ACO(function, target, useless_stats, seed)

            print("Resultados")
            print(f"Combinação: {input_}")
            print(f"Melhor output da build: { result['best_fitness'] }")
            print(f"Melhor build: { result['best_individual'] }")
            print(f"Index da build: { result['best_index'] }")
            print(f"Stop by: { result['stop_by'] }")
            # print(f"Log: { result['logs'] }")
            print("=------------=")

            # Escreve num arquivo os resultados
            with open(f"ACO_{file_name}_{seed}.txt", "w") as f:
                for log in result['logs']:
                    f.writelines(log)
print("FIM")
