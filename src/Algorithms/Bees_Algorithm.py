from typing import Callable

from Base_Classes.Artifact import *
from Base_Classes.Build import *

import numpy as np
# from numpy.random import rand as np_random_rand
# from numpy import max as np_max, argmax as np_argmax
import pandas as pd

from codetiming import Timer


# Cria as builds (indivíduos) da população 
def create_population(database: list[Artifact], pop_size: int) -> tuple[list[Build], dict[str, np.ndarray]]:

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

    builds = np.array([], dtype = Build)
    for i in np.arange(pop_size):
        b = Build(flowers[i], feathers[i], sands[i], goblets[i], circlets[i])
        builds = np.append(builds, b, axis = None)

    bag_of = {
        "flower": flowers,
        "feather": feathers,
        "sand": sands,
        "goblet": goblets,
        "circlet": circlets,
    }

    return builds, bag_of



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



def neighboorhood_search(start_search_point: Build,
                         artifact_index: int, 
                         bag_of: dict[str, np.ndarray],
                         ngh: int,
                         pop_size: int
                        ) -> Build:
    # recruit_bee = [] # build criada com base na abelha que encontrou o flower patch

    # for artifact_type in ["flower", "feather", "sand", "goblet", "circlet"]:
    artifact_type = np.random.choice(["flower", "feather", "sand", "goblet", "circlet"], 1, replace=False)[0]
    recruit_index = (artifact_index + np.random.randint(-ngh, +ngh+1)) % pop_size
    # print(f"artifact index -> {artifact_index} | recruit index {recruit_index}")
    # recruit_bee.append(bag_of[artifact_type][recruit_index])
    
    start_search_point[artifact_type] = bag_of[artifact_type][recruit_index] # Modifica um artefato da build inicial
    return start_search_point # Build(*recruit_bee)

"""
fitness -> Função fitness que será usada
target_fitness -> Um dos objetivos de parada
"""
@Timer(name="decorator", text="Tempo da busca: {:.4f} segundos")
def Bees_Algorithm(fitness: Callable, 
                   target_fitness: int, 
                   seed: int):

    # Parâmetros
    RANDOM_SEED = seed
    MAX_GENERATIONS = 10000
    MAX_NO_CHANGE_GENERATIONS = 1000
    POP_SIZE = 250
    SEARCH_GROUPS = 30
    ELITE_SEARCH_GROUPS = 1
    NORMAL_SEARCH_GROUPS = SEARCH_GROUPS - ELITE_SEARCH_GROUPS
    ELITE_GROUP_RECRUITS = 10
    NORMAL_GROUP_RECRUITS = 8
    NGH = 5 # Neiborhood
    NO_GROUP_BEE = POP_SIZE - (ELITE_SEARCH_GROUPS * ELITE_GROUP_RECRUITS + NORMAL_SEARCH_GROUPS * NORMAL_GROUP_RECRUITS)

    # Configuração inicial
    np.random.seed(seed=RANDOM_SEED)
    artifacts = Artifact.read_database()
    population, bag_of_artifacts = create_population(artifacts, POP_SIZE)
    vectorized_fitness = np.vectorize(fitness)

    generation_count = 0  # Controla a quantidade máxima de gerações
    no_change_count  = 0  # Controla a quantidade de gerações sem mudança
    logs: list[str]  = [] # controla as informações de output

    # Cria as matrizes para guardar as abelhas e fitness
    # Cada linha é um flower patch
    elite_flower_patches = np.zeros(shape=(ELITE_SEARCH_GROUPS, ELITE_GROUP_RECRUITS), dtype = Build)
    flower_patches = np.zeros(shape=(NORMAL_SEARCH_GROUPS, NORMAL_GROUP_RECRUITS), dtype = Build)
    random_scouts = np.zeros(shape=(1, NO_GROUP_BEE), dtype = Build)
    elite_fitness = np.zeros(shape=(ELITE_SEARCH_GROUPS, ELITE_GROUP_RECRUITS))
    normal_fitness = np.zeros(shape=(NORMAL_SEARCH_GROUPS, NORMAL_GROUP_RECRUITS))
    random_fitness = np.zeros(shape=(1, NO_GROUP_BEE))

    # Calcula o fitness da geração atual
    generation_fitness = np.array(list(map(fitness, population)))

    # Guarda o melhor indivíduo
    global_best_fitness = np.max(generation_fitness)
    global_best_index = np.argmax(generation_fitness)
    global_best_individual = population[global_best_index]

    # Separa as abelhas em scout, elite scout e random 
    temp_df = pd.DataFrame()
    temp_df["bees"] = population
    temp_df["Fitness"] = generation_fitness
    sorted_bees = temp_df.sort_values(by=["Fitness"], ascending = False)["bees"]
    elite_scout_bees = sorted_bees[:ELITE_SEARCH_GROUPS].to_numpy()
    scout_bees = sorted_bees[ELITE_SEARCH_GROUPS:SEARCH_GROUPS].to_numpy()
    random_scouts = sorted_bees[SEARCH_GROUPS:].to_numpy()

    # Preenche as abelhas iniciais nos flower patches
    for i in range(ELITE_SEARCH_GROUPS): 
        elite_flower_patches[i][0] = elite_scout_bees[i]
    for i in range(NORMAL_SEARCH_GROUPS): 
        flower_patches[i][0] = scout_bees[i]


    # Loop principal
    while True:

        # Critérios de parada
        if not (global_best_fitness < target_fitness):
            stop_by = "Reach fitness target"
            break
        if not (generation_count < MAX_GENERATIONS):
            stop_by = "Reach MAX_GENERATIONS"
            break
        if not (no_change_count < MAX_NO_CHANGE_GENERATIONS):
            stop_by = "Reach MAX_NO_CHANGE_GENERATIONS"
            break

        # Explora a vizinhança do espaço de busca com as abelhas recrutadas
        for i in np.arange(ELITE_SEARCH_GROUPS):
            elite_flower_patch: np.ndarray = elite_flower_patches[i]
            best_elite_scout_bee: Build = elite_flower_patch[0] # Escolhe a abelha "inicial" do flower patch
            artifact_index = np.argmax(np.array(best_elite_scout_bee["flower"]) == bag_of_artifacts["flower"])

            for recruit_index in np.arange(1, ELITE_GROUP_RECRUITS):
                elite_flower_patch[recruit_index] = neighboorhood_search(best_elite_scout_bee, artifact_index, bag_of_artifacts, NGH, POP_SIZE) # [1] 
                # elite_fitness[i][recruit_index] = fitness(elite_flower_patch[recruit_index]) # [2]

        for i in np.arange(NORMAL_SEARCH_GROUPS):
            flower_patch: np.ndarray = flower_patches[i]
            best_scout_bee: Build = flower_patch[0]
            artifact_index = np.argmax(np.array(best_scout_bee["flower"]) == bag_of_artifacts["flower"])

            for recruit_index in np.arange(1, NORMAL_GROUP_RECRUITS):
                flower_patch[recruit_index] = neighboorhood_search(best_scout_bee, artifact_index, bag_of_artifacts, NGH, POP_SIZE)

        

        # Calcula o fitness da geração atual
        elite_fitness = vectorized_fitness(elite_flower_patches)
        normal_fitness = vectorized_fitness(flower_patches)

        # Ordena pelo fitness para cada flower patch
        for i in np.arange(ELITE_SEARCH_GROUPS):
            temp_df = pd.DataFrame()
            temp_df["bees"] = elite_flower_patches[i]
            temp_df["fitness"] = elite_fitness[i]

            temp_df.sort_values(by=["fitness"], ascending = False)

            elite_flower_patches[i] = temp_df["bees"].to_numpy()
            elite_fitness[i] = temp_df["fitness"].to_numpy()

        for i in np.arange(NORMAL_SEARCH_GROUPS):
            temp_df = pd.DataFrame()
            temp_df["bees"] = flower_patches[i]
            temp_df["fitness"] = normal_fitness[i]

            temp_df.sort_values(by=["fitness"], ascending = False)

            flower_patches[i] = temp_df["bees"].to_numpy()
            normal_fitness[i] = temp_df["fitness"].to_numpy()

        
        # Atualiza o melhor indivíduo
        for i in np.arange(ELITE_SEARCH_GROUPS):
            patch_best_fitness = elite_fitness[i][0]

            if patch_best_fitness > global_best_fitness:
                global_best_fitness = patch_best_fitness
                global_best_individual = elite_flower_patches[i][0]
                global_best_index = (i, 'elite') # Guarda a patch/linha que possui a melhor abelha/build
                no_change_count = 0

        for i in np.arange(NORMAL_SEARCH_GROUPS):
            patch_best_fitness = normal_fitness[i][0]

            if patch_best_fitness > global_best_fitness:
                global_best_fitness = patch_best_fitness
                global_best_individual = flower_patches[i][0]
                global_best_index = (i, 'normal') # Guarda a patch/linha que possui a melhor abelha/build
                no_change_count = 0

        # Tratamento das abelhas sem grupo
        for i in np.arange(NO_GROUP_BEE):
            artifact_index = np.argmax(np.array(random_scouts[i]["flower"]) == bag_of_artifacts["flower"])
            random_scouts[i] = neighboorhood_search(random_scouts[i], artifact_index, bag_of_artifacts, NGH, POP_SIZE) # [1] 
        random_fitness = vectorized_fitness(random_scouts)
            
        generation_count += 1
        no_change_count += 1

        # Log das informações da geração
        sum_fitness = np.concatenate(elite_fitness).sum() + np.concatenate(normal_fitness).sum() + random_fitness.sum()
        mean_fitness = sum_fitness / POP_SIZE
        logs.append(f'best_fitness:{global_best_fitness}|best_individual:{global_best_individual}|sum:{sum_fitness}|mean:{mean_fitness}\n')

    return {
        'best_fitness': global_best_fitness,
        'best_individual': global_best_individual,
        'best_index': global_best_index,
        'stop_by': stop_by,
        'logs': logs
    }




print("ÍNÍCIO")
if __name__ == '__main__':
    INPUTS = ["Hu Tao + Báculo de Homa R1 + Habilidade Elemental + Ataque Carregado", "Zhongli + Borla Preta R5 + Dano de Absorção do Escudo"]
    file_names = ["first", "second"]
    functions = [fitness_1, fitness_2]
    targets = [30000, 50000] # 30k dano e 50k absorção de escudo

    seeds = [seed for seed in range(0, 30, 1)]

    for function, target, file_name, input_ in \
        zip(functions, targets, file_names, INPUTS):
            
        for seed in seeds:
            result = Bees_Algorithm(function, target, seed)

            print("Resultados")
            print(f"Combinação: {input_}")
            print(f"Melhor output da build: { result['best_fitness'] }")
            print(f"Melhor build: { result['best_individual'] }")
            print(f"Index da build: { result['best_index'] }")
            print(f"Stop by: { result['stop_by'] }")
            # print(f"Log: { result['logs'] }")
            print("=------------=")

            # Escreve num arquivo os resultados
            with open(f"BA_{file_name}_{seed}.txt", "w") as f:
                for log in result['logs']:
                    f.writelines(log)
print("FIM")
