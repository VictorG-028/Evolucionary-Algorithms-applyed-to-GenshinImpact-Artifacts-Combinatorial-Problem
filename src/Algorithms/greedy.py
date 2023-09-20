from typing import Callable

from Base_Classes.Artifact import *
from Base_Classes.Build import *

import numpy as np
from numpy import max as np_max
from numpy import argmax as np_argmax
import pandas as pd

from codetiming import Timer


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



"""
fitness -> Função fitness que será usada
"""
@Timer(name="decorator", text="Tempo da busca: {:.4f} segundos")
def greedy(fitness: Callable):

    # Configuração inicial
    database = Artifact.read_database()
    vectorized_fitness = np.vectorize(fitness)

    # Separa a base de dados por artefato 
    flowers  = np.array(list(filter(lambda a: a.type_ == 'Flower',  database)), dtype=Artifact)
    feathers = np.array(list(filter(lambda a: a.type_ == 'Feather', database)), dtype=Artifact)
    sands    = np.array(list(filter(lambda a: a.type_ == 'Sand',    database)), dtype=Artifact)
    goblets  = np.array(list(filter(lambda a: a.type_ == 'Goblet',  database)), dtype=Artifact)
    circlets = np.array(list(filter(lambda a: a.type_ == 'Circlet', database)), dtype=Artifact)

    best_fitness = 0
    
    for flower in flowers:
        for feather in feathers:
            for sand in sands:
                for goblet in goblets:
                    # for circlet in circlets:
                    #     b = Build(flower, feather, sand, goblet, circlet)
                    #     current_fitness = fitness(b)

                    #     if current_fitness > best_fitness:
                    #         best_fitness = current_fitness
                    #         best_build = b
                    
                    circlets_build = list(map(lambda circlet: Build(flower, feather, sand, goblet, circlet), circlets))
                    current_fitnessess = vectorized_fitness(circlets_build)
                    current_best_fitness = np_max(current_fitnessess)
                    current_best_build = np_argmax(current_fitnessess)

                    if current_best_fitness > best_fitness:
                        best_fitness = current_best_fitness
                        best_build = current_best_build
    return {
        'best_fitness': best_fitness,
        'best_build': best_build,
    }


print("ÍNÍCIO")
if __name__ == '__main__':
    INPUTS = ["Hu Tao + Báculo de Homa R1 + Habilidade Elemental + Ataque Carregado", "Zhongli + Borla Preta R5 + Dano de Absorção do Escudo"]
    file_names = ["first", "second"]
    functions = [fitness_1, fitness_2]

    for function, file_name, input_ in zip(functions, file_names, INPUTS):
            
            result = greedy(function)

            print("Resultados")
            print(f"Combinação: {input_}")
            print(f"Melhor output da build: { result['best_fitness'] }")
            print(f"Melhor build: { result['best_build'] }")
            print("=------------=")

            # Escreve num arquivo os resultados
            with open(f"greedy_{file_name}.txt", "w") as f:
                f.writeline(f"best_fitness: {result['best_fitness']}\n")
                f.writeline(f"best_build: {result['best_build']}")
print("FIM")
