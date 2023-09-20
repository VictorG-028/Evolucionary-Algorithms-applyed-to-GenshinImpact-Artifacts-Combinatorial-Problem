from __future__ import annotations
from collections.abc import Generator

from numpy import random
from os import mkdir, path

##################################
### Start of const global vars ###

# Fonte dos valores: https://genshin-impact.fandom.com/wiki/Artifacts/Stats
POSSIBLE_SUB_STATS = [
    'HP', 'HP%', 'ATK', 'ATK%', 'DEF', 'DEF%', 'EM', 'ER', 'CR', 'CD'
]
POSSIBLE_PROCS = {
    'HP':   [209.13, 239.00, 268.88, 298.75],
    'HP%':  [4.08, 4.66, 5.25, 5.83],
    'ATK':  [13.62, 15.56, 17.51, 19.45],
    'ATK%': [4.08, 4.66, 5.25, 5.83],
    'DEF':  [16.20, 18.52, 20.83, 23.15],
    'DEF%': [5.10, 5.83, 6.56, 7.29],
    'EM':   [16.32, 18.65, 20.98, 23.31],
    'ER':   [4.53, 5.18, 5.83, 6.48],
    'CR':   [2.72, 3.11, 3.50, 3.89],
    'CD':   [5.44, 6.22, 6.99, 7.77],
}

POSSIBLE_SAND_MAIN_STATS = {
    'HP%':  46.6,
    'DEF%': 58.3,
    'ATK%': 46.6,
    'EM':   187,
    'ER':   51.8,
}

POSSIBLE_GOBLET_MAIN_STATS = {
    'HP%':      46.6,
    'DEF%':     58.3,
    'ATK%':     46.6,
    'EM':       187,
    'Physical': 58.3,
    'Anemo':    46.6,
    'Geo':      46.6,
    'Electro':  46.6,
    'Hydro':    46.6,
    'Pyro':     46.6,
    'Cryo':     46.6,
}

POSSIBLE_CIRCLET_MAIN_STATS = {
    'HP%':     46.6,
    'DEF%':    58.3,
    'ATK%':    46.6,
    'EM':      187,
    'CR':      31.1,
    'CD':      62.2,
    'Healing': 35.9,
}

# ms = MainStat(4780, 'HP')
# ss1 = SubStat(9.9 , 'HP%')
# ss2 = SubStat(10.3, 'ER')
# ss3 = SubStat(13.2, 'CD')
# ss4 = SubStat(44  , 'DEF')

DATABASES_NAMES = ["Flowers", "Feathers", "Sands", "Goblets", "Circlets"]
ARTIFACTS_PER_SLOT = 300 # Combination -> artifacts_per_slot^5
CREATE_NEW_DATABASE = False
MAKE_TEST = False

### End of const global vars ###
################################
###     Start of classes     ###

class MainStat:
    value: float
    type_: str  # 'HP%'  | 'HP'   | 'ATK' | 
                # 'ATK%' | 'DEF%' | 'EM'  | 
                #  'ER'  | 'CR'   | 'CD'  | 
           # 'Physical' | 'Anemo'| 'Geo' | 
           # 'Electro'  | 'hydro'| 'Pyro'| 
           # 'Cryo'     | 'Healing'

    def __init__(self, value: float, type_: str) -> None:
        assert type_ in ['HP%', 'HP', 'ATK', 'ATK%', 'DEF%', 'EM', 'ER', 'CR', 'CD', 'Physical', 'Anemo', 'Geo', 'Electro', 'Hydro', 'Pyro', 'Cryo', 'Healing']
        self.value = value
        self.type_ = type_

    def __str__(self) -> str:
        return f"[{self.value:_^.2f} {self.type_}]"
    
    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, self.__class__):
            checks = [
                round(self.value, ndigits = 2) == round(__o.value, ndigits = 2),
                self.type_ == __o.type_
            ]

            return all(checks)
        else: 
            return False



class SubStat:
    value: float
    type_: str      # 'HP%' | 'HP'  | 'ATK'  | 
                    #'ATK%' | 'DEF' | 'DEF%' | 
                    #  'EM' | 'ER'  | 'CR'   | 
                    #  'CD'

    def __init__(self, value: float, type_: str) -> None:
        assert type_ in ['HP%', 'HP', 'ATK', 'ATK%', 'DEF', 'DEF%', 'EM', 'ER', 'CR', 'CD']
        self.value = value
        self.type_ = type_

    def __str__(self) -> str:
        return f"[{self.value:_^.2f} {self.type_}]"
    
    def __repr__(self) -> str:
        return f"[{self.value:_^.2f} {self.type_}]"

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, self.__class__):
            checks = [
                round(self.value, ndigits = 2) == round(__o.value, ndigits = 2),
                self.type_ == __o.type_
            ]

            return all(checks)
        else: 
            return False



class Artifact:

    type_: str # 'Flower' | 'Feather' | 
               # 'Sand'   | 'Goblet'  | 'Circlet'
    main_stat: MainStat   
    sub_stats: list[SubStat, SubStat, SubStat, SubStat]
    mutation_rate: float # Usado apenas no ES

    def __init__(self, type_: str, main_stat: MainStat, sub_stats: list[SubStat, SubStat, SubStat, SubStat]) -> None:
        assert type_ in ['Flower', 'Feather', 'Sand', 'Goblet', 'Circlet']
        self.type_ = type_
        self.main_stat = main_stat
        self.sub_stats = sub_stats

    def __str__(self) -> str:
        return f"{self.main_stat}~{'~'.join([str(s) for s in self.sub_stats])}"

    def __repr__(self) -> str:
        return f"{self.type_}{self.main_stat}"

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, self.__class__):
            checks = [
                self.type_ == __o.type_,
                self.main_stat == __o.main_stat,
                self.sub_stats == __o.sub_stats
            ]

            return all(checks)
        else: 
            return False



    """ 
    ban_substat -> substat that is already assigned in the main stat can't 
                    be assigned again
    returns 4 substats (Yield each substat in a generator fashion)
    """
    @staticmethod
    def generate_substats(ban_substat: str) -> Generator[SubStat]:
        filtered_substats = [
            substat for substat in POSSIBLE_SUB_STATS if substat != ban_substat
        ]
        rolls = 4 + random.randint(1, size=1) # 4 or 5 rolls
        substats_types: list[str] = list(
            random.choice(filtered_substats, 4, replace = False)
        )
        proc_types: list[str] = list(
            random.choice(substats_types, rolls, replace = True)
        )

        for substat_type in substats_types:
            procs = substats_types.count(substat_type) \
                    + proc_types.count(substat_type)
            value = sum(random.choice(POSSIBLE_PROCS[substat_type], procs))
            yield SubStat(value, substat_type)



    """ 
    n -> how many flowers are going to be generated 
    returns an instance of Artifact of type_ "Flower" with randonly generated 
    stats using Genshin Impact rules
    """
    @staticmethod
    def generate_flowers(n: int = 1) -> Artifact | list[Artifact]:
        flowers = []
        flower: Artifact

        for _ in range(n):
            ban = 'HP'
            substats = [substat for substat in Artifact.generate_substats(ban)]

            flower = Artifact('Flower', MainStat(4780, 'HP'), substats)
            flowers.append(flower)

        return flowers if n > 1 else flower



    """ 
    n -> how many feathers are going to be generated 
    returns an instance of Artifact of type_ "Feather" with randonly generated stats using Genshin Impact rules
    """
    @staticmethod
    def generate_feathers(n: int = 1) -> Artifact:
        feathers = []
        feather: Artifact

        for _ in range(n):
            ban = 'ATK'
            substats = [substat for substat in Artifact.generate_substats(ban)]

            feather = Artifact('Feather', MainStat(311, 'ATK'), substats)
            feathers.append(feather)

        return feathers if n > 1 else feather


    """ 
    n -> how many sands are going to be generated 
    returns an instance of Artifact of type_ "Sand" with randonly generated stats using Genshin Impact rules
    """
    @staticmethod
    def generate_sands(n: int = 1) -> Artifact:
        sands = []
        sand: Artifact
        
        for _ in range(n):
            main_stat_type: str = random.choice(list(POSSIBLE_SAND_MAIN_STATS.keys()), 1)[0]
            main_stat_value: float = POSSIBLE_SAND_MAIN_STATS[main_stat_type]
            ban = main_stat_type
            substats = [substat for substat in Artifact.generate_substats(ban)]

            sand = Artifact('Sand', MainStat(main_stat_value, main_stat_type), substats)
            sands.append(sand)

        return sands if n > 1 else sand



    """ 
    n -> how many goblets are going to be generated 
    returns an instance of Artifact of type_ "Goblet" with randonly generated stats using Genshin Impact rules
    """
    @staticmethod
    def generate_goblets(n: int = 1) -> Artifact:
        goblets = []
        goblet: Artifact
        
        for _ in range(n):
            main_stat_type: str = random.choice(list(POSSIBLE_GOBLET_MAIN_STATS.keys()), 1)[0]
            main_stat_value: float = POSSIBLE_GOBLET_MAIN_STATS[main_stat_type]
            ban = main_stat_type
            substats = [substat for substat in Artifact.generate_substats(ban)]

            goblet = Artifact('Goblet', MainStat(main_stat_value, main_stat_type), substats)
            goblets.append(goblet)

        return goblets if n > 1 else goblet



    """ 
    n -> how many circlet are going to be generated 
    returns an instance of Artifact of type_ "Circlet" with randonly generated stats using Genshin Impact rules
    """
    @staticmethod
    def generate_circlets(n: int = 1) -> Artifact:
        circlets = []
        circlet: Artifact
        
        for _ in range(n):
            main_stat_type: str = random.choice(list(POSSIBLE_CIRCLET_MAIN_STATS.keys()), 1)[0]
            main_stat_value: float = POSSIBLE_CIRCLET_MAIN_STATS[main_stat_type]
            ban = main_stat_type
            substats = [substat for substat in Artifact.generate_substats(ban)]

            circlet = Artifact('Circlet', MainStat(main_stat_value, main_stat_type), substats)
            circlets.append(circlet)

        return circlets if n > 1 else circlet


    """
    artifact -> str() representation of the artifact
    type_ -> type of the artifact to be created
    return an instance of Artifact created from a str
    """
    @staticmethod
    def create_artifact_from_str(artifact: str, type_: str) -> Artifact:
        artifact = artifact.split("~")

        main_stat = artifact.pop(0)[1:-1].split()
        main_stat = MainStat(float(main_stat[0]), main_stat[1])

        substats = []
        for _ in range(4):
            substat = artifact.pop(0)[1:-1].split()
            substats.append(SubStat( float(substat[0]), substat[1] ))

        return Artifact(type_, main_stat, substats)



    """
    database -> list of artifacts tha will be stored in .txt format
    artifacts_per_slot -> number of artifacts in each .txt file. Each file stores a specific artifact type.
    return nothing
    """
    @staticmethod
    def save_database(database: list[Artifact], 
                      artifacts_per_slot: int, 
                      test_mode: bool = False
                     ) -> None:
        if not path.exists("databases"): 
            mkdir("databases")
        test = "test_" if test_mode else ""
        temp_copy = database.copy()
        for name in DATABASES_NAMES:
            with open(f"databases/{test}{name}_database.txt", "w") as f:
                for _ in range(artifacts_per_slot):
                    artifact = temp_copy.pop(0)
                    f.writelines(str(artifact)+"\n")



    """  
    test_mode -> read from test .txt files
    returns a list containing all artifacts of text file
    """
    @staticmethod
    def read_database(test_mode: bool = False) -> list[Artifact]:
        database: list[Artifact] = []
        test = "test_" if test_mode else ""
        
        for name in DATABASES_NAMES:
            with open(f"databases/{test}{name}_database.txt", "r") as f:
                type_ = name[:-1]
                for artifact in f.readlines():
                    artifact = artifact.replace("\n", "")
                    artifact = Artifact.create_artifact_from_str(artifact, type_)
                    database.append(artifact)
        return database 



    """ 
    self -> current artifact to be used to calculate the diference
    other -> other artifact to calculate diference between them
    returns the amount of different types_
    """
    def diference(self, other: Artifact) -> int:
        
        ammount_of_diferences = 0

        if self.main_stat.type_ == other.main_stat.type_:
            ammount_of_diferences += 1

        for self_substat, other_substat in zip(self.sub_stats, other.sub_stats):
            if self_substat.type_ == other_substat.type_:
                ammount_of_diferences += 1

        return ammount_of_diferences # retorna 0 até 5



    """ 
    self -> current artifact to be used to calculate the ammount of uselles substat
    useless -> list of useless substats to be used as a reference for self artifact
    returns the amount of uselles substat types_
    """
    def count_useless_substats(self, useless: list[str]) -> int:
        
        ammount_of_useless = 0

        if self.main_stat.type_ in useless:
            ammount_of_useless += 1

        for substat in self.sub_stats:
            if substat.type_ in useless:
                ammount_of_useless += 1

        return ammount_of_useless
        


###     End of classes     ###
##############################
###     Start of tests     ###

if (MAKE_TEST):

    flowers = Artifact.generate_flowers(ARTIFACTS_PER_SLOT)
    feathers = Artifact.generate_feathers(ARTIFACTS_PER_SLOT)
    sands = Artifact.generate_sands(ARTIFACTS_PER_SLOT)
    goblets = Artifact.generate_goblets(ARTIFACTS_PER_SLOT)
    circlets = Artifact.generate_circlets(ARTIFACTS_PER_SLOT)


    # for flower in flowers:
    #     print(flower)
    # print("-----------")

    # for feather in feathers:
    #     print(feather)
    # print("-----------")

    # for sand in sands:
    #     print(sand)
    # print("-----------")

    # for goblet in goblets:
    #     print(goblet)
    # print("-----------")

    # for circlet in circlets:
    #     print(circlet)
    # print("-----------")

    # teste = [s for s in Artifact.generate_substats()]
    # print(teste)
    # print("-----------")

    database = flowers + feathers + sands + goblets + circlets
    Artifact.save_database(database, ARTIFACTS_PER_SLOT, MAKE_TEST)
    database_2 = Artifact.read_database(MAKE_TEST)

    print("-----------")
    # print(database)
    for a in database:
        print(a)
    print("-----------")
    # print(database_2)
    for b in database_2:
        print(b)
    print("-----------")

    assert database == database_2


if (CREATE_NEW_DATABASE):
    flowers = Artifact.generate_flowers(ARTIFACTS_PER_SLOT)
    feathers = Artifact.generate_feathers(ARTIFACTS_PER_SLOT)
    sands = Artifact.generate_sands(ARTIFACTS_PER_SLOT)
    goblets = Artifact.generate_goblets(ARTIFACTS_PER_SLOT)
    circlets = Artifact.generate_circlets(ARTIFACTS_PER_SLOT)

    database = flowers + feathers + sands + goblets + circlets
    Artifact.save_database(database, ARTIFACTS_PER_SLOT)
