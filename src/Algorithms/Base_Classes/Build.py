from Base_Classes.Artifact import * # Descomentar quando usar um algoritmo
# from Algorithms.Base_Classes.Artifact import * # Descomentar quando usar PlotSamples ou Compara_Samples



class Sheet(dict):
    ATK: float
    ATK_Percentage: float
    HP: float
    HP_Percentage: float
    DEF: float
    DEF_Percentage: float
    EM: float
    ER: float
    CR: float
    CD: float
    Physical: float
    Anemo: float
    Geo: float
    Cryo: float
    Pyro: float
    Hydro: float
    Electro: float
    Healing: float

    def __init__(self, sheet: dict) -> None:

        for key in sheet.keys():
            self[key] = sheet[key]



class Build(dict):
    flower: Artifact
    feather: Artifact
    sand: Artifact
    goblet: Artifact
    circlet: Artifact

    def __init__(self, 
                flower: Artifact, 
                feather: Artifact, 
                sand: Artifact, 
                goblet: Artifact, 
                circlet: Artifact
                ) -> None:
        assert flower.type_  == "Flower"  \
           and feather.type_ == "Feather" \
           and sand.type_    == "Sand"    \
           and goblet.type_  == "Goblet"  \
           and circlet.type_ == "Circlet"
        self["flower"]  = flower
        self["feather"] = feather
        self["sand"]    = sand
        self["goblet"]  = goblet
        self["circlet"] = circlet
    
    def __str__(self) -> str:
        return f'[{str(self["flower"])}-{str(self["feather"])}-{str(self["sand"])}-{str(self["goblet"])}-{str(self["circlet"])}]'



    def get_artifact_sheet(self) -> Sheet:

        sheet = {
            "ATK": 0,
            "ATK%": 0,
            "HP": 0,
            "HP%": 0,
            "DEF": 0,
            "DEF%": 0,
            "EM": 0,
            "ER": 0,
            "CR": 0,
            "CD": 0,
            "Physical": 0,
            "Anemo": 0,
            "Geo": 0,
            "Cryo": 0,
            "Pyro": 0,
            "Hydro": 0,
            "Electro": 0,
            "Healing": 0
        }


        sheet[self["flower"].main_stat.type_] += self["flower"].main_stat.value
        sheet[self["feather"].main_stat.type_] += self["feather"].main_stat.value
        sheet[self["sand"].main_stat.type_] += self["sand"].main_stat.value
        sheet[self["goblet"].main_stat.type_] += self["goblet"].main_stat.value
        sheet[self["circlet"].main_stat.type_] += self["circlet"].main_stat.value

        all_substats: list[SubStat] = []
        all_substats += self["flower"].sub_stats
        all_substats += self["feather"].sub_stats
        all_substats += self["sand"].sub_stats
        all_substats += self["goblet"].sub_stats
        all_substats += self["circlet"].sub_stats

        for substat in all_substats:
            sheet[substat.type_] += substat.value

        return Sheet(sheet)

