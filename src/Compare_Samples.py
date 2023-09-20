import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore

plt.style.use(style='Solarize_Light2')

SAMPLES = 30
ALGORITHMS = 4
FITNESSESS = 2


def pre_process_data() -> tuple[pd.DataFrame, list[str]]:

    df = pd.DataFrame()
    file_names = []
    best_sample_fitnessess = []

    algorithms = ["GA"] * 2 * SAMPLES + ["ES"] * 2 * SAMPLES + ["BA"] * 2 * SAMPLES + ["ACO"] * 2 * SAMPLES
    seeds = [i for i in range(30)] * 8
    fitnessess = ["first"] * SAMPLES + ["second"] * SAMPLES
    fitnessess = fitnessess * 4

    for algorithm, fitness, seed in zip(algorithms, fitnessess, seeds):
        with open(f"samples/{algorithm}_{fitness}_{seed}.txt", 'r') as f:

            all_generations_best_fitnessess = np.zeros(shape=[5000], dtype=int)
            for i, line in enumerate(f.readlines()):
                line = line.split("|")
                best_fitness = int(line[0].split(":")[1].split(".")[0])
                all_generations_best_fitnessess[i] = best_fitness
            
            file_names.append(f"{algorithm}_{fitness}_{seed}")
            best_sample_fitnessess.append(np.max(all_generations_best_fitnessess))

    sample_names = list(map(lambda name: name[:-2], file_names[:SAMPLES*8:SAMPLES])) # ['GA_first_0', 'GA_second_0', 'ES_first_0', 'ES_second_0', 'BA_first_0', 'BA_second_0', 'ACO_first_0', 'ACO_second_0']

    # for i, sample_name in ['GA_first_0', 'GA_second_0', 'ES_first_0', 'ES_second_0', 'BA_first_0', 'BA_second_0', 'ACO_first_0', 'ACO_second_0']:
    #     df[sample_name] = best_sample_fitnessess[SAMPLES*i:SAMPLES*(i+1)]

    df['GA_first']   = best_sample_fitnessess[SAMPLES*0:SAMPLES*1]
    df['GA_second']  = best_sample_fitnessess[SAMPLES*1:SAMPLES*2]
    df["ES_first"]   = best_sample_fitnessess[SAMPLES*2:SAMPLES*3]
    df["ES_second"]  = best_sample_fitnessess[SAMPLES*3:SAMPLES*4]
    df["BA_first"]   = best_sample_fitnessess[SAMPLES*4:SAMPLES*5]
    df["BA_second"]  = best_sample_fitnessess[SAMPLES*5:SAMPLES*6]
    df["ACO_first"]  = best_sample_fitnessess[SAMPLES*6:SAMPLES*7]
    df["ACO_second"] = best_sample_fitnessess[SAMPLES*7:SAMPLES*8]

    # df["file_names"]             = file_names
    # df["best_sample_fitnessess"] = best_sample_fitnessess

    return df, sample_names



if __name__ == '__main__':
    df, sample_names = pre_process_data()

    print(df)
    print(df.describe())
    # df["GA_first"].plot.hist()
    # plt.show()

    s = 0
    for sample_name in sample_names:
        s += df[sample_name].sum()
    global_mean = np.divide(s, (SAMPLES*8))

    means = {
        "GA_first": 0,
        "GA_second": 0,
        "ES_first": 0,
        "ES_second": 0,
        "GA_first": 0,
        "GA_second": 0,
        "ES_first": 0,
        "ES_second": 0
    }

    # Global F
    SSW = 0
    SSB = 0
    for sample_name in sample_names:
        mean = df[sample_name].mean()
        means[sample_name] = mean
        print(f"{sample_name} - média {mean} - variância {np.var(df[sample_name].to_numpy()):.2f}")
        SSW += np.sum(np.apply_along_axis(lambda observation:  np.power(observation - mean, 2), 0, df[sample_name].to_numpy()))
        SSB += np.sum(np.apply_along_axis(lambda observation:  np.power(observation - global_mean, 2), 0, df[sample_name].to_numpy()))

    global_F = np.divide(SSB, SSW)
    print(f"F global: {global_F}")
    SSW = 0
    SSB = 0


    # F do GA_1 com ES_1
    # F do GA_2 com ES_2
    # F do BA_1 com ES_1
    # F do BA_2 com ES_2
    # F do ACO_1 com ES_1
    # F do ACO_2 com ES_2
    locals_F = []
    for algo1, algo2, fit in zip(
        ["GA", "BA", "ACO"] *2,
        ["ES"] *6,
        ["first"] *3 + ["second"] *3
    ):
        global_mean = np.divide(df[f"{algo1}_{fit}"].sum() + df[f"{algo2}_{fit}"].sum(), SAMPLES*2)
        SSW = np.sum(np.apply_along_axis(lambda observation:  np.power(observation - means[f"{algo1}_{fit}"], 2), 0, df[f"{algo1}_{fit}"].to_numpy())) + \
                np.sum(np.apply_along_axis(lambda observation:  np.power(observation - means[f"{algo2}_{fit}"], 2), 0, df[f"{algo2}_{fit}"].to_numpy()))
        SSB = np.sum(np.apply_along_axis(lambda observation:  np.power(observation - global_mean, 2), 0, df[f"{algo1}_{fit}"].to_numpy())) + \
                np.sum(np.apply_along_axis(lambda observation:  np.power(observation - global_mean, 2), 0, df[f"{algo2}_{fit}"].to_numpy()))
        local_F = np.divide(SSB, SSW)
        print(f"F entre {algo1}_{fit} e {algo2}_{fit}: {local_F}", end="\n")
        locals_F.append(local_F)

    print(locals_F)


    # print(df.apply(zscore))

    # Faz o plot da comparação entre os algoritmos
    OLD_COMPARISSON = False
    if (OLD_COMPARISSON):
        fig, axis = plt.subplots(2, 2, figsize = (15, 8))
        fig.subplots_adjust(hspace=0.5)

        steps = [500, 100, 500, 100]
        params = [
            (0, 0, 0, 30, 22500, 25000, steps[0], "GA", "Fitness 1", range(22500, 25000, steps[0])), 
            (0, 1, 30, 60, 27900, 28500, steps[1], "GA", "Fitness 2", range(27900, 28500, steps[1])), 
            (1, 0, 60, 90, 20000, 23000, steps[2], "ES", "Fitness 1", range(20000, 23000, steps[2])), 
            (1, 1, 90, 120, 27200, 28000, steps[3], "ES", "Fitness 2", range(27200, 28000, steps[3]))
        ]
        for i, j, start, end, xlim_min, xlim_max, step, algo_name, fitness_name, b in params:

            axis[i][j].hist(df["best_sample_fitnessess"][start:end], bins=b)
            # axis[i][j].lines(x = df["best_sample_fitnessess"][start:end])

            axis[i][j].set(xlim=(xlim_min, xlim_max), xticks = np.arange(xlim_min, xlim_max+1, step),
                    ylim=(0, 22), yticks = np.arange(1, 22+1, 2))
            axis[i][j].set_title(f"Histograma: {algo_name}_{fitness_name}")
            axis[i][j].set_xlabel("Fitness")
            axis[i][j].set_ylabel("Frequência")
        plt.show()
        exit()

    fig, axis = plt.subplots(2, 2, figsize = (15, 8))
    


