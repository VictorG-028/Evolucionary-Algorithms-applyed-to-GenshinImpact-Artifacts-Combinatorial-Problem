import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use(style='Solarize_Light2')

def pre_process_data() -> pd.DataFrame:

    df = pd.DataFrame()
    file_names = []
    best_sample_fitnessess = []

    with open("Best_Samples_Fitness.txt", 'r') as f:

        for line in f.readlines():
            line = line.split()

            file_names.append(line[4])
            best_sample_fitnessess.append(line[6])

        df["file_names"]             = file_names
        df["best_sample_fitnessess"] = best_sample_fitnessess
        df["best_sample_fitnessess"] = df["best_sample_fitnessess"].apply(lambda num: int(num[:-2]))
    
    return df
        
df = pre_process_data()
# df["best_sample_fitnessess"].astype("int").hist()



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

