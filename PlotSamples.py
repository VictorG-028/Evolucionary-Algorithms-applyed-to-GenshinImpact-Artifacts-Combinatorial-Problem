import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Build import Build

plt.style.use(style='Solarize_Light2')



def plot_all_samples():

	N_SAMPLES = 30

	algorithm_names = ["GA", "GA"] * N_SAMPLES + ["ES", "ES"] * N_SAMPLES
	inputs = (["first"] * N_SAMPLES + ["second"] * N_SAMPLES) * 2
	seeds = [str(seed) for seed in range(0, N_SAMPLES, 1)] * 4

	for algo_name, input_, seed in zip(algorithm_names, inputs, seeds):

		file_name = f"{algo_name}_{input_}_{seed}.txt"

		best_fitnessess = []
		best_individuals = []
		sums = []
		means = []
		with open("Samples/" + file_name, "r") as f:
			for line in f.readlines():
				line = line.rstrip("\n").split("|")

				best_fitness = line[0].split(":")[1]
				best_individual = line[1].split(":")[1]
				fitness_sum = line[2].split(":")[1]
				fitness_mean = line[3].split(":")[1]

				best_fitnessess.append(best_fitness)
				best_individuals.append(best_individual)
				sums.append(fitness_sum)
				means.append(fitness_mean)


		df = pd.DataFrame()
		df["build_score"] = np.array(best_fitnessess, dtype = np.float64)
		df["best_build"] = np.array(best_individuals, dtype = Build)
		df["generation_fitness_sum"] = np.array(sums, dtype = np.float64)
		df["generation_fitness_mean"] = np.array(means, dtype = np.float64)

		upper_xlim = len(best_fitnessess)

		if upper_xlim <= 1000:
			x_step = 100
		elif upper_xlim <= 5000:
			x_step = 1000
		elif upper_xlim == 5001:
			x_step = 1667
		elif upper_xlim == 5002:
			x_step = 2001
		else:
			x_step = 2000

		if input_ == "first":
			upper_ylim = 30000
			y_step = 5000
		else:
			upper_ylim = 50000
			y_step = 5000
		
		fig, ax = plt.subplots()
		# x = np.random.uniform(0, 1000, size = len(df["build_score"]))
		x = np.arange(0, upper_xlim)
		y = df["build_score"].round()
		# ax.fill_between(x, y, alpha = .5, linewidth = 0)
		
		print(f"Melhor fitness da sample {file_name} -> {y.max()}")

		ax.plot(x, y, color='#0080FF', label = "fitness", linewidth = 2)
		ax.set(xlim=(0, upper_xlim), xticks = np.arange(0, upper_xlim+1, x_step),
				ylim=(0, upper_ylim), yticks = np.arange(0, upper_ylim+1, y_step))
		plt.title(f"Best Fitness Over Generations\nFile name: {file_name}")
		plt.xlabel(f"Generations ({upper_xlim})")
		plt.ylabel("Fitness")
		plt.legend()
		plt.show()

		# if input("Q para sair: ") == "Q": exit()



def plot_best_samples():

	def get_information_from_sample() -> pd.DataFrame:

		algorithm_names = ["GA", "GA"] + ["ES", "ES"]
		inputs = (["first"] + ["second"]) * 2
		seeds = [26, 12, 25, 5]

		X, Y, x_steps, upper_xlims, upper_ylims, y_steps = [], [], [], [], [], []
		file_names = []

		for algo_name, input_, seed in zip(algorithm_names, inputs, seeds):

			file_name = f"{algo_name}_{input_}_{seed}.txt"
			file_names.append(file_name)

			best_fitnessess = []
			best_individuals = []
			sums = []
			means = []
			with open("Samples/" + file_name, "r") as f:
				for line in f.readlines():
					line = line.rstrip("\n").split("|")

					best_fitness = line[0].split(":")[1]
					best_individual = line[1].split(":")[1]
					fitness_sum = line[2].split(":")[1]
					fitness_mean = line[3].split(":")[1]

					best_fitnessess.append(best_fitness)
					best_individuals.append(best_individual)
					sums.append(fitness_sum)
					means.append(fitness_mean)


			df = pd.DataFrame()
			df["build_score"] = np.array(best_fitnessess, dtype = np.float64)
			df["best_build"] = np.array(best_individuals, dtype = Build)
			df["generation_fitness_sum"] = np.array(sums, dtype = np.float64)
			df["generation_fitness_mean"] = np.array(means, dtype = np.float64)

			upper_xlim = len(best_fitnessess)

			if upper_xlim <= 1000:
				x_step = 100
			elif upper_xlim <= 5000:
				x_step = 1000
			elif upper_xlim == 5001:
				x_step = 1667
			elif upper_xlim == 5002:
				x_step = 2001
			else:
				x_step = 2000

			if input_ == "first":
				upper_ylim = 30000
				y_step = 5000
			else:
				upper_ylim = 50000
				y_step = 5000
			
			x = np.arange(0, upper_xlim)
			y = df["build_score"].round()

			X.append(x)
			Y.append(y)
			x_steps.append(x_step)
			y_steps.append(y_step)
			upper_xlims.append(upper_xlim)
			upper_ylims.append(upper_ylim)
		
		return X, Y, x_steps, upper_xlims, upper_ylims, y_steps, file_names

	X, Y, x_steps, upper_xlims, upper_ylims, y_steps, file_names = get_information_from_sample()

	fig, axis = plt.subplots(2, 2, figsize = (15, 8))
	fig.subplots_adjust(hspace=0.5)

	for i in range(2):
		for j in range(2):
			axis[i][j].plot(X[2*i+j], Y[2*i+j], color='#0080FF', label = "fitness", linewidth = 2)
			axis[i][j].set(xlim=(0, upper_xlims[2*i+j]), xticks = np.arange(0, upper_xlims[2*i+j]+1, x_steps[2*i+j]),
					ylim=(0, upper_ylims[2*i+j]), yticks = np.arange(0, upper_ylims[2*i+j]+1, y_steps[2*i+j]))
			axis[i][j].set_title(f"Best Fitness Over Generations\nFile name: {file_names[2*i+j]}")
			axis[i][j].set_xlabel(f"Generations ({upper_xlims[2*i+j]})")
			axis[i][j].set_ylabel("Fitness")
			axis[i][j].legend()
	
	plt.show()



if __name__ == '__main__':
	plot_best_samples()
	# plot_all_samples()

