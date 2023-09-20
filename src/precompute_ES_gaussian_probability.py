import numpy as np

number_diferent_atributes = [0, 1, 2, 3, 4, 5]

MUTATION_RATE = 2.5
MEDIA = 5
BOOST = 2.5
# BOOST = 1

probability = lambda delta_x: BOOST * 2 * \
            np.divide(
                1.0, MUTATION_RATE * np.sqrt(2*np.pi)
            ) * \
            np.power(
                np.e, -np.divide(np.power(delta_x - MEDIA, 2), 
                (2 * np.power(MUTATION_RATE, 2)))
            )

# precomputed_probabiltes = [0.04319277321055045, 0.08873666774356447, 0.1553488439865704, 0.23175324220918622, 0.2946161122426587, 0.3191538243211462]
precomputed_probabiltes = [0.10798193302637614, 0.22184166935891117, 0.38837210996642596, 0.5793831055229656, 0.7365402806066468, 0.7978845608028655]
precomputed_probabiltes = list(map(probability, number_diferent_atributes))

print(precomputed_probabiltes)
print(sum(precomputed_probabiltes)) # Should be equal to BOOST, bit is only near to BOOST
