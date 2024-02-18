import random
import matplotlib.pyplot as plt

# Parameters
TARGET_STRING = "SIMEN_VANGBERG*554916"
POPULATION_SIZE = 1000
MUTATION_RATE = 0.03
CROSSOVER_PROBABILITY = 0.5
MAX_GENERATIONS = 1000

# Generate initial population
def generate_individual():
    characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_*'
    return ''.join(random.choice(characters) for _ in range(len(TARGET_STRING)))

def generate_population():
    return [generate_individual() for _ in range(POPULATION_SIZE)]

# Fitness function
def calculate_fitness(individual):
    score = sum(1 for i in range(len(TARGET_STRING)) if individual[i] == TARGET_STRING[i])
    return score / len(TARGET_STRING)

# Selection (Tournament selection)
def select_individual(population):
    tournament_size = 5
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=calculate_fitness)

# Crossover (Single point crossover)
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_PROBABILITY:
        crossover_point = random.randint(0, len(TARGET_STRING))
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    else:
        return parent1

# Mutation
def mutate(individual):
    mutated_individual = list(individual)
    for i in range(len(mutated_individual)):
        if random.random() < MUTATION_RATE:
            characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_*'
            mutated_individual[i] = random.choice(characters)
    return ''.join(mutated_individual)

# Genetic algorithm
def genetic_algorithm():
    population = generate_population()
    generations = 0
    best_fitnesses = []
    best_individual = None
    while generations < MAX_GENERATIONS:
        fitness_scores = [calculate_fitness(individual) for individual in population]
        best_fitness = max(fitness_scores)
        best_fitnesses.append(best_fitness)
        best_fit_index = fitness_scores.index(best_fitness)
        best_fit_individual = population[best_fit_index]
        if best_individual is None or calculate_fitness(best_individual) < best_fitness:
            best_individual = best_fit_individual
        print(f"Generation {generations}: Best fitness = {best_fitness}, Best individual = {best_individual}")
        if best_fit_individual == TARGET_STRING:
            print(f"Best fit individual found in {generations} generations: {best_fit_individual}")
            return best_fitnesses
        selected_population = [select_individual(population) for _ in range(POPULATION_SIZE)]
        population = [crossover(selected_population[i], selected_population[i+1]) for i in range(0, POPULATION_SIZE, 2)]
        population = [mutate(individual) for individual in population]
        generations += 1
    print(f"Maximum generations reached. Best fit individual not found.")
    return best_fitnesses

# Run the genetic algorithm
best_fitnesses = genetic_algorithm()

# Plot the best fitness values
plt.plot(best_fitnesses)
plt.title('Best Fitness Values Over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.grid(True)
plt.show()
