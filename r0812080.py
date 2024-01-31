import numpy as np
import Reporter
import threading

from numba import njit
from collections import Counter

class Parameters:
    def __init__(self):
        # Tuning parameters
        self.llambda = 50
        self.mu = 100
        self.alpha = 0.2
        self.min_alpha = 0.05

        self.lso_variation_share = 1
        self.lso_init_share = 0.5
        self.nn_init_share = 0.2

        self.alpha_fitness_sharing = 1
        self.sigma_share = 0.1

        self.k_crowding = 3
        self.k_elimination = 5
        self.k_selection = 3
        self.k_greedy = 1
        self.k_elitism = 1

        self.swap_its = 5
        self.swap_share = 0.1

        # Methods
        self.elimination = [λ_μ_elimination, μ_elimination, k_t_elimination][0]
        self.selection = [k_t_selection][0]
        self.initialisation = [nn_initialisation, random_initialisation][0]
        self.recombination = [
            partially_mapped_crossover,
            edge_crossover,
            cycle_crossover,
            order_crossover,
            c_s_order_crossover,
        ][0]
        self.mutation = [
            scramble_mutation,
            swap_mutation,
            insert_mutation,
            inversion_mutation,
        ][0]

        self.local_search_operator = [three_opt, two_opt][0]

        # ON/OFF flags
        self.alpha_self_ad = 1
        self.zero_included = 1
        self.threading = 0
        self.elitism = 1
        self.initialize_random = 0
        self.local_search_initialisation = 1
        self.local_search_variation = 1

        self.fitness_sharing_selection = 0
        self.fitness_sharing_elimination = 1
        self.crowding = 0
        self.no_inf = 1
        
        self.island_model = 1


class r0812080:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def optimize(self, filename):
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Common parameters
        params1 = Parameters()
        params2 = Parameters()
        params1.length = params2.length = len(distanceMatrix)
        params1.sigma = params2.sigma = params1.sigma_share * params1.length
        params1.nn_init_size = params2.nn_init_size = int(params1.llambda * params1.nn_init_share)
        params1.random_init_size = params2.random_init_size = params1.llambda - params1.nn_init_size
        params1.swap_size = params2.swap_size = int(params1.llambda//2 * params1.swap_share)
        

        # Island1 parameters
        params1.mutation = [
            insert_mutation,
            inversion_mutation,
            swap_mutation,
            scramble_mutation,
        ]
        params1.recombination =  order_crossover

        # Island2 parameters
        params2.mutation = [
            insert_mutation,
            inversion_mutation,
            swap_mutation,
            scramble_mutation,
        ]
        params2.recombination =  partially_mapped_crossover

        # Initialisation
        island1, island2 = params1.initialisation(distanceMatrix, params1)

        iteration = 1
        while True:
            # Swap islands
            if iteration % params1.swap_its == 0:
                new_individuals_1 = [
                    island2.pop(np.random.randint(0, len(island2)))
                    for _ in range(params1.swap_size)
                ]
                new_individuals_2 = [
                    island1.pop(np.random.randint(0, len(island1)))
                    for _ in range(params1.swap_size)
                ]

                island1 = island1 + new_individuals_1
                island2 = island2 + new_individuals_2

            # Evolve islands
            if params1.threading:
                islands = [island1, island2]
                t1 = threading.Thread(target = evolve_island, args=(0, islands, params1, distanceMatrix))
                t2 = threading.Thread(target = evolve_island, args=(1, islands, params2, distanceMatrix))
                t1.start()
                t2.start()
                t1.join()
                t2.join()

                island1 = islands[0]
                island2 = islands[1]
            else:

                island1 = evolve_island(
                    island1,
                    params1,
                    distanceMatrix,
                )
                island2 = evolve_island(
                    island2,
                    params2,
                    distanceMatrix,
                )

            # Merge islands
            new_population = island1 + island2 

            # Report
            all_fitnesses = [individual.fitness for individual in new_population]
            fitness_argmin = np.argmin(all_fitnesses)
            bestObjective = all_fitnesses[fitness_argmin]
            bestSolution = new_population[fitness_argmin]
            meanObjective = np.mean(all_fitnesses)

            timeLeft = self.reporter.report(
                meanObjective, bestObjective, np.array(bestSolution.permutation)
            )
            if timeLeft < 0:
                break
            iteration += 1
        return 0



# UTILS

# Individual class to represent candidate solutions
class Individual:
    def __init__(
        self,
        permutation=None,
        alpha=0.2,
    ):
        self.permutation = permutation
        self.fitness_share = 0
        self.alpha = alpha

    def update_fitness(self, distanceMatrix):
        self.fitness = fitness(distanceMatrix, self.permutation)

# Fitness function (total distance of permutation)
@njit
def fitness(distanceMatrix, permutation):
    length = len(permutation)
    fitness = 0
    for i in range(length):
        fitness += distanceMatrix[permutation[i]][permutation[(i + 1) % length]]
        if fitness == np.inf:
            break
    return fitness

# Distance function (hamming distance between two permutations)
@njit
def distance(individual1, individual2, zero_included=True):
    if zero_included:
        # Check if two permutations describe the same cycle,
        # for example: [0, 1, 2, 3] and [3, 0, 1, 2] describe the same cycle.
        return np.sum(
            np.roll(individual1, -np.where(individual1 == 0)[0][0])
            != np.roll(individual2, -np.where(individual2 == 0)[0][0])
        )

    return np.sum(individual1 != individual2)


# INITIALISATION


# Random initialisation
def random_initialisation(distanceMatrix, params):
    population = []
    for _ in range(params.llambda):
        permutation = np.random.permutation(params.length)
        if params.local_search_initialisation:
            permutation = params.local_search_operator(distanceMatrix, permutation)
        individual = Individual(permutation)
        individual.update_fitness(distanceMatrix)
        population.append(individual)
    return population


# nn initialisation
def nn_initialisation(
    distanceMatrix,
    params,
):

    adjacent_cities = [[j for j in range(params.length) if distanceMatrix[i][j] != np.inf and distanceMatrix[i][j] != 0
            ] for i in range(params.length)]

    random_part = []
    size = 0
    if params.initialize_random:
        while size < params.random_init_size:
            new_individual = Individual(np.random.permutation(params.length))
            new_individual.update_fitness(distanceMatrix)
            random_part.append(new_individual)
            size += 1
    else:
        choices_start_cities = np.random.permutation(params.length)
        while size < params.random_init_size:
            current_index = 0
            choices = [[]] * params.length
            choices_index = [0] * params.length
            permutation = [choices_start_cities[size]]
            while current_index < params.length:
                if current_index == params.length - 1:
                    if params.no_inf:
                        if distanceMatrix[permutation[-1]][permutation[0]] == np.inf:
                            current_index -= 1
                            permutation.pop()
                            continue
                    break
                if (
                    len(set(adjacent_cities[permutation[current_index]]) - set(permutation))
                    <= choices_index[current_index]
                ):
                    choices_index[current_index] = 0
                    current_index -= 1
                    permutation.pop()
                    continue
                if choices_index[current_index] == 0:
                    choices[current_index] = np.random.permutation(
                        list(set(adjacent_cities[permutation[current_index]]) - set(permutation))
                    )

                next_city = choices[current_index][choices_index[current_index]]
                choices_index[current_index] += 1
                permutation.append(next_city)
                current_index += 1
            permutation = np.array(permutation)
            new_individual = Individual(permutation)
            new_individual.update_fitness(distanceMatrix)
            random_part.append(new_individual)
            size += 1

    for i in range(int(params.lso_init_share * params.random_init_size)):
        individual = random_part[i]
        result = params.local_search_operator(distanceMatrix, individual.permutation)
        if not (result == individual.permutation).all():
            individual.permutation = result
            individual.update_fitness(distanceMatrix)
    size = 0
    nn_part = []
    choices_start_cities = np.random.permutation(params.length)
    while size < params.nn_init_size:
        current_index = 0
        choices_index = [0] * params.length
        permutation = [choices_start_cities[size]]
        while current_index < params.length:
            if current_index == params.length - 1:
                if params.no_inf:
                    if distanceMatrix[permutation[-1]][permutation[0]] == np.inf:
                        current_index -= 1
                        permutation.pop()
                        continue
                break


            if (
                len(set(adjacent_cities[permutation[current_index]]) - set(permutation))
                <= choices_index[current_index]
            ):
                choices_index[current_index] = 0
                current_index -= 1
                permutation.pop()
                continue
            choices = sorted(
                set(adjacent_cities[permutation[current_index]]) - set(permutation),
                key=lambda x: distanceMatrix[permutation[current_index]][x],
            )
            next_city = choices[choices_index[current_index]]
            choices_index[current_index] += 1
            permutation.append(next_city)
            current_index += 1
        permutation = np.array(permutation)
        new_individual = Individual(permutation)
        new_individual.update_fitness(distanceMatrix)
        nn_part.append(new_individual)
        size += 1

    for i in range(int(params.lso_init_share * params.nn_init_size)):
        individual = nn_part[i]
        result = params.local_search_operator(distanceMatrix, individual.permutation)
        if not (result == individual.permutation).all():
            individual.permutation = result
            individual.update_fitness(distanceMatrix)

    island1 = random_part[:params.random_init_size // 2] + nn_part[:params.nn_init_size // 2]
    island2 = random_part[params.random_init_size // 2:] + nn_part[params.nn_init_size // 2:]
    return island1, island2

# SELECTION


# k-tournament selection
def k_t_selection(population, params):
    selected = np.random.choice(population, params.k_selection, replace=False)
    if params.fitness_sharing_selection:
        i = np.argmin([individual.fitness_share for individual in selected])
    else:
        i = np.argmin([individual.fitness for individual in selected])
    return selected[i]


# MUTATION

# Scramble mutation
def scramble_mutation(offspring, params):
    if params.alpha_self_ad:
        alpha = offspring.alpha
    else:
        alpha = params.alpha
    if np.random.rand() < alpha:
        i, j = np.random.choice(len(offspring.permutation), 2, replace=False)
        if i > j:
            i, j = j, i
        np.random.shuffle(offspring.permutation[i : j + 1])

# Swap mutation
def swap_mutation(offspring, params):
    if params.alpha_self_ad:
        alpha = offspring.alpha
    else:
        alpha = params.alpha
    if np.random.rand() < alpha:
        i, j = np.random.choice(len(offspring.permutation), 2, replace=False)
        offspring.permutation[i], offspring.permutation[j] = (
            offspring.permutation[j],
            offspring.permutation[i],
        )

# Insert mutation
def insert_mutation(offspring, params):
    if params.alpha_self_ad:
        alpha = offspring.alpha
    else:
        alpha = params.alpha
    if np.random.rand() < alpha:
        i, j = np.random.choice(len(offspring.permutation), 2, replace=False)
        if i > j:
            i, j = j, i
        offspring.permutation[i : j + 1] = np.roll(offspring.permutation[i : j + 1], 1)

# Inversion mutation
def inversion_mutation(offspring, params):
    if params.alpha_self_ad:
        alpha = offspring.alpha
    else:
        alpha = params.alpha
    if np.random.rand() < alpha:
        i, j = np.random.choice(len(offspring.permutation), 2, replace=False)
        if i > j:
            i, j = j, i
        offspring.permutation[i : j + 1] = np.flip(offspring.permutation[i : j + 1])


# RECOMBINATION


# Edge crossover (EX)
def edge_crossover(parent1, parent2, params):
    permutation1 = parent1.permutation
    permutation2 = parent2.permutation
    size = len(permutation1)

    edge_table = {}
    for key in range(size):
        edge_list = []

        index = int(np.where(permutation1 == key)[0])
        edge_list.append(permutation1[(index - 1) % size])
        edge_list.append(permutation1[(index + 1) % size])

        index = int(np.where(permutation2 == key)[0])
        edge_list.append(permutation2[(index - 1) % size])
        edge_list.append(permutation2[(index + 1) % size])
        edge_table[key] = edge_list

    offspring = np.full(size, -1)
    current_elem = np.random.randint(size) 
    offspring[0] = current_elem
    go_forward = True
    forward_index = 0
    backward_index = 0
    iter_count = 0

    while iter_count < size - 1:
        remove_ref(edge_table, current_elem)
        if go_forward and len(edge_table[current_elem]) == 0:
            go_forward = False
            current_elem = offspring[backward_index]

        elif (not go_forward) and len(edge_table[current_elem]) == 0:
            go_forward = True
            forward_index += 1
            iter_count += 1
            current_elem = np.random.choice(
                list(set(range(size)).difference(set(offspring)))
            )
            offspring[forward_index] = current_elem

        else:
            current_elem = find_next_element(edge_table, current_elem)
            if go_forward:
                forward_index += 1
                offspring[forward_index] = current_elem
            else:
                backward_index = (backward_index - 1) % size
                offspring[backward_index] = current_elem
            iter_count += 1

    child = Individual(offspring)
    if params.alpha_self_ad:
        beta = 2 * np.random.random() - 0.5
        alpha = parent1.alpha + beta * (parent2.alpha - parent1.alpha)
        alpha = max(params.min_alpha, alpha)
        child.alpha = alpha
    return [child]

def find_next_element(edge_table, elem):
    if len(edge_table[elem]) == 1:
        return edge_table[elem][0]
    counts = Counter(edge_table[elem]).most_common()
    max_count = counts[0][1]
    opt_next_elem = [value for (value, count) in counts if count == max_count]
    if len(opt_next_elem) == 1:
        return opt_next_elem[0]
    list_lengths = [len(set(edge_table[x])) for x in opt_next_elem]
    min_list_length = min(list_lengths)
    opt_next_elem = [
        x for x in opt_next_elem if len(set(edge_table[x])) == min_list_length
    ]
    return np.random.choice(opt_next_elem)

def remove_ref(edge_table, elem):
    for key in edge_table:
        edge_table[key] = [x for x in edge_table[key] if x != elem]



# Partially Mapped Crossover (PMX)
# @njit
def partially_mapped_crossover(parent1, parent2, params):
    permutation1 = parent1.permutation
    permutation2 = parent2.permutation
    length = len(permutation1)
    i, j = np.random.choice(length, 2, replace=False)
    if i > j:
        i, j = j, i
    child = np.full(length, -1)
    child[i : j + 1] = permutation1[i : j + 1]
    for k in range(i, j + 1):
        if permutation2[k] not in child:
            l = k
            while child[l] != -1:
                l = np.where(permutation2 == child[l])[0][0]
            child[l] = permutation2[k]
    for k in range(length):
        if child[k] == -1:
            child[k] = permutation2[k]

    child = Individual(child)
    if params.alpha_self_ad:
        beta = 2 * np.random.random() - 0.5
        alpha = parent1.alpha + beta * (parent2.alpha - parent1.alpha)
        alpha = max(params.min_alpha, alpha)
        child.alpha = alpha
    return [child]

# Order Crossover (OX)
# @njit
def order_crossover(parent1, parent2, params, i=None, j=None):
    if i == None and j == None:
        length = len(parent1.permutation)
        i, j = np.random.choice(length, 2, replace=False)
        permutation1 = parent1.permutation
        permutation2 = parent2.permutation

    else:
        permutation1 = parent1
        permutation2 = parent2
        length = len(permutation1)
    if i > j:
        i, j = j, i
    child = np.full(length, -1)
    child[i : j + 1] = permutation1[i : j + 1]
    l = (j + 1) % length
    k = (j + 1) % length
    while child[l] == -1:
        while permutation2[k] in child:
            k = (k + 1) % length
        child[l] = permutation2[k]
        l = (l + 1) % length

    child = Individual(child)
    if params.alpha_self_ad:
        beta = 2 * np.random.random() - 0.5
        alpha = parent1.alpha + beta * (parent2.alpha - parent1.alpha)
        alpha = max(params.min_alpha, alpha)
        child.alpha = alpha
    return [child]

# Cycle Crossover (CX)
# @njit
def cycle_crossover(parent1, parent2, params):
    permutation1 = parent1.permutation
    permutation2 = parent2.permutation

    lookup_dict = {city: i for i, city in enumerate(permutation1)}

    cycles = [-1] * len(permutation1)
    cycle_start = (i for i, value in enumerate(cycles) if value == -1)

    for cycle_number, i in enumerate(cycle_start, 1):
        while cycles[i] == -1:
            cycles[i] = cycle_number
            i = lookup_dict[permutation2[i]]

    child1 = np.array(
        [permutation1[i] if n % 2 else permutation2[i] for i, n in enumerate(cycles)]
    )
    child2 = np.array(
        [permutation2[i] if n % 2 else permutation1[i] for i, n in enumerate(cycles)]
    )

    child1 = Individual(child1)
    child2 = Individual(child2)
    if params.alpha_self_ad:
        beta = 2 * np.random.random() - 0.5

        alpha = parent1.alpha + beta * (parent2.alpha - parent1.alpha)
        alpha = max(params.min_alpha, alpha)
        child1.alpha = alpha

        alpha = parent2.alpha + beta * (parent1.alpha - parent2.alpha)
        alpha = max(params.min_alpha, alpha)
        child2.alpha = alpha

    return [child1, child2]

# Complete Subtour Order Crossover (CSOX)
# @njit
def c_s_order_crossover(parent1, parent2, params):
    permutation1 = parent1.permutation
    permutation2 = parent2.permutation
    [i, j] = np.random.choice(len(permutation1) -2, 2, replace=False)
    i +=1
    j +=1
    if i > j:
        i, j = j, i
    children = []
    for k in range(3):
        if k == 0:
            position1 = i
            position2 = j
        elif k == 1:
            position1 = 0
            position2 = i - 1
        elif k == 2:
            position1 = j + 1
            position2 = len(permutation1) - 1
        children.append(order_crossover(permutation1, permutation2, params, position1, position2)[0])
        children.append(order_crossover(permutation2, permutation1, params, position1, position2)[0])

    
    return children


# ELIMINATION


# (lambda + mu)-elimination
def λ_μ_elimination(merged, params, size):
    if params.fitness_sharing_elimination:
        return fitness_sharing_el(merged, params, size)
    else:
        sorted_population = sorted(merged, key=lambda individual: individual.fitness)
        return sorted_population[:size]


# (lambda, mu)-elimination
def μ_elimination(offspring, _, size):
    sorted_offspring = sorted(
        offspring, key=lambda individual: individual.fitness
    )
    return sorted_offspring[:size]

# k-tournament elimination
def k_t_elimination(merged, params, size):
    if params.fitness_sharing_elimination:
        new_population = fitness_sharing_el(merged, params, size, mu_lambda=False)
 
    else:
        new_population = []
        for _ in range(size):
            selected = np.random.choice(merged, params.k_elimination, replace=False)
            choice = np.argmin([individual.fitness for individual in selected])
            new_population.append(selected[choice])

    return new_population


# DIVERSITY PROMOTION MECHANISMS


# Crowding (De Jong’s original crowding algorithm)
def crowding(population, individual, params):
    selected = np.random.choice(population, params.k_crowding, replace=False)
    distances = [distance(individual.permutation, x.permutation) for x in selected]
    closest_individual = np.argmin(distances)
    return np.delete(population, closest_individual)

# Island model
def evolve_island(
    population,
    params,
    distanceMatrix,
):
    offspring = []
    length_offspring = 0
    size = len(population)

    if params.fitness_sharing_selection:
        fitness_sharing_sel(population, params)

    while length_offspring < params.mu//2:
        parent1 = params.selection(population, params)
        parent2 = params.selection(population, params)
        children = params.recombination(parent1, parent2, params)
        for child in children:
            mutation = params.mutation[np.random.randint(len(params.mutation))]
            mutation(child, params)
            child.update_fitness(distanceMatrix)
            length_offspring += 1
            offspring.append(child)

    if params.elitism:
        best_individuals = []
        for i in range(params.k_elitism):
            best_index = np.argmin([individual.fitness for individual in population])
        best_individuals.append(population.pop(best_index))

    
    for individual in population:
        mutation = params.mutation[np.random.randint(len(params.mutation))]
        mutation(individual, params)
        individual.update_fitness(distanceMatrix)
    if params.elitism:
        for best_individual in best_individuals:
            population.insert(0, best_individual)

    if params.local_search_variation:
        for i in range(int(params.lso_variation_share * len(population))):            
            individual = population[i]
            result = params.local_search_operator(distanceMatrix, individual.permutation)
            if not (result == individual.permutation).all():
                individual.permutation = result
                individual.update_fitness(distanceMatrix)
    
    merged = population + offspring
    new_population = params.elimination(merged, params, size)
    return new_population

# Fitness sharing

# Fitness sharing elimination
def fitness_sharing_el(population, params, size, mu_lambda=True): 
    new_population = []
    new_individual = None
    for individual in population:
        individual.fitness_share = individual.fitness
    for _ in range(size):
        if new_individual != None:
            for individual in population:
                if individual.fitness_share != np.inf:
                    dist = distance(new_individual.permutation, individual.permutation)
                    if dist <= params.sigma:
                        individual.fitness_share += individual.fitness * (
                            1 - (dist / params.sigma) ** params.alpha_fitness_sharing
                        )
        if mu_lambda:
            new_individual_index = np.argmin(
                [individual.fitness_share for individual in population]
            )

        else:
            selected = np.random.choice(size, params.k_elimination, replace=False)
            new_individual_index = np.argmin([population[i].fitness_share for i in selected])

        new_individual = population.pop(new_individual_index)
        new_population.append(new_individual)
    return new_population

# Fitness sharing selection
def fitness_sharing_sel(population, params):
    size = len(population)
    distances_list = [0] * (size**2)
    for i in range(size - 1):
        for j in range(i + 1, size):
            distances_list[i * size + j] = distance(
                population[i].permutation, population[j].permutation
            )
    for i, individual in enumerate(population):
        sum = 1
        for j in range(i):
            distance_ = distances_list[j * size + i]
            if distance_ <= params.sigma:
                sum += 1 - (distance_ / params.sigma) ** params.alpha_fitness_sharing
        for j in range(i + 1, size):
            distance_ = distances_list[i * size + j]
            if distance_ <= params.sigma:
                sum += 1 - (distance_ / params.sigma) ** params.alpha_fitness_sharing

        individual.fitness_share = individual.fitness * sum

# LOCAL SEARCH OPERATORS
        
# 2-opt
@njit
def two_opt(distanceMatrix, permutation, max_changes=100):
    length = len(permutation)
    breakk = False
    nb_of_changes = 0
    best_fitness_change = 0
    for start_edge_1 in range(length - 2):
        old_edge1 = distanceMatrix[permutation[start_edge_1]][
            permutation[start_edge_1 + 1]]
        for start_edge_2 in range(start_edge_1 + 2, length):
            old_edge2 = distanceMatrix[permutation[start_edge_2]][
                permutation[start_edge_2 + 1]]
            new_edge1 = distanceMatrix[permutation[start_edge_1]][
            permutation[start_edge_2]
            ]
            new_edge2 = distanceMatrix[permutation[start_edge_1 + 1]][
            permutation[start_edge_2 + 1]
            ]
            if new_edge1 == np.inf or new_edge2 == np.inf:
                continue
            new_fitness_segment = fitness(distanceMatrix,  np.flip(permutation[start_edge_1 + 1: start_edge_2 + 1])) 
            if new_fitness_segment == np.inf:
                continue
            if old_edge1 == np.inf or old_edge2 == np.inf: 
                best_start_edge_1, best_start_edge_2 = (
                    start_edge_1,
                    start_edge_2
                )
                breakk = True
                break
            old_fitness_segment = fitness(distanceMatrix, permutation[start_edge_1 + 1: start_edge_2 + 1])
            if old_fitness_segment == np.inf:
                best_start_edge_1, best_start_edge_2 = (
                    start_edge_1,
                    start_edge_2
                )
                breakk = True
                break
            fitness_change_1 = old_edge1 - new_edge1
            fitness_change_2 = old_edge2 - new_edge2
            fitness_change_3 = old_fitness_segment - new_fitness_segment
            fitness_change = fitness_change_1 + fitness_change_2 + fitness_change_3

            if fitness_change > best_fitness_change:
                best_fitness_change = fitness_change
                best_start_edge_1, best_start_edge_2 = (
                    start_edge_1,
                    start_edge_2
                )
                nb_of_changes += 1
                if nb_of_changes >= max_changes:
                    breakk = True
                    break
        if breakk:
            break
    if best_fitness_change == 0:
        return permutation
    else:
        return np.concatenate(
            (
                permutation[: best_start_edge_1 + 1],
                np.flip(permutation[best_start_edge_1 + 1 : best_start_edge_2 + 1]),
                permutation[best_start_edge_2 + 1 :],
            )
        )

# 3-opt
@njit
def three_opt(distanceMatrix, permutation, max_changes=100):
    length = len(permutation)
    breakk = False
    nb_of_changes = 0
    best_fitness_change = 0
    for start_edge_1 in range(length - 2):
        old_edge1 = distanceMatrix[permutation[start_edge_1]][
                permutation[start_edge_1 + 1]]
        best_fitness_change_1 = -np.inf
        for start_edge_2 in range(start_edge_1 + 1, length - 1):
            new_edge1 = distanceMatrix[permutation[start_edge_1]][
                permutation[start_edge_2 + 1]
            ]
            fitness_change_1 = old_edge1 - new_edge1
            if new_edge1 == np.inf or fitness_change_1 < best_fitness_change_1:
                continue
            else:
                best_fitness_change_1 = fitness_change_1
            for start_edge_3 in range(start_edge_2 + 1, length):
                new_edge2 = distanceMatrix[permutation[start_edge_3]][
                    permutation[start_edge_1 + 1]
                ]
                new_edge3 = distanceMatrix[permutation[start_edge_2]][
                    permutation[(start_edge_3 + 1) % length]
                ]
                if new_edge2 == np.inf or new_edge3 == np.inf:
                    continue

                old_edge2 = distanceMatrix[permutation[start_edge_2]][
                    permutation[start_edge_2 + 1]
                ]
                old_edge3 = distanceMatrix[permutation[start_edge_3]][
                    permutation[(start_edge_3 + 1) % length]
                ]
                if old_edge1 == np.inf or old_edge2 == np.inf or old_edge3 == np.inf:
                    best_start_edge_1, best_start_edge_2, best_start_edge_3 = (
                        start_edge_1,
                        start_edge_2,
                        start_edge_3,
                    )
                    breakk = True
                    break

                fitness_change_2 = old_edge2 - new_edge2
                fitness_change_3 = old_edge3 - new_edge3
                fitness_change = fitness_change_1 + fitness_change_2 + fitness_change_3

                if fitness_change > best_fitness_change:
                    best_fitness_change = fitness_change
                    best_start_edge_1, best_start_edge_2, best_start_edge_3 = (
                        start_edge_1,
                        start_edge_2,
                        start_edge_3,
                    )
                    nb_of_changes += 1
                    if nb_of_changes >= max_changes:
                        breakk = True
                        break
            if breakk:
                break
        if breakk:
            break
    if best_fitness_change == 0:
        return permutation

    else:
        return np.concatenate(
            (
                permutation[: best_start_edge_1 + 1],
                permutation[best_start_edge_2 + 1 : best_start_edge_3 + 1],
                permutation[best_start_edge_1 + 1 : best_start_edge_2 + 1],
                permutation[best_start_edge_3 + 1 :],
            )
        )
