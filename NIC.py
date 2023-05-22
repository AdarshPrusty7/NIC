# Ant Colony Optimisation for Quadratic Assignment Problem

import random
import matplotlib.pyplot as plt

################################################# ANT CLASS ##################################################
class Ant:
    # Constructor
    def __init__(self, id):
        self.id = id
        # Creating a start node and a set of paths is the same thing as randomly assigning a start
        self.current_location = random.randint(0, NUM_OF_LOCATIONS - 1)
        # Tabu list of cities yet to be visited excluding current location
        self.tabu_list = [i for i in range(NUM_OF_LOCATIONS)]
        self.tabu_list.remove(self.current_location)
        self.path = [self.current_location]
        self.cost_of_current_path = 0

    # Move to next location biased by pheromone
    def move(self):
        # Calculate transition probabilities
        transition_probabilities = [transition_probability(
            self, self.current_location, j) for j in self.tabu_list]
        # If sum of transition probabilities is 0, set all transition probabilities to 1.01
        if sum(transition_probabilities) == 0:
            transition_probabilities = [1.01 for i in transition_probabilities]
        # Choose next location based on transition probabilities
        self.current_location = random.choices(
            self.tabu_list, weights=transition_probabilities)[0]
        # Remove next location from tabu list
        self.tabu_list.remove(self.current_location)
        # Add next location to path
        self.path.append(self.current_location)

################################################# READING DATA ##################################################


# Open Uni50a.dat file and establish constants
with open('Uni50a.dat', 'r') as f:
    # Read the first line and assign that to NUM_OF_LOCATIONS
    NUM_OF_LOCATIONS = int(f.readline())
    f.readline()  # Skips the next line
    unformatted_distance_matrix = [next(f) for x in range(NUM_OF_LOCATIONS)]
    DISTANCE_MATRIX = [[int(num) for num in line.split()]
                       for line in unformatted_distance_matrix]
    f.readline()
    unformatted_flow_matrix = [next(f) for x in range(NUM_OF_LOCATIONS)]
    FLOW_MATRIX = [[int(num) for num in line.split()]
                   for line in unformatted_flow_matrix]


################################################# SETTING VARIABLES ##################################################

# Initialise Pheromone Matrix of values between 0 and 1
pheromone_matrix = [[random.random() for i in range(NUM_OF_LOCATIONS)]
                    for j in range(NUM_OF_LOCATIONS)]

# Initialise Heuristic Matrix accounting for 0 values in distance matrix
HEURISTIC_MATRIX = [[1 / DISTANCE_MATRIX[i][j] if DISTANCE_MATRIX[i][j] !=
                     0 else 0 for i in range(NUM_OF_LOCATIONS)] for j in range(NUM_OF_LOCATIONS)]

# Initialise Alpha, Beta, M, EVAPO_RATE
ALPHA = 1
BETA = 5
M = 10
EVAPO_RATE = 0.5

# Number of fitness evaluations
fitness_evals = 0

# Initialise Ants
ANTS = [Ant(i) for i in range(M)]

################################################# FUNCTIONS ##################################################
# Cost function
def cost_function(solution):
    cost = 0
    for i in range(len(solution)):
        for j in range(len(solution)):
            cost += DISTANCE_MATRIX[i][j] * \
                FLOW_MATRIX[solution[i]][solution[j]]
    return cost

# Transition probability function
def transition_probability(ant, ilocation, jlocation):
    # If jlocation has not been visited, calculate transition probability, else return 0

    if jlocation in ant.tabu_list:
        if len(ant.tabu_list) == 1:
            return 1.1  # If there is only one location left, return 1.1 to ensure it is chosen
        denominator = sum([pheromone_matrix[ilocation][k]**ALPHA *
                          HEURISTIC_MATRIX[ilocation][k]**BETA for k in ant.tabu_list])
        # If denominator is 0, set it to 0.01 to avoid division by 0
        denominator = 0.0001 if denominator == 0 else denominator
        return (pheromone_matrix[ilocation][jlocation]**ALPHA * HEURISTIC_MATRIX[ilocation][jlocation]**BETA) / denominator
    else:
        return 0

# Pheromone update function
def pheromone_update():
    # Pheromone update is just pheromones + 1/ fitness of path for each ant
    global pheromone_matrix
    for ant in ANTS:
        for i in range(len(ant.path) - 1):
            # Update pheromone matrix with new pheromones for each ant
            pheromone_matrix[ant.path[i]][ant.path[i + 1]
                                          ] += (1 / cost_function(ant.path))

# Pheromonoe evaporation function
def pheromone_evaporation():
    return [[EVAPO_RATE * pheromone_matrix[i][j] for i in range(NUM_OF_LOCATIONS)] for j in range(NUM_OF_LOCATIONS)]

# Regenerate tabu list and path for all ants
def regenerate_ant():
    for ant in ANTS:
        ant.tabu_list = [i for i in range(NUM_OF_LOCATIONS)]
        # Remove current location from tabu list
        ant.tabu_list.remove(ant.current_location)
        # Initialise path
        ant.path = [ant.current_location]

################################################# MAIN FUNCTION ##################################################
# Main function to run the algorithm
def start(trial_num):
    global fitness_evals
    global pheromone_matrix

    best_solutions = []
    actual_best_fitness = [999999999, 0, []]
    while fitness_evals < 10000:
        # Move ants and generate new paths
        for ant in ANTS:
            for i in range(NUM_OF_LOCATIONS-1):
                ant.move()
        # Update pheromone matrix based on new paths
        pheromone_update()
        # Evaporate pheromones based on evaporation rate
        pheromone_matrix = pheromone_evaporation()
        # Increment fitness evaluations
        fitness_evals += M
        # Print best solution
        p = [cost_function(ant.path) for ant in ANTS]
        bestSol = min(p)
        # add best solution to list
        best_solutions.append(bestSol)

        # Sets overall actual best result and generation
        if bestSol < actual_best_fitness[0]:
            actual_best_fitness[0] = bestSol
            actual_best_fitness[1] = fitness_evals/M
            actual_best_fitness[2] = [
                x.path for x in ANTS if cost_function(x.path) == bestSol]

        bestPath = [x.path for x in ANTS if cost_function(
            x.path) == bestSol][0]
        regenerate_ant()

    # Create a graph to plot the best fitness over time
    # Set title to include actual_best_fitness and the number of generations
    fig, ax = plt.subplots()
    ax.set_title("Best fitness over time\nBest fitness: " +
                 str(actual_best_fitness[0]) + " at generation: " + str(actual_best_fitness[1]-1))
    ax.set_xlabel("Generations")
    ax.set_ylabel("Best fitness")
    # Plot actual best fitness as a scatter point
    ax.scatter(actual_best_fitness[1]-1, actual_best_fitness[0], color="red")
    # Plot the best fitness over fitness evaluations / M with colour blue
    ax.plot([i for i in range(len(best_solutions))],
            best_solutions, color="blue")
    # Save the graph with M and EVAPO_RATE and trial number in the name
    fig.savefig("M" + str(M) + "E" + str(EVAPO_RATE) +
                "T" + str(trial_num) + ".png")
    print("M = " + str(M), "E = " + str(EVAPO_RATE),
          actual_best_fitness[2], actual_best_fitness[0], actual_best_fitness[1])

# Run main function with different values of M and EVAPO_RATE to get different graphs


def main():
    global M
    global EVAPO_RATE
    global ANTS
    global pheromone_matrix
    global fitness_evals

    # Run each one 5 times to produce 5 graphs for each combination of M and EVAPO_RATE
    # M = 100, EVAPO_RATE = 0.9
    M = 100
    EVAPO_RATE = 0.9
    for i in range(5):
        ANTS = [Ant(i) for i in range(M)]
        # Set new pheromone matrix
        pheromone_matrix = [
            [1 for i in range(NUM_OF_LOCATIONS)] for j in range(NUM_OF_LOCATIONS)]
        fitness_evals = 0
        start(i)

    # M = 100, EVAPO_RATE = 0.5
    M = 100
    EVAPO_RATE = 0.5
    for i in range(5):
        ANTS = [Ant(i) for i in range(M)]
        pheromone_matrix = [
            [1 for i in range(NUM_OF_LOCATIONS)] for j in range(NUM_OF_LOCATIONS)]
        fitness_evals = 0
        start(i)

    # M = 10, EVAPO_RATE = 0.9
    M = 10
    EVAPO_RATE = 0.9
    for i in range(5):
        ANTS = [Ant(i) for i in range(M)]
        pheromone_matrix = [
            [1 for i in range(NUM_OF_LOCATIONS)] for j in range(NUM_OF_LOCATIONS)]
        fitness_evals = 0
        start(i)

    # M = 10, EVAPO_RATE = 0.5
    M = 10
    EVAPO_RATE = 0.5
    for i in range(5):
        ANTS = [Ant(i) for i in range(M)]
        pheromone_matrix = [
            [1 for i in range(NUM_OF_LOCATIONS)] for j in range(NUM_OF_LOCATIONS)]
        fitness_evals = 0
        start(i)


main()
