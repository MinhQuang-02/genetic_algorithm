import random
import numpy as np
import math
import copy
import time

# Đọc tập tin TSP và tính ma trận trọng số
def read_tsp_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    index = 0
    while not lines[index].startswith('NODE_COORD_SECTION'):
        index += 1
    index += 1
    coordinates = []
    while not lines[index].startswith('EOF'):
        parts = lines[index].split()
        coordinates.append((int(parts[0]), float(parts[1]), float(parts[2])))
        index += 1
    return coordinates

def euclidean_distance(coord1, coord2):
    x1, y1 = coord1[1], coord1[2]
    x2, y2 = coord2[1], coord2[2]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_weight_matrix(coordinates):
    num_points = len(coordinates)
    weight_matrix = [[0] * num_points for _ in range(num_points)]
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                weight_matrix[i][j] = euclidean_distance(coordinates[i], coordinates[j])
    return weight_matrix

# Hàm tính độ thích nghi (fitness)
def calculate_fitness(individual, weight_matrix):
    fitness = 0
    for i in range(len(individual)):
        fitness += weight_matrix[individual[i - 1]][individual[i]]
    return int(fitness)

# Hàm tạo cá thể
def create_individual(num_points, num_created, gts_rate=100):
    if num_created % gts_rate == 0:
        return create_individual_gts(num_points)
    else:
        return create_individual_random(num_points)

# Hàm tạo cá thể ngẫu nhiên
def create_individual_random(num_points):
    individual = random.sample(range(num_points), num_points)
    return individual

# Hàm tạo cá thể tham lam
def create_individual_gts(num_points):
    # Chọn ngẫu nhiên một điểm bắt đầu
    start_city = random.randint(0, num_points - 1)
    # Tạo một danh sách các thành phố chưa được ghé thăm
    unvisited_cities = set(range(num_points))
    unvisited_cities.remove(start_city)
    # Bắt đầu từ điểm bắt đầu và tạo một cá thể bằng cách thêm các thành phố vào theo thứ tự gần nhất
    individual = [start_city]
    current_city = start_city
    while unvisited_cities:
        nearest_city = min(unvisited_cities, key=lambda x: weight_matrix[current_city][x])
        individual.append(nearest_city)
        unvisited_cities.remove(nearest_city)
        current_city = nearest_city
    return individual

# Hàm lai ghép (crossover) PMX
def crossover(parent1, parent2):
    child = [-1] * len(parent1)
    start, end = sorted([random.randint(0, len(parent1) - 1) for _ in range(2)])
    for i in range(start, end + 1):
        child[i] = parent1[i]
    remaining = [item for item in parent2 if item not in child]
    j = 0
    for i in range(len(child)):
        if child[i] == -1:
            child[i] = remaining[j]
            j += 1
    return child

# Hàm đột biến (mutation) reversed
def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        # Chọn ngẫu nhiên hai vị trí trên chuỗi gen
        start, end = sorted(random.sample(range(len(individual)), 2))
        # Đảo ngược phần của cá thể từ start đến end
        individual[start:end+1] = reversed(individual[start:end+1])
    return individual

# Hàm lựa chọn dựa trên thứ hạng (rank-based selection)
def select_rank_based(population, weight_matrix, num_parents):
    # Tính fitness và xếp hạng các cá thể
    fitness_values = [calculate_fitness(individual, weight_matrix) for individual in population]
    ranked_individuals = sorted(zip(population, fitness_values), key=lambda x: x[1])

    # Tính xác suất lựa chọn dựa trên thứ hạng
    rank_probabilities = [i/sum(range(1, len(population)+1)) for i in range(len(population), 0, -1)]

    # Lựa chọn cá thể cho quần thể mới bằng cách lấy mẫu ngẫu nhiên theo xác suất
    selected_parents = random.choices(population, weights=rank_probabilities, k=num_parents)
    return selected_parents

# Hàm thực thi thuật toán GA cho bài toán TSP
def genetic_algorithm_tsp(weight_matrix, population_size=300, generations=3000, crossover_rate=0.8, mutation_rate=0.03):
    num_points = len(weight_matrix)
    # Khởi tạo quần thể ban đầu
    population = [create_individual(num_points, 0)] * population_size

    # Vòng lặp qua các thế hệ
    for gen in range(1, generations + 1):
        new_population = []
        # Lai ghép và đột biến
        for _ in range(population_size // 2):
            parents = select_rank_based(population, weight_matrix, 2)
            if random.random() < crossover_rate:
                offspring1 = crossover(parents[0], parents[1])
                offspring2 = crossover(parents[1], parents[0])
            else:
                offspring1, offspring2 = parents[0][:], parents[1][:]
            new_population.extend([mutate(offspring1, mutation_rate), mutate(offspring2, mutation_rate)])

        # Chọn các cá thể tốt nhất từ cả quần thể ban đầu và quần thể mới tạo ra
        combined_population = population + new_population
        population = [individual for individual, _ in sorted(zip(combined_population, [calculate_fitness(individual, weight_matrix) for individual in combined_population]), key=lambda x: x[1])[:population_size]]

        # In kết quả tốt nhất sau mỗi 100 lần lặp
        if gen % 100 == 0:
            best_individual = min(population, key=lambda x: calculate_fitness(x, weight_matrix))
            best_fitness = calculate_fitness(best_individual, weight_matrix)
            print(f"Generation {gen}: Best fitness = {best_fitness}")

    # Trả về cá thể tốt nhất và giá trị fitness của nó sau khi kết thúc vòng lặp
    best_individual = min(population, key=lambda x: calculate_fitness(x, weight_matrix))
    best_fitness = calculate_fitness(best_individual + [best_individual[0]], weight_matrix)
    return best_individual, best_fitness

def write_solution_to_file(file_name, best_distance, best_tour):
    with open(file_name, 'w') as file:
        file.write(f"{best_distance}\n")
        best_tour_with_initial = [city + 1 for city in best_tour] + [best_tour[0]]
        file.write(" ".join(map(str, best_tour_with_initial)) + "\n")

# Chạy chương trình chính
if __name__ == "__main__":
    
    filename = "pr226.tsp"  # Thay thế bằng tên tập tin TSP của bạn
    coordinates = read_tsp_file(filename)
    weight_matrix = calculate_weight_matrix(coordinates)

    start_time = time.perf_counter()
    best_route, best_fitness = genetic_algorithm_tsp(weight_matrix)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print("Time taken to find the solution: ", elapsed_time, "seconds")
    print("Best route:", [city + 1 for city in best_route])
    print("Best fitness:", best_fitness)

    output_file_path = "pr226_1out.txt"  # Thay thế bằng tên tập tin đầu ra của bạn
    write_solution_to_file(output_file_path, best_fitness, best_route)
