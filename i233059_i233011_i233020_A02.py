import random
import math
import copy
import heapq

# =============================================================
# DATA STRUCTURES
# =============================================================

class Surgery:
    def __init__(self, surgery_id, name, duration, urgency, surgeon):
        self.surgery_id = surgery_id
        self.name = name
        self.duration = duration
        self.urgency = urgency
        self.surgeon = surgeon


# =============================================================
# VALIDATION FUNCTION (COMMON)
# =============================================================

def is_valid_schedule(schedule, surgeries_dict, rooms, total_slots):
    room_slots = {room: set() for room in rooms}
    surgeon_slots = {}

    for sid, (room, start) in schedule.items():
        s = surgeries_dict[sid]
        end = start + s.duration

        if end > total_slots:
            return False

        if s.surgeon not in surgeon_slots:
            surgeon_slots[s.surgeon] = set()

        for t in range(start, end):
            if t in room_slots[room] or t in surgeon_slots[s.surgeon]:
                return False
            room_slots[room].add(t)
            surgeon_slots[s.surgeon].add(t)

    return True


# =============================================================
# COST FUNCTION (g(n))
# =============================================================

def calculate_cost(schedule, surgeries_dict):
    cost = 0
    for sid, (room, slot) in schedule.items():
        s = surgeries_dict[sid]

        if s.urgency == 2:
            cost += max(0, slot - 4) * 2
        elif s.urgency == 3:
            cost += max(0, slot - 2) * 4

    return cost


# =============================================================
# HEURISTIC (h(n))
# =============================================================

def heuristic(schedule, surgeries_dict):
    h = 0
    for sid, (room, slot) in schedule.items():
        s = surgeries_dict[sid]

        if s.urgency == 2:
            h += max(0, slot - 4)
        elif s.urgency == 3:
            h += max(0, slot - 2)

    return h


# =============================================================
# A* SEARCH
# =============================================================

class State:
    def __init__(self, schedule, g):
        self.schedule = schedule
        self.g = g
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f


def generate_neighbors(schedule, surgeries_dict, rooms, total_slots):
    neighbors = []

    for sid in schedule:
        for room in rooms:
            for delay in [1, 2]:
                new_schedule = copy.deepcopy(schedule)
                old_room, old_slot = new_schedule[sid]
                new_schedule[sid] = (room, old_slot + delay)

                if is_valid_schedule(new_schedule, surgeries_dict, rooms, total_slots):
                    neighbors.append(new_schedule)

    return neighbors


def astar(initial_schedule, surgeries_dict, rooms, total_slots):
    open_list = []
    visited = set()

    start = State(initial_schedule, 0)
    start.h = heuristic(initial_schedule, surgeries_dict)
    start.f = start.g + start.h

    heapq.heappush(open_list, start)

    while open_list:
        current = heapq.heappop(open_list)

        key = tuple(sorted(current.schedule.items()))
        if key in visited:
            continue
        visited.add(key)

        if is_valid_schedule(current.schedule, surgeries_dict, rooms, total_slots):
            return current.schedule

        for neighbor in generate_neighbors(current.schedule, surgeries_dict, rooms, total_slots):
            g = calculate_cost(neighbor, surgeries_dict)
            new_state = State(neighbor, g)
            new_state.h = heuristic(neighbor, surgeries_dict)
            new_state.f = new_state.g + new_state.h

            heapq.heappush(open_list, new_state)

    return None


# =============================================================
# SIMULATED ANNEALING
# =============================================================

def random_neighbor(schedule, surgeries_dict, rooms, total_slots):
    new_schedule = copy.deepcopy(schedule)
    sid = random.choice(list(new_schedule.keys()))
    new_schedule[sid] = (
        random.choice(rooms),
        random.randint(0, total_slots - 1)
    )
    return new_schedule


def simulated_annealing(initial_schedule, surgeries_dict, rooms, total_slots):
    current = copy.deepcopy(initial_schedule)
    best = copy.deepcopy(current)

    T = 100
    cooling = 0.95

    while T > 1:
        neighbor = random_neighbor(current, surgeries_dict, rooms, total_slots)

        if not is_valid_schedule(neighbor, surgeries_dict, rooms, total_slots):
            continue

        delta = calculate_cost(neighbor, surgeries_dict) - calculate_cost(current, surgeries_dict)

        if delta < 0 or random.random() < math.exp(-delta / T):
            current = neighbor

        if calculate_cost(current, surgeries_dict) < calculate_cost(best, surgeries_dict):
            best = current

        T *= cooling

    return best


# =============================================================
# GENETIC ALGORITHM
# =============================================================

def generate_random_schedule(surgeries_dict, rooms, total_slots):
    return {
        sid: (random.choice(rooms), random.randint(0, total_slots - 1))
        for sid in surgeries_dict
    }


def fitness(schedule, surgeries_dict, rooms, total_slots):
    if not is_valid_schedule(schedule, surgeries_dict, rooms, total_slots):
        return float('inf')
    return calculate_cost(schedule, surgeries_dict)


def crossover(p1, p2):
    child = {}
    for sid in p1:
        child[sid] = p1[sid] if random.random() < 0.5 else p2[sid]
    return child


def mutate(schedule, rooms, total_slots, rate=0.1):
    new_schedule = copy.deepcopy(schedule)
    for sid in new_schedule:
        if random.random() < rate:
            new_schedule[sid] = (
                random.choice(rooms),
                random.randint(0, total_slots - 1)
            )
    return new_schedule


def genetic_algorithm(surgeries_dict, rooms, total_slots):
    population = [
        generate_random_schedule(surgeries_dict, rooms, total_slots)
        for _ in range(20)
    ]

    for _ in range(50):
        population.sort(key=lambda s: fitness(s, surgeries_dict, rooms, total_slots))
        new_pop = population[:5]

        while len(new_pop) < 20:
            p1 = random.choice(population[:10])
            p2 = random.choice(population[:10])

            child = crossover(p1, p2)
            child = mutate(child, rooms, total_slots)

            new_pop.append(child)

        population = new_pop

    return min(population, key=lambda s: fitness(s, surgeries_dict, rooms, total_slots))


# =============================================================
# TEST RUN
# =============================================================

if __name__ == "__main__":
    rooms = ["Room 1", "Room 2"]
    total_slots = 10

    surgeries = {
        "S1": Surgery("S1", "A", 2, 1, "Dr A"),
        "S2": Surgery("S2", "B", 3, 2, "Dr B"),
        "S3": Surgery("S3", "C", 2, 3, "Dr A"),
    }

    schedule = {
        "S1": ("Room 1", 0),
        "S2": ("Room 2", 1),
        "S3": ("Room 1", 3),
    }

    print("A* Result:")
    print(astar(schedule, surgeries, rooms, total_slots))

    print("\nSimulated Annealing:")
    print(simulated_annealing(schedule, surgeries, rooms, total_slots))

    print("\nGenetic Algorithm:")
    print(genetic_algorithm(surgeries, rooms, total_slots))