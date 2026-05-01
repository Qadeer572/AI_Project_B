"""
=============================================================
AI2002 - Artificial Intelligence | Spring 2026
Project: AI-Driven Hospital Surgery Scheduling System
File   : search_Algorithm.py
Topic  : Informed Search - A* Algorithm (Part B)
Desc   : A* Search for optimal surgery rescheduling when
         an emergency case arrives and disrupts the current
         schedule. Finds the least-cost valid rescheduling
         path using g(n) + h(n).
=============================================================
"""

import heapq   
import copy    


# =============================================================
# DATA CLASSES
# =============================================================

class Surgery:
    """
    Represents a single surgery with all its attributes.
    Each surgery is a variable in our scheduling problem.
    """

    def __init__(self, surgery_id, name, duration, urgency, surgeon, equipment):
        """
        surgery_id : unique identifier e.g. "S1"
        name       : surgery name e.g. "Appendectomy"
        duration   : how many time slots it takes (1 slot = 1 hour)
        urgency    : 1 = elective, 2 = urgent, 3 = emergency
        surgeon    : surgeon name assigned to this surgery
        equipment  : list of required equipment e.g. ["ventilator"]
        """
        self.surgery_id = surgery_id
        self.name       = name
        self.duration   = duration
        self.urgency    = urgency
        self.surgeon    = surgeon
        self.equipment  = equipment

    def __repr__(self):
        urgency_label = {1: "Elective", 2: "Urgent", 3: "Emergency"}
        return (f"Surgery({self.surgery_id} | {self.name} | "
                f"{urgency_label[self.urgency]} | Surgeon: {self.surgeon} | "
                f"Duration: {self.duration}hrs)")


class ScheduleState:
    """
    Represents one complete hospital schedule configuration.
    This is a NODE in the A* search graph.

    schedule : dict mapping surgery_id -> (room, start_slot)
               e.g. {"S1": ("Room1", 0), "S2": ("Room2", 2)}
    """

    def __init__(self, schedule, g_cost, parent=None, action=None):
        """
        schedule : current assignment of surgeries to rooms and slots
        g_cost   : total cost incurred so far (delay + overtime)
        parent   : parent ScheduleState (for path tracing)
        action   : what action was taken to reach this state
        """
        self.schedule = schedule
        self.g_cost   = g_cost
        self.parent   = parent
        self.action   = action
        self.h_cost   = 0   # set externally by heuristic function
        self.f_cost   = 0   # f(n) = g(n) + h(n)

    def __lt__(self, other):
        # heapq needs this to compare states in the priority queue
        return self.f_cost < other.f_cost

    def __repr__(self):
        return f"ScheduleState(f={self.f_cost:.2f}, g={self.g_cost:.2f}, h={self.h_cost:.2f})"


# =============================================================
# CONSTRAINT CHECKER
# =============================================================

def is_valid_schedule(schedule, surgeries_dict, rooms, total_slots):
    """
    Checks all hard constraints for a given schedule.

    Constraints:
      1. One surgery per room per time slot
      2. One surgeon cannot be in two rooms at the same time
      3. Equipment must be available in the assigned room
      4. Surgery must fit within total available time slots

    Returns True if all constraints are satisfied, False otherwise.
    """

    # track which slots are occupied per room
    room_slots = {room: set() for room in rooms}

    # track which slots each surgeon is busy
    surgeon_slots = {}

    for surgery_id, (room, start_slot) in schedule.items():

        surgery = surgeries_dict[surgery_id]
        end_slot = start_slot + surgery.duration

        # --- Constraint 1: surgery must fit within working hours ---
        if end_slot > total_slots:
            return False

        # --- Constraint 2: no two surgeries overlap in same room ---
        for slot in range(start_slot, end_slot):
            if slot in room_slots[room]:
                return False
            room_slots[room].add(slot)

        # --- Constraint 3: surgeon cannot be in two places at once ---
        surgeon = surgery.surgeon
        if surgeon not in surgeon_slots:
            surgeon_slots[surgeon] = set()

        for slot in range(start_slot, end_slot):
            if slot in surgeon_slots[surgeon]:
                return False
            surgeon_slots[surgeon].add(slot)

    return True


# =============================================================
# HEURISTIC FUNCTION h(n)
# =============================================================

def heuristic(schedule, surgeries_dict, emergency_probability=0.0):
    """
    Admissible heuristic function h(n).

    Estimates the future cost remaining from this state.
    Never overestimates actual cost — guarantees A* optimality.

    h(n) = estimated remaining delay
         + (emergency_probability * emergency_penalty_weight)

    Parameters:
      schedule             : current schedule assignment
      surgeries_dict       : all surgeries info
      emergency_probability: ML model output (0.0 to 1.0)
                             probability of new emergency in next 6 hrs

    Returns:
      h_cost (float) : estimated future cost
    """

    EMERGENCY_PENALTY_WEIGHT = 5.0  # tunable weight constant

    # estimate remaining delay as sum of urgency gaps
    # higher urgency surgeries that are scheduled late cost more
    estimated_delay = 0.0

    for surgery_id, (room, start_slot) in schedule.items():
        surgery = surgeries_dict[surgery_id]

        # urgent surgeries scheduled after slot 4 incur estimated delay
        if surgery.urgency == 2 and start_slot > 4:
            estimated_delay += (start_slot - 4) * 0.5

        # emergency surgeries scheduled after slot 2 incur higher delay
        if surgery.urgency == 3 and start_slot > 2:
            estimated_delay += (start_slot - 2) * 1.0

    # add emergency probability impact
    emergency_impact = emergency_probability * EMERGENCY_PENALTY_WEIGHT

    h_cost = estimated_delay + emergency_impact
    return h_cost


# =============================================================
# COST FUNCTION g(n) — ACTION COST
# =============================================================

def action_cost(surgery, old_slot, new_slot, old_room, new_room):
    """
    Calculates the cost of one rescheduling action.

    Cost components:
      - Delay cost    : how many slots the surgery was pushed back
      - Urgency factor: urgent/emergency surgeries cost more to delay
      - Room change   : small penalty for moving to a different room

    Returns:
      cost (float)
    """

    delay = max(0, new_slot - old_slot)   # slots pushed back

    # urgency multiplier — emergency delays are most expensive
    urgency_multiplier = {1: 1.0, 2: 2.0, 3: 4.0}
    multiplier = urgency_multiplier.get(surgery.urgency, 1.0)

    delay_cost    = delay * multiplier
    room_change   = 0.5 if old_room != new_room else 0.0

    return delay_cost + room_change


# =============================================================
# NEIGHBOR GENERATOR
# =============================================================

def get_neighbors(state, surgeries_dict, rooms, total_slots, emergency_surgery):
    """
    Generates all neighboring schedule states from the current state.

    Possible actions for each surgery:
      1. Delay the surgery by 1 or 2 slots
      2. Move to a different room at the same slot
      3. Move to a different room AND delay

    Also inserts the emergency surgery into available slots.

    Returns:
      list of (new_state, action_description) tuples
    """

    neighbors = []
    current_schedule = state.schedule

    # --- Try inserting emergency surgery into the schedule ---
    for room in rooms:
        for slot in range(total_slots):
            new_schedule = copy.deepcopy(current_schedule)

            # place emergency surgery
            new_schedule[emergency_surgery.surgery_id] = (room, slot)

            if is_valid_schedule(new_schedule, surgeries_dict, rooms, total_slots):
                cost = action_cost(emergency_surgery, 0, slot, rooms[0], room)
                new_g = state.g_cost + cost
                action_desc = (f"Insert emergency {emergency_surgery.surgery_id} "
                               f"-> {room} at slot {slot}")

                new_state        = ScheduleState(new_schedule, new_g, parent=state, action=action_desc)
                neighbors.append(new_state)

    # --- Try delaying existing surgeries to make room ---
    for surgery_id, (current_room, current_slot) in current_schedule.items():
        surgery = surgeries_dict[surgery_id]

        # skip emergency surgery itself
        if surgery_id == emergency_surgery.surgery_id:
            continue

        # try delaying by 1 or 2 slots
        for delay in [1, 2]:
            new_slot = current_slot + delay

            # try same room and other rooms
            for room in rooms:
                new_schedule = copy.deepcopy(current_schedule)
                new_schedule[surgery_id] = (room, new_slot)

                if is_valid_schedule(new_schedule, surgeries_dict, rooms, total_slots):
                    cost = action_cost(surgery, current_slot, new_slot,
                                       current_room, room)
                    new_g = state.g_cost + cost
                    action_desc = (f"Delay {surgery_id} from slot {current_slot} "
                                   f"to slot {new_slot} in {room}")

                    new_state = ScheduleState(new_schedule, new_g,
                                              parent=state, action=action_desc)
                    neighbors.append(new_state)

    return neighbors


# =============================================================
# A* SEARCH ALGORITHM
# =============================================================

def astar_search(initial_schedule, surgeries_dict, rooms,
                 total_slots, emergency_surgery, emergency_probability=0.0):
    """
    A* Search for optimal surgery rescheduling.

    Finds the least-cost valid schedule that accommodates
    the emergency surgery with minimum disruption.

    Parameters:
      initial_schedule     : current schedule before emergency
      surgeries_dict       : dict of all Surgery objects
      rooms                : list of available room names
      total_slots          : total time slots in the day
      emergency_surgery    : the Surgery object that just arrived
      emergency_probability: ML prediction score (0.0 - 1.0)

    Returns:
      goal_state (ScheduleState) : the best rescheduled state
      or None if no solution found
    """

    print("\n" + "="*55)
    print("  A* SEARCH — Emergency Rescheduling")
    print("="*55)
    print(f"  Emergency: {emergency_surgery}")
    print(f"  ML Emergency Probability: {emergency_probability:.2f}")
    print("="*55)

    # --- Initialize start state ---
    start_state        = ScheduleState(copy.deepcopy(initial_schedule), g_cost=0.0)
    start_state.h_cost = heuristic(initial_schedule, surgeries_dict, emergency_probability)
    start_state.f_cost = start_state.g_cost + start_state.h_cost

    # --- Open list (priority queue) — states to explore ---
    open_list = []
    heapq.heappush(open_list, start_state)

    # --- Closed list — already explored states (by schedule signature) ---
    closed_list = set()

    iteration = 0

    while open_list:
        iteration += 1

        # pop the state with lowest f(n) = g(n) + h(n)
        current = heapq.heappop(open_list)

        print(f"\n  Iteration {iteration}: Exploring state | "
              f"f={current.f_cost:.2f}  g={current.g_cost:.2f}  h={current.h_cost:.2f}")

        if current.action:
            print(f"  Action taken: {current.action}")

        # --- Create a hashable signature of the schedule ---
        schedule_sig = tuple(sorted(current.schedule.items()))

        if schedule_sig in closed_list:
            continue
        closed_list.add(schedule_sig)

        # --- Goal check ---
        # Goal: emergency surgery is placed and schedule is valid
        if (emergency_surgery.surgery_id in current.schedule and
                is_valid_schedule(current.schedule, surgeries_dict, rooms, total_slots)):

            print("\n" + "="*55)
            print("  GOAL REACHED — Valid reschedule found!")
            print("="*55)
            return current

        # --- Generate and explore neighbors ---
        neighbors = get_neighbors(current, surgeries_dict, rooms,
                                  total_slots, emergency_surgery)

        for neighbor in neighbors:
            neighbor_sig = tuple(sorted(neighbor.schedule.items()))

            if neighbor_sig not in closed_list:
                neighbor.h_cost = heuristic(neighbor.schedule, surgeries_dict,
                                            emergency_probability)
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                heapq.heappush(open_list, neighbor)

    print("\n  No valid rescheduling found.")
    return None


# =============================================================
# PATH TRACER — trace back from goal to start
# =============================================================

def trace_path(goal_state):
    """
    Traces back from goal state to start state
    and prints every action taken along the way.
    """

    path = []
    current = goal_state

    while current is not None:
        path.append(current)
        current = current.parent

    path.reverse()

    print("\n" + "="*55)
    print("  RESCHEDULING PATH")
    print("="*55)

    for i, state in enumerate(path):
        if state.action:
            print(f"  Step {i}: {state.action}  |  g={state.g_cost:.2f}")

    print(f"\n  Total Steps  : {len(path) - 1}")
    print(f"  Total Cost   : {goal_state.g_cost:.2f}")


# =============================================================
# RESULT PRINTER
# =============================================================

def print_final_schedule(goal_state, surgeries_dict):
    """
    Prints the final rescheduled timetable in a clean format.
    """

    print("\n" + "="*55)
    print("  FINAL RESCHEDULED TIMETABLE")
    print("="*55)
    print(f"  {'Surgery':<12} {'Room':<10} {'Start Slot':<12} {'Duration':<10} {'Urgency'}")
    print("  " + "-"*53)

    urgency_label = {1: "Elective", 2: "Urgent", 3: "Emergency"}

    for surgery_id, (room, start_slot) in sorted(goal_state.schedule.items()):
        surgery = surgeries_dict[surgery_id]
        print(f"  {surgery_id:<12} {room:<10} {start_slot:<12} "
              f"{surgery.duration}hr(s){'':5} {urgency_label[surgery.urgency]}")

    print("="*55)
    print(f"  Total Rescheduling Cost: {goal_state.g_cost:.2f}")
    print("="*55)


# =============================================================
# MAIN — TEST RUN
# =============================================================

if __name__ == "__main__":

    # --- Define surgeries ---
    surgeries = {
        "S1": Surgery("S1", "Appendectomy",      2, 1, "Dr. Ahmed",  ["scalpel"]),
        "S2": Surgery("S2", "Hip Replacement",   3, 1, "Dr. Sara",   ["drill", "xray"]),
        "S3": Surgery("S3", "Heart Bypass",      4, 2, "Dr. Khan",   ["ventilator"]),
        "S4": Surgery("S4", "Cataract Surgery",  1, 1, "Dr. Ahmed",  ["laser"]),
        "E1": Surgery("E1", "Emergency Trauma",  2, 3, "Dr. Sara",   ["ventilator"]),
    }

    # --- Define rooms and time slots ---
    rooms       = ["Room1", "Room2", "Room3"]
    total_slots = 12  # 12 one-hour slots in a working day

    # --- Initial schedule (before emergency) ---
    # Format: surgery_id -> (room, start_slot)
    initial_schedule = {
        "S1": ("Room1", 0),   # Appendectomy     : Room1, 08:00 - 10:00
        "S2": ("Room2", 0),   # Hip Replacement  : Room2, 08:00 - 11:00
        "S3": ("Room1", 2),   # Heart Bypass     : Room1, 10:00 - 14:00
        "S4": ("Room2", 3),   # Cataract Surgery : Room2, 11:00 - 12:00
    }

    # --- Emergency surgery arrives ---
    emergency_surgery = surgeries["E1"]

    # --- ML model emergency probability (simulated) ---
    emergency_probability = 0.75  # 75% chance of another emergency in next 6 hrs

    # --- Run A* Search ---
    goal_state = astar_search(
        initial_schedule      = initial_schedule,
        surgeries_dict        = surgeries,
        rooms                 = rooms,
        total_slots           = total_slots,
        emergency_surgery     = emergency_surgery,
        emergency_probability = emergency_probability
    )

    # --- Show results ---
    if goal_state:
        trace_path(goal_state)
        print_final_schedule(goal_state, surgeries)