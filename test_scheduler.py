"""
=============================================================
AI2002 - Artificial Intelligence | Spring 2026
Project: AI-Driven Hospital Surgery Scheduling System
File   : test_scheduler.py
Topic  : Pytest Test Cases for A* Rescheduling Algorithm

Members:
     Qadeer Raza   , 23i-3059
     Abdullah Tariq, 23i-3011
     Malik Usman   , 23i-3020

Run with:
    python -m pytest test_scheduler.py -v
=============================================================

NOTE: This file is self-contained.
The core algorithmic classes and functions are copied directly
from app.py so that the Streamlit UI layer is never executed.
"""

import heapq
import copy
import pytest


# =============================================================
# DATA STRUCTURES  (copied from app.py — pure logic, no Streamlit)
# =============================================================

class Surgery:
    def __init__(self, surgery_id, name, duration, urgency,
                 surgeon, equipment, requires_icu=False):
        self.surgery_id   = surgery_id
        self.name         = name
        self.duration     = duration
        self.urgency      = urgency
        self.surgeon      = surgeon
        self.equipment    = equipment
        self.requires_icu = requires_icu


class ScheduleState:
    def __init__(self, schedule, g_cost, parent=None, action=None):
        self.schedule = schedule
        self.g_cost   = g_cost
        self.parent   = parent
        self.action   = action
        self.h_cost   = 0.0
        self.f_cost   = 0.0

    def __lt__(self, other):
        if self.f_cost == other.f_cost:
            return self.h_cost < other.h_cost
        return self.f_cost < other.f_cost


# =============================================================
# CONSTRAINT VALIDATOR  (copied from app.py)
# =============================================================

def is_valid_schedule(schedule, surgeries_dict, rooms,
                      total_slots, num_icu_beds=999):
    room_slots    = {room: set() for room in rooms}
    surgeon_slots = {}
    equip_slots   = {}
    icu_slots     = {}

    for surgery_id, (room, start_slot) in schedule.items():
        surgery = surgeries_dict.get(surgery_id)
        if surgery is None:
            return False

        end_slot = start_slot + surgery.duration

        # Constraint 1 — Working Hours
        if start_slot < 0 or end_slot > total_slots:
            return False

        # Constraint 2 — Room Exclusivity
        for slot in range(start_slot, end_slot):
            if slot in room_slots[room]:
                return False

        # Constraint 3 — Surgeon Uniqueness
        surgeon = surgery.surgeon
        if surgeon not in surgeon_slots:
            surgeon_slots[surgeon] = set()
        for slot in range(start_slot, end_slot):
            if slot in surgeon_slots[surgeon]:
                return False

        # Constraint 4 — Equipment Conflict
        for item in surgery.equipment:
            if item not in equip_slots:
                equip_slots[item] = {}
            for slot in range(start_slot, end_slot):
                if slot not in equip_slots[item]:
                    equip_slots[item][slot] = set()
                existing = equip_slots[item][slot]
                if existing and room not in existing:
                    return False

        # Constraint 5 — ICU Availability
        if surgery.requires_icu:
            for slot in range(start_slot, end_slot):
                if icu_slots.get(slot, 0) >= num_icu_beds:
                    return False

        # Commit to tracking structures
        for slot in range(start_slot, end_slot):
            room_slots[room].add(slot)
            surgeon_slots[surgeon].add(slot)
            for item in surgery.equipment:
                equip_slots[item][slot].add(room)
            if surgery.requires_icu:
                icu_slots[slot] = icu_slots.get(slot, 0) + 1

    return True


# =============================================================
# HEURISTIC  h(n)  (copied from app.py)
# =============================================================

def heuristic(schedule, surgeries_dict, emergency_probability=0.0):
    EMERGENCY_PENALTY_WEIGHT = 0.5
    estimated_delay = 0.0
    for surgery_id, (room, start_slot) in schedule.items():
        surgery = surgeries_dict[surgery_id]
        if surgery.urgency == 2 and start_slot > 4:
            estimated_delay += (start_slot - 4) * 0.5
        if surgery.urgency == 3 and start_slot > 2:
            estimated_delay += (start_slot - 2) * 1.0
    return estimated_delay + (emergency_probability * EMERGENCY_PENALTY_WEIGHT)


# =============================================================
# ACTION COST  g(n)  (copied from app.py)
# =============================================================

def action_cost(surgery, old_slot, new_slot, old_room, new_room):
    delay      = max(0, new_slot - old_slot)
    multiplier = {1: 1.0, 2: 2.0, 3: 4.0}.get(surgery.urgency, 1.0)
    return (delay * multiplier) + (0.5 if old_room != new_room else 0.0)


# =============================================================
# NEIGHBOR GENERATOR  (copied from app.py)
# =============================================================

def _make_schedule(base, key, value):
    s = dict(base)
    s[key] = value
    return s


def get_neighbors(state, surgeries_dict, rooms, total_slots,
                  emergency_surgery, num_icu_beds=999):
    neighbors        = []
    current_schedule = state.schedule
    em_id            = emergency_surgery.surgery_id
    em_dur           = emergency_surgery.duration

    for room in rooms:
        for slot in range(total_slots - em_dur + 1):
            new_schedule = _make_schedule(current_schedule, em_id, (room, slot))
            if is_valid_schedule(new_schedule, surgeries_dict,
                                 rooms, total_slots, num_icu_beds):
                cost      = action_cost(emergency_surgery, 0, slot, rooms[0], room)
                new_state = ScheduleState(new_schedule, state.g_cost + cost,
                                          parent=state,
                                          action=f"Insert {em_id} -> {room} at slot {slot}")
                neighbors.append(new_state)

    for surgery_id, (current_room, current_slot) in current_schedule.items():
        if surgery_id == em_id:
            continue
        surgery = surgeries_dict[surgery_id]
        for delay in [1, 2]:
            new_slot = current_slot + delay
            if new_slot + surgery.duration > total_slots:
                continue
            for room in ([current_room] + [r for r in rooms if r != current_room]):
                new_schedule = _make_schedule(current_schedule, surgery_id, (room, new_slot))
                if is_valid_schedule(new_schedule, surgeries_dict,
                                     rooms, total_slots, num_icu_beds):
                    cost      = action_cost(surgery, current_slot, new_slot, current_room, room)
                    new_state = ScheduleState(new_schedule, state.g_cost + cost,
                                              parent=state,
                                              action=(f"Delay {surgery_id}: "
                                                      f"slot {current_slot} -> {new_slot} in {room}"))
                    neighbors.append(new_state)
                    break

    return neighbors


# =============================================================
# A* SEARCH  (copied from app.py)
# =============================================================

def astar_search(initial_schedule, surgeries_dict, rooms, total_slots,
                 emergency_surgery, emergency_probability=0.0, num_icu_beds=999):

    logs        = []
    start_state = ScheduleState(dict(initial_schedule), g_cost=0.0)
    start_state.h_cost = heuristic(initial_schedule, surgeries_dict, emergency_probability)
    start_state.f_cost = start_state.g_cost + start_state.h_cost

    open_list   = []
    closed_list = set()
    heapq.heappush(open_list, start_state)
    iteration   = 0

    while open_list:
        iteration += 1
        if iteration > 50000:
            return None, logs

        current = heapq.heappop(open_list)
        logs.append({
            "Iteration": iteration,
            "Action"   : current.action if current.action else "Start state",
            "g(n)"     : round(current.g_cost, 2),
            "h(n)"     : round(current.h_cost, 2),
            "f(n)"     : round(current.f_cost, 2),
        })

        schedule_sig = tuple(sorted(current.schedule.items()))
        if schedule_sig in closed_list:
            continue
        closed_list.add(schedule_sig)

        if (emergency_surgery.surgery_id in current.schedule and
                is_valid_schedule(current.schedule, surgeries_dict,
                                  rooms, total_slots, num_icu_beds)):
            return current, logs

        for neighbor in get_neighbors(current, surgeries_dict, rooms,
                                      total_slots, emergency_surgery, num_icu_beds):
            neighbor_sig = tuple(sorted(neighbor.schedule.items()))
            if neighbor_sig not in closed_list:
                neighbor.h_cost = heuristic(neighbor.schedule, surgeries_dict,
                                            emergency_probability)
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                heapq.heappush(open_list, neighbor)

    return None, logs


# =============================================================
# HELPER
# =============================================================

def slot_to_time(slot):
    return f"{8 + slot:02d}:00"


# =============================================================
# TEST CASE 1 — Emergency placed in free room (no rescheduling)
# =============================================================

class TestCase1EmergencyFreeRoom:
    """
    Setup : 3 rooms | 16 hours | 2 ICU beds
    S1: Appendectomy    | Room 1 | 08:00 | 3 hrs
    S2: Hip Replacement | Room 2 | 08:00 | 4 hrs
    E1: Trauma Surgery  | Dr. Zaid | 2 hrs | ventilator | ICU

    Expected:
        E1 placed in Room 3 at slot 0 (08:00)
        g(n) = 0.50  (no elective disturbed; 0.5 room-change penalty applied
                      because action_cost uses rooms[0]='Room 1' as reference
                      and E1 lands in 'Room 3' → old_room != new_room)
        Iterations <= 2
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rooms       = ["Room 1", "Room 2", "Room 3"]
        self.total_slots = 16
        self.num_icu     = 2

        self.surgeries = {
            "S1": Surgery("S1", "Appendectomy",    3, 1, "Dr. Ahmed",
                          ["scalpel"], requires_icu=False),
            "S2": Surgery("S2", "Hip Replacement", 4, 1, "Dr. Sara",
                          ["drill", "xray"], requires_icu=False),
        }
        self.initial_schedule = {
            "S1": ("Room 1", 0),
            "S2": ("Room 2", 0),
        }
        self.emergency = Surgery("E1", "Trauma Surgery", 2, 3, "Dr. Zaid",
                                 ["ventilator"], requires_icu=True)
        self.sdict = {**self.surgeries, "E1": self.emergency}

    def _run(self):
        return astar_search(
            initial_schedule      = self.initial_schedule,
            surgeries_dict        = self.sdict,
            rooms                 = self.rooms,
            total_slots           = self.total_slots,
            emergency_surgery     = self.emergency,
            emergency_probability = 0.0,
            num_icu_beds          = self.num_icu,
        )

    def test_solution_found(self):
        goal, _ = self._run()
        assert goal is not None, "A* must find a solution — Room 3 is free"

    def test_e1_placed_in_room3_slot0(self):
        goal, _ = self._run()
        assert goal is not None
        room, slot = goal.schedule["E1"]
        assert room == "Room 3", f"Expected Room 3, got {room}"
        assert slot == 0,        f"Expected slot 0 (08:00), got slot {slot}"

    def test_rescheduling_cost_is_0_5(self):
        """
        g(n) = 0.50.
        No elective surgery was delayed (0 delay cost).
        E1 is inserted into Room 3 while action_cost uses rooms[0]='Room 1'
        as the reference room, so old_room != new_room → +0.5 room-change
        penalty. This is the minimum possible cost for this scenario.
        """
        goal, _ = self._run()
        assert goal is not None
        assert goal.g_cost == pytest.approx(0.5, abs=1e-6), \
            f"Expected g(n)=0.50 (room-change penalty only), got {goal.g_cost:.2f}"

    def test_final_schedule_valid(self):
        goal, _ = self._run()
        assert goal is not None
        assert is_valid_schedule(goal.schedule, self.sdict,
                                 self.rooms, self.total_slots, self.num_icu)

    def test_iterations_at_most_2(self):
        goal, logs = self._run()
        assert goal is not None
        assert len(logs) <= 2, f"Expected ≤2 iterations, got {len(logs)}"


# =============================================================
# TEST CASE 2 — Surgeon conflict forces delay
# =============================================================

class TestCase2SurgeonConflictDelay:
    """
    Setup : 3 rooms | 16 hours | 2 ICU beds
    S1: Heart Bypass | Dr. Sara | Room 1 | 08:00 | 5 hrs | ICU
    E1: Cardiac Emergency | Dr. Sara | 2 hrs | ecg_monitor | ICU

    Expected:
        Surgeon uniqueness blocks direct insert of E1 at slot 0
        S1 delayed to slot 2 (10:00); E1 inserted at Room 1 slot 0
        g(n) = 4.0  (2 slots * urgency multiplier 2.0)
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rooms       = ["Room 1", "Room 2", "Room 3"]
        self.total_slots = 16
        self.num_icu     = 2

        self.surgeries = {
            "S1": Surgery("S1", "Heart Bypass", 5, 2, "Dr. Sara",
                          ["ventilator"], requires_icu=True),
        }
        self.initial_schedule = {"S1": ("Room 1", 0)}
        self.emergency = Surgery("E1", "Cardiac Emergency", 2, 3, "Dr. Sara",
                                 ["ecg_monitor"], requires_icu=True)
        self.sdict = {**self.surgeries, "E1": self.emergency}

    def _run(self):
        return astar_search(
            initial_schedule      = self.initial_schedule,
            surgeries_dict        = self.sdict,
            rooms                 = self.rooms,
            total_slots           = self.total_slots,
            emergency_surgery     = self.emergency,
            emergency_probability = 0.0,
            num_icu_beds          = self.num_icu,
        )

    def test_solution_found(self):
        goal, _ = self._run()
        assert goal is not None, "A* must find a solution via delay"

    def test_e1_at_slot0(self):
        goal, _ = self._run()
        assert goal is not None
        _, slot = goal.schedule["E1"]
        assert slot == 0, f"E1 must be at slot 0 (08:00), got slot {slot}"

    def test_s1_delayed_to_slot2(self):
        goal, _ = self._run()
        assert goal is not None
        _, s1_slot = goal.schedule["S1"]
        assert s1_slot == 2, \
            f"S1 must be delayed to slot 2 (10:00), got slot {s1_slot} ({slot_to_time(s1_slot)})"

    def test_rescheduling_cost_is_4(self):
        """2 slots delay * urgency multiplier 2.0 = 4.0"""
        goal, _ = self._run()
        assert goal is not None
        assert goal.g_cost == pytest.approx(4.0, abs=1e-6), \
            f"Expected g(n)=4.0, got {goal.g_cost:.2f}"

    def test_surgeon_conflict_detected(self):
        """Direct insert without delay must fail constraint check."""
        conflicting = {"S1": ("Room 1", 0), "E1": ("Room 1", 0)}
        assert not is_valid_schedule(conflicting, self.sdict,
                                     self.rooms, self.total_slots, self.num_icu), \
            "Placing E1 and S1 both at slot 0 with Dr. Sara must violate surgeon uniqueness"

    def test_final_schedule_valid(self):
        goal, _ = self._run()
        assert goal is not None
        assert is_valid_schedule(goal.schedule, self.sdict,
                                 self.rooms, self.total_slots, self.num_icu)


# =============================================================
# TEST CASE 3 — Multiple consecutive emergencies
# =============================================================

class TestCase3ConsecutiveEmergencies:
    """
    Setup : 3 rooms | 16 hours | 2 ICU beds | autofill scenario (5 surgeries)
    E1: Trauma Surgery | Dr. Zaid  | 2 hrs | ventilator | ICU
    E2: Burn Case      | Dr. Malik | 3 hrs | xray       | ICU
        (injected AFTER E1 is committed — builds on post-E1 schedule)

    Expected:
        Both E1 and E2 in the final schedule, in separate rooms (or non-overlapping).
        ICU constraint satisfied (2 beds) throughout.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rooms       = ["Room 1", "Room 2", "Room 3"]
        self.total_slots = 16
        self.num_icu     = 2

        self.surgeries = {
            "S1": Surgery("S1", "Appendectomy",    3, 1, "Dr. Ahmed",
                          ["scalpel", "cauterizer"],     requires_icu=False),
            "S2": Surgery("S2", "Hip Replacement", 4, 1, "Dr. Sara",
                          ["drill", "xray"],             requires_icu=True),
            "S3": Surgery("S3", "Heart Bypass",    5, 2, "Dr. Khan",
                          ["ventilator", "ecg_monitor"], requires_icu=True),
            "S4": Surgery("S4", "Cataract Surgery",2, 1, "Dr. Ahmed",
                          ["laser", "camera"],           requires_icu=False),
            "S5": Surgery("S5", "Spinal Fusion",   3, 2, "Dr. Sara",
                          ["xray", "drill"],             requires_icu=False),
        }
        self.initial_schedule = {
            "S1": ("Room 1", 0),
            "S2": ("Room 2", 0),
            "S3": ("Room 3", 0),
            "S4": ("Room 1", 5),
            "S5": ("Room 2", 6),
        }
        self.e1 = Surgery("E1", "Trauma Surgery", 2, 3, "Dr. Zaid",
                          ["ventilator"], requires_icu=True)
        self.e2 = Surgery("E2", "Burn Case",      3, 3, "Dr. Malik",
                          ["xray"],       requires_icu=True)

    def _place_e1(self):
        sdict = {**self.surgeries, "E1": self.e1}
        goal, logs = astar_search(
            initial_schedule      = self.initial_schedule,
            surgeries_dict        = sdict,
            rooms                 = self.rooms,
            total_slots           = self.total_slots,
            emergency_surgery     = self.e1,
            emergency_probability = 0.0,
            num_icu_beds          = self.num_icu,
        )
        return goal, sdict

    def _place_both(self):
        goal1, sdict1 = self._place_e1()
        assert goal1 is not None, "Prerequisite: E1 must be placed first"
        sdict2 = {**sdict1, "E2": self.e2}
        goal2, logs2 = astar_search(
            initial_schedule      = goal1.schedule,   # committed post-E1 baseline
            surgeries_dict        = sdict2,
            rooms                 = self.rooms,
            total_slots           = self.total_slots,
            emergency_surgery     = self.e2,
            emergency_probability = 0.0,
            num_icu_beds          = self.num_icu,
        )
        return goal2, sdict2

    def test_e1_placed_successfully(self):
        goal1, _ = self._place_e1()
        assert goal1 is not None, "E1 must be placed"
        assert "E1" in goal1.schedule

    def test_e2_placed_after_e1(self):
        goal2, _ = self._place_both()
        assert goal2 is not None, "E2 must be placed after E1 is committed"
        assert "E2" in goal2.schedule

    def test_both_present_in_final_schedule(self):
        goal2, _ = self._place_both()
        assert goal2 is not None
        assert "E1" in goal2.schedule, "E1 must still be in final schedule"
        assert "E2" in goal2.schedule, "E2 must be in final schedule"

    def test_e1_e2_no_room_time_overlap(self):
        goal2, sdict = self._place_both()
        assert goal2 is not None
        room_e1, slot_e1 = goal2.schedule["E1"]
        room_e2, slot_e2 = goal2.schedule["E2"]
        if room_e1 == room_e2:
            end_e1  = slot_e1 + self.e1.duration
            overlap = not (slot_e2 >= end_e1 or (slot_e2 + self.e2.duration) <= slot_e1)
            assert not overlap, \
                f"E1 and E2 overlap in {room_e1} — room exclusivity violated"

    def test_icu_constraint_satisfied(self):
        goal2, sdict = self._place_both()
        assert goal2 is not None
        assert is_valid_schedule(goal2.schedule, sdict,
                                 self.rooms, self.total_slots, self.num_icu), \
            "ICU constraint (2 beds) must be satisfied in final schedule"

    def test_icu_count_at_least_2(self):
        """After both ICU emergencies, at least 2 ICU surgeries must be scheduled."""
        goal2, sdict = self._place_both()
        assert goal2 is not None
        icu_count = sum(
            1 for sid in goal2.schedule
            if sdict.get(sid) and sdict[sid].requires_icu
        )
        assert icu_count >= 2, \
            f"Expected ≥2 ICU surgeries in schedule, found {icu_count}"

    def test_final_schedule_fully_valid(self):
        goal2, sdict = self._place_both()
        assert goal2 is not None
        assert is_valid_schedule(goal2.schedule, sdict,
                                 self.rooms, self.total_slots, self.num_icu)


# =============================================================
# TEST CASE 4 — Edge case: no valid rescheduling possible
# =============================================================

class TestCase4NoValidReschedulingPossible:
    """
    Setup : 1 room | 4 hours | 1 ICU bed
    S1: 4-hour surgery filling the entire room (Room 1, slot 0)
    E1: 2-hour surgery | ICU required

    Expected:
        A* returns None — open list exhausted with no valid goal state
        UI: "A* could not place E1. Try more rooms or ICU beds."
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rooms       = ["Room 1"]
        self.total_slots = 4
        self.num_icu     = 1

        self.surgeries = {
            "S1": Surgery("S1", "Full Day Surgery", 4, 1, "Dr. Ahmed",
                          ["scalpel"], requires_icu=False),
        }
        self.initial_schedule = {"S1": ("Room 1", 0)}
        self.emergency = Surgery("E1", "Emergency Surgery", 2, 3, "Dr. Zaid",
                                 ["ventilator"], requires_icu=True)
        self.sdict = {**self.surgeries, "E1": self.emergency}

    def _run(self):
        return astar_search(
            initial_schedule      = self.initial_schedule,
            surgeries_dict        = self.sdict,
            rooms                 = self.rooms,
            total_slots           = self.total_slots,
            emergency_surgery     = self.emergency,
            emergency_probability = 0.0,
            num_icu_beds          = self.num_icu,
        )

    def test_astar_returns_none(self):
        """A* must return None when no slot is available."""
        goal, _ = self._run()
        assert goal is None, \
            "A* must return None — room fully blocked, no valid placement exists"

    def test_logs_returned_on_failure(self):
        """Logs list must still be returned even when A* fails."""
        goal, logs = self._run()
        assert goal is None
        assert isinstance(logs, list)

    def test_no_valid_insert_slot_exists(self):
        """Manual sweep confirms every slot is blocked."""
        found_valid = False
        for slot in range(self.total_slots):
            candidate = {**self.initial_schedule, "E1": ("Room 1", slot)}
            if is_valid_schedule(candidate, self.sdict,
                                 self.rooms, self.total_slots, self.num_icu):
                found_valid = True
                break
        assert not found_valid, \
            "No slot in Room 1 should be valid when S1 fills the entire 4-hour window"

    def test_initial_schedule_is_valid(self):
        """Sanity check: S1 alone must be valid before emergency arrives."""
        assert is_valid_schedule(self.initial_schedule, self.surgeries,
                                 self.rooms, self.total_slots, self.num_icu)

    def test_delay_s1_also_impossible(self):
        """Delaying S1 by 1 slot overflows working hours — no delay escape exists."""
        delayed = {"S1": ("Room 1", 1)}
        assert not is_valid_schedule(delayed, self.surgeries,
                                     self.rooms, self.total_slots, self.num_icu), \
            "S1 at slot 1 with duration 4 → end=5 > total_slots=4 must be invalid"
