"""
=============================================================
AI2002 - Artificial Intelligence | Spring 2026
Project: AI-Driven Hospital Surgery Scheduling System

app.py  —  Main Streamlit dashboard
           Run with:  streamlit run app.py

           IMPORTANT: Run train_model.py first to generate
                      emergency_model.pkl before launching.

Members:
     Qadeer Raza   , 23i-3059
     Abdullah Tariq, 23i-3011
     Malik Usman   , 23i-3020

Constraints Implemented:
    1. Room Exclusivity      — one surgery per room per slot
    2. Surgeon Uniqueness    — surgeon cannot be in two rooms simultaneously
    3. Working Hours         — surgery must finish within total working hours
    4. Equipment Conflict    — same equipment cannot be used in two rooms at same time
    5. ICU Availability      — surgery needing ICU is only scheduled if a bed is free
=============================================================
"""

import os
import copy
import heapq
import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================
# PAGE CONFIG
# =============================================================

st.set_page_config(
    page_title="Hospital Surgery Scheduler",
    page_icon="🏥",
    layout="wide"
)

# =============================================================
# CUSTOM CSS
# =============================================================

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1B3A6B;
    }
    .metric-label {
        font-size: 11px;
        color: #666;
        margin-top: 4px;
    }
    .section-title {
        font-size: 15px;
        font-weight: bold;
        color: #1B3A6B;
        padding-bottom: 4px;
        border-bottom: 2px solid #1B3A6B;
        margin-bottom: 10px;
    }
    .emergency-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .success-box {
        background: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .ml-box {
        background: #e8f4fd;
        border-left: 5px solid #2E7BC4;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .ml-box-warn {
        background: #fff8e1;
        border-left: 5px solid #ff9800;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .ml-box-danger {
        background: #fdecea;
        border-left: 5px solid #D0021B;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .constraint-badge {
        display: inline-block;
        background: #1B3A6B;
        color: white;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 11px;
        margin: 2px 3px;
    }
    .log-entry {
        background: #f1f3f5;
        border-left: 3px solid #2E7BC4;
        padding: 6px 10px;
        margin: 4px 0;
        font-size: 13px;
        font-family: monospace;
        border-radius: 0 4px 4px 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================
# LOAD ML MODEL
# =============================================================

MODEL_PATH = "emergency_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

ml_model = load_model()


# =============================================================
# ML PREDICTION
# =============================================================

def predict_emergency_probability(schedule, surgeries_dict, total_slots):
    now         = datetime.datetime.now()
    hour_of_day = now.hour
    is_weekend  = 1 if now.weekday() >= 5 else 0

    num_surgeries = len(schedule)
    num_urgent    = sum(
        1 for sid in schedule
        if surgeries_dict.get(sid) and surgeries_dict[sid].urgency >= 2
    )

    total_occupied = sum(
        surgeries_dict[sid].duration
        for sid in schedule if surgeries_dict.get(sid)
    )
    max_possible   = max(total_slots * max(len(schedule), 1), 1)
    room_occupancy = min(total_occupied / max_possible, 1.0)

    durations    = [surgeries_dict[sid].duration
                    for sid in schedule if surgeries_dict.get(sid)]
    avg_duration = float(np.mean(durations)) if durations else 2.0

    features = pd.DataFrame([{
        "hour_of_day"          : hour_of_day,
        "is_weekend"           : is_weekend,
        "num_surgeries"        : num_surgeries,
        "num_urgent"           : num_urgent,
        "room_occupancy_pct"   : round(room_occupancy, 2),
        "avg_surgery_duration" : round(avg_duration, 1),
    }])

    if ml_model is not None:
        prob = float(ml_model.predict_proba(features)[0][1])
    else:
        prob = 0.10 + (num_urgent * 0.05) + (room_occupancy * 0.20)
        if 17 <= hour_of_day <= 23:
            prob += 0.25
        if is_weekend:
            prob += 0.10
        prob = min(prob, 1.0)

    if prob < 0.35:
        risk_label = "🟢 Low Risk"
        risk_class = "ml-box"
    elif prob < 0.65:
        risk_label = "🟡 Medium Risk"
        risk_class = "ml-box-warn"
    else:
        risk_label = "🔴 High Risk"
        risk_class = "ml-box-danger"

    return prob, features, risk_label, risk_class


# =============================================================
# DATA STRUCTURES
# =============================================================

class Surgery:
    def __init__(self, surgery_id, name, duration, urgency,
                 surgeon, equipment, requires_icu=False):
        self.surgery_id   = surgery_id
        self.name         = name
        self.duration     = duration
        self.urgency      = urgency
        self.surgeon      = surgeon
        self.equipment    = equipment        # list[str]
        self.requires_icu = requires_icu     # bool


class ScheduleState:
    def __init__(self, schedule, g_cost, parent=None, action=None):
        self.schedule = schedule
        self.g_cost   = g_cost
        self.parent   = parent
        self.action   = action
        self.h_cost   = 0.0
        self.f_cost   = 0.0

    def __lt__(self, other):
        # Tie-break by h(n) when f(n) is equal — prioritise states closer to goal
        if self.f_cost == other.f_cost:
            return self.h_cost < other.h_cost
        return self.f_cost < other.f_cost


# =============================================================
# CONSTRAINT VALIDATION — all 5 constraints
# =============================================================

def is_valid_schedule(schedule, surgeries_dict, rooms,
                      total_slots, num_icu_beds=999):
    """
    Hard Constraint 1 — Working Hours:
        start_slot >= 0  and  start_slot + duration <= total_slots

    Hard Constraint 2 — Room Exclusivity:
        No two surgeries share the same room at the same time slot.

    Hard Constraint 3 — Surgeon Uniqueness:
        A surgeon cannot operate in two rooms simultaneously.

    Hard Constraint 4 — Equipment Conflict:
        The same equipment item cannot be used in two DIFFERENT rooms
        at the same time slot.

    Hard Constraint 5 — ICU Availability:
        The number of ICU-requiring surgeries running concurrently
        must not exceed num_icu_beds.
    """

    room_slots    = {room: set() for room in rooms}
    surgeon_slots = {}
    # equip_slots[item][slot] = set of rooms currently using it
    equip_slots   = {}
    # icu_slots[slot] = count of concurrent ICU surgeries
    icu_slots     = {}

    for surgery_id, (room, start_slot) in schedule.items():
        surgery = surgeries_dict.get(surgery_id)
        if surgery is None:
            return False

        # ── Fix 1: Guard against rooms that no longer exist in the room list ──
        # Prevents KeyError crash when the user reduces the room count after
        # surgeries have already been added to a now-removed room.
        if room not in room_slots:
            return False

        end_slot = start_slot + surgery.duration

        # ── Constraint 1: Working Hours ───────────────────────
        if start_slot < 0 or end_slot > total_slots:
            return False

        # ── Constraint 2: Room Exclusivity ────────────────────
        for slot in range(start_slot, end_slot):
            if slot in room_slots[room]:
                return False

        # ── Constraint 3: Surgeon Uniqueness ──────────────────
        surgeon = surgery.surgeon
        if surgeon not in surgeon_slots:
            surgeon_slots[surgeon] = set()
        for slot in range(start_slot, end_slot):
            if slot in surgeon_slots[surgeon]:
                return False

        # ── Constraint 4: Equipment Conflict ──────────────────
        for item in surgery.equipment:
            if item not in equip_slots:
                equip_slots[item] = {}
            for slot in range(start_slot, end_slot):
                if slot not in equip_slots[item]:
                    equip_slots[item][slot] = set()
                existing = equip_slots[item][slot]
                # conflict only when same equipment used in a DIFFERENT room
                if existing and room not in existing:
                    return False

        # ── Constraint 5: ICU Availability ────────────────────
        if surgery.requires_icu:
            for slot in range(start_slot, end_slot):
                if icu_slots.get(slot, 0) >= num_icu_beds:
                    return False

        # ── Commit this surgery to tracking structures ─────────
        for slot in range(start_slot, end_slot):
            room_slots[room].add(slot)
            surgeon_slots[surgeon].add(slot)
            for item in surgery.equipment:
                equip_slots[item][slot].add(room)
            if surgery.requires_icu:
                icu_slots[slot] = icu_slots.get(slot, 0) + 1

    return True


# =============================================================
# HEURISTIC  h(n)
# =============================================================

def heuristic(schedule, surgeries_dict, emergency_probability=0.0):
    # EMERGENCY_PENALTY_WEIGHT is kept at 0.5 to ensure h(n) stays admissible.
    # emergency_probability is in [0,1] so the penalty is bounded at [0, 0.5],
    # which is always a conservative lower-bound estimate of future cost.
    # The original value of 5.0 could overestimate and break A* optimality.
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
# ACTION COST  g(n)
# =============================================================

def action_cost(surgery, old_slot, new_slot, old_room, new_room):
    delay      = max(0, new_slot - old_slot)
    multiplier = {1: 1.0, 2: 2.0, 3: 4.0}.get(surgery.urgency, 1.0)
    return (delay * multiplier) + (0.5 if old_room != new_room else 0.0)


# =============================================================
# NEIGHBOR GENERATION
# =============================================================

def _make_schedule(base, key, value):
    """Fast shallow copy with one update — avoids expensive deepcopy."""
    s = dict(base)
    s[key] = value
    return s


def get_neighbors(state, surgeries_dict, rooms, total_slots,
                  emergency_surgery, num_icu_beds=999):
    neighbors        = []
    current_schedule = state.schedule
    em_id            = emergency_surgery.surgery_id
    em_dur           = emergency_surgery.duration

    # ── Try inserting emergency into every room/slot ──────────
    # Optimisation: only test slots where the emergency can
    # physically finish within working hours (prune invalid slots early)
    for room in rooms:
        for slot in range(total_slots - em_dur + 1):   # prune slots that overflow
            new_schedule = _make_schedule(current_schedule, em_id, (room, slot))

            if is_valid_schedule(new_schedule, surgeries_dict,
                                 rooms, total_slots, num_icu_beds):
                cost      = action_cost(emergency_surgery, 0, slot, rooms[0], room)
                new_state = ScheduleState(new_schedule, state.g_cost + cost,
                                          parent=state,
                                          action=f"Insert {em_id} → {room} at slot {slot}")
                neighbors.append(new_state)

    # ── Try delaying existing surgeries by 1 or 2 slots ──────
    # Reduced from [1,2,3] to [1,2] — keeps branching factor manageable
    # Same room first (cheaper), then other rooms only if needed
    for surgery_id, (current_room, current_slot) in current_schedule.items():
        if surgery_id == em_id:
            continue

        surgery = surgeries_dict[surgery_id]

        for delay in [1, 2]:
            new_slot = current_slot + delay
            # prune: if delayed surgery overflows working hours, skip entirely
            if new_slot + surgery.duration > total_slots:
                continue

            # try same room first (no room-change penalty, cheaper)
            for room in ([current_room] +
                         [r for r in rooms if r != current_room]):
                new_schedule = _make_schedule(current_schedule,
                                              surgery_id, (room, new_slot))

                if is_valid_schedule(new_schedule, surgeries_dict,
                                     rooms, total_slots, num_icu_beds):
                    cost      = action_cost(surgery, current_slot,
                                            new_slot, current_room, room)
                    new_state = ScheduleState(new_schedule,
                                              state.g_cost + cost,
                                              parent=state,
                                              action=(f"Delay {surgery_id}: "
                                                      f"slot {current_slot}"
                                                      f" → {new_slot} in {room}"))
                    neighbors.append(new_state)
                    break   # found valid placement for this delay — stop trying rooms

    return neighbors


# =============================================================
# A* SEARCH
# =============================================================

def astar_search(initial_schedule, surgeries_dict, rooms, total_slots,
                 emergency_surgery, emergency_probability=0.0,
                 num_icu_beds=999, max_iterations=50000):

    MAX_TREE_NODES = 25          # cap nodes captured for tree visualisation
    logs             = []
    tree_nodes       = []        # captured explored nodes for tree drawing
    state_id_to_idx  = {}        # id(ScheduleState) -> index in tree_nodes

    start_state = ScheduleState(dict(initial_schedule), g_cost=0.0)
    start_state.h_cost = heuristic(initial_schedule, surgeries_dict,
                                    emergency_probability)
    start_state.f_cost = start_state.g_cost + start_state.h_cost

    open_list   = []
    closed_list = set()
    heapq.heappush(open_list, start_state)
    iteration   = 0

    while open_list:
        iteration += 1

        if iteration > max_iterations:
            return None, logs, tree_nodes

        current = heapq.heappop(open_list)

        logs.append({
            "Iteration" : iteration,
            "Action"    : current.action if current.action else "Start state",
            "g(n)"      : round(current.g_cost, 2),
            "h(n)"      : round(current.h_cost, 2),
            "f(n)"      : round(current.f_cost, 2),
        })

        schedule_sig = tuple(sorted(current.schedule.items()))
        if schedule_sig in closed_list:
            continue
        closed_list.add(schedule_sig)

        # ── Capture node for tree visualisation (limited to MAX_TREE_NODES) ──
        if len(tree_nodes) < MAX_TREE_NODES:
            parent_idx = state_id_to_idx.get(id(current.parent), -1)
            node_idx   = len(tree_nodes)
            state_id_to_idx[id(current)] = node_idx
            is_goal_node = (
                emergency_surgery.surgery_id in current.schedule and
                is_valid_schedule(current.schedule, surgeries_dict,
                                  rooms, total_slots, num_icu_beds)
            )
            tree_nodes.append({
                "node_id"   : node_idx,
                "schedule"  : dict(current.schedule),
                "g"         : round(current.g_cost, 2),
                "h"         : round(current.h_cost, 2),
                "f"         : round(current.f_cost, 2),
                "action"    : current.action,
                "parent_id" : parent_idx,
                "is_goal"   : is_goal_node,
            })

        is_goal = (
            emergency_surgery.surgery_id in current.schedule and
            is_valid_schedule(current.schedule, surgeries_dict,
                              rooms, total_slots, num_icu_beds)
        )
        if is_goal:
            # Mark the path nodes in tree_nodes by tracing parent chain
            path_state = current
            while path_state is not None:
                idx = state_id_to_idx.get(id(path_state))
                if idx is not None and idx < len(tree_nodes):
                    tree_nodes[idx]["on_path"] = True
                path_state = path_state.parent
            return current, logs, tree_nodes

        for neighbor in get_neighbors(current, surgeries_dict, rooms,
                                       total_slots, emergency_surgery,
                                       num_icu_beds):
            neighbor_sig = tuple(sorted(neighbor.schedule.items()))
            if neighbor_sig not in closed_list:
                neighbor.h_cost = heuristic(neighbor.schedule, surgeries_dict,
                                             emergency_probability)
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                heapq.heappush(open_list, neighbor)

    return None, logs, tree_nodes


def trace_path(goal_state):
    path, current = [], goal_state
    while current:
        path.append(current)
        current = current.parent
    path.reverse()
    return path


# =============================================================
# GANTT CHART
# =============================================================

URGENCY_COLORS = {1: "#4A90D9", 2: "#F5A623", 3: "#D0021B"}
URGENCY_LABELS = {1: "Elective", 2: "Urgent",  3: "Emergency"}


def build_gantt(schedule, surgeries_dict, rooms, total_slots,
                title="Surgery Schedule"):
    slot_labels = {i: f"{8 + i}:00" for i in range(total_slots + 1)}
    fig = go.Figure()

    for surgery_id, (room, start_slot) in schedule.items():
        surgery  = surgeries_dict[surgery_id]
        end_slot = start_slot + surgery.duration
        color    = URGENCY_COLORS[surgery.urgency]
        label    = URGENCY_LABELS[surgery.urgency]
        icu_tag  = " 🏥" if surgery.requires_icu else ""
        equip_str = ", ".join(surgery.equipment) if surgery.equipment else "None"

        fig.add_trace(go.Bar(
            name             = f"{surgery_id} ({label})",
            x                = [surgery.duration],
            y                = [room],
            base             = [start_slot],
            orientation      = "h",
            marker_color     = color,
            marker_line      = dict(color="white", width=1),
            text             = f"{surgery_id}<br>{surgery.name}{icu_tag}",
            textposition     = "inside",
            insidetextanchor = "middle",
            hovertemplate    = (
                f"<b>{surgery.name}</b><br>"
                f"Room: {room}<br>"
                f"Surgeon: {surgery.surgeon}<br>"
                f"Start: {slot_labels.get(start_slot, f'Slot {start_slot}')}<br>"
                f"End:   {slot_labels.get(end_slot,   f'Slot {end_slot}')}<br>"
                f"Duration: {surgery.duration} hr(s)<br>"
                f"Urgency: {label}<br>"
                f"Equipment: {equip_str}<br>"
                f"ICU: {'Yes' if surgery.requires_icu else 'No'}"
                f"<extra></extra>"
            )
        ))

    tick_vals  = list(range(total_slots + 1))
    tick_texts = [slot_labels.get(i, "") for i in tick_vals]

    fig.update_layout(
        title         = dict(text=title, font=dict(size=13, color="#1B3A6B")),
        barmode       = "overlay",
        xaxis         = dict(
            title     = "Time (Hours from 08:00)",
            tickvals  = tick_vals,
            ticktext  = tick_texts,
            range     = [0, total_slots],
            showgrid  = True,
            gridcolor = "#e0e0e0",
            gridwidth = 1,
            dtick     = 1,
        ),
        yaxis         = dict(
            title         = "Operating Room",
            categoryorder = "array",
            categoryarray = rooms,
        ),
        plot_bgcolor  = "white",
        paper_bgcolor = "white",
        height        = max(320, 90 * len(rooms) + 120),
        showlegend    = True,
        legend        = dict(orientation="h", y=-0.25),
        margin        = dict(l=10, r=10, t=45, b=10),
    )
    return fig


# =============================================================
# SESSION STATE INIT
# =============================================================

DEFAULTS = {
    "surgeries_dict"      : {},
    "schedule"            : {},
    "goal_state"          : None,
    "astar_logs"          : [],
    "path_steps"          : [],
    "emergency_ran"       : False,
    "total_cost"          : 0.0,
    "reschedule_count"    : 0,
    "saved_rooms"              : ["Room 1", "Room 2", "Room 3"],
    "saved_slots"              : 16,
    "saved_icu_beds"           : 2,
    "surgery_counter"          : 1,
    "emergency_counter"        : 1,   # tracks E1, E2, E3 ... independently
    "ml_probability"           : None,
    "ml_risk_label"            : None,
    "ml_risk_class"            : None,
    "pre_emergency_schedule"   : {},
    "pre_emergency_sdict"      : {},
    "astar_tree"               : [],   # captured tree nodes for visualisation
}

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =============================================================
# HEADER
# =============================================================

st.markdown("## 🏥 Hospital Surgery Scheduling System")
st.markdown(
    "**AI2002 — Artificial Intelligence | Spring 2026** &nbsp;|&nbsp; "
    "A\\* Search Algorithm &nbsp;+&nbsp; Logistic Regression Emergency Predictor"
)
st.markdown(
    '<span class="constraint-badge">✔ Room Exclusivity</span>'
    '<span class="constraint-badge">✔ Surgeon Uniqueness</span>'
    '<span class="constraint-badge">✔ Working Hours</span>'
    '<span class="constraint-badge">✔ Equipment Conflict</span>'
    '<span class="constraint-badge">✔ ICU Availability</span>',
    unsafe_allow_html=True
)
st.divider()


# =============================================================
# SIDEBAR
# =============================================================

sb = st.sidebar
sb.markdown("## ⚙️ Setup Panel")
sb.divider()

# ── Step 1: Hospital Setup ────────────────────────────────────
sb.markdown("### 🏨 Step 1 — Hospital Setup")

# Always allow changing setup.
# Fix 1 (KeyError guard in is_valid_schedule) prevents crashes if rooms
# are reduced after surgeries are added — the validator returns False safely.
# A soft warning is shown when surgeries already exist so the user knows
# that existing surgeries in removed rooms will fail constraint checks.
_saved_rooms_count = len(st.session_state.saved_rooms) if st.session_state.saved_rooms else 3
_saved_slots       = st.session_state.saved_slots       if st.session_state.saved_slots  else 16
_saved_icu         = st.session_state.saved_icu_beds    if st.session_state.saved_icu_beds else 2

num_rooms   = sb.number_input("Number of Operating Rooms",
                               min_value=1, max_value=6,
                               value=int(_saved_rooms_count), step=1)
total_slots = sb.number_input("Working Hours per Day",
                               min_value=4, max_value=24,
                               value=int(_saved_slots), step=1)
num_icu     = sb.number_input(
    "Number of ICU Beds", min_value=1, max_value=10,
    value=int(_saved_icu), step=1,
    help="Max number of ICU-requiring surgeries allowed to run simultaneously"
)

# Soft warning only — does not block the user
if st.session_state.surgeries_dict:
    _prev_rooms = st.session_state.saved_rooms
    _new_rooms  = [f"Room {i+1}" for i in range(int(num_rooms))]
    _removed    = [r for r in _prev_rooms if r not in _new_rooms]
    if _removed or int(total_slots) < int(_saved_slots) or int(num_icu) < int(_saved_icu):
        sb.warning(
            "⚠️ Setup changed after surgeries were added. "
            "Any existing surgery that no longer fits the new configuration "
            "will be rejected by the constraint validator. "
            "Use **Reset** to start fresh if needed."
        )

rooms       = [f"Room {i+1}" for i in range(int(num_rooms))]
slot_labels = {i: f"{8+i}:00" for i in range(int(total_slots) + 1)}

sb.caption(f"Rooms: {', '.join(rooms)}")
sb.caption(f"Hours: 08:00 — {8 + int(total_slots)}:00  ({int(total_slots)} slots)")
sb.caption(f"ICU Beds available: {int(num_icu)}")
sb.divider()

# ── Step 2: Add Elective Surgeries ───────────────────────────
sb.markdown("### 🔵 Step 2 — Add Elective Surgeries")

with sb.form("add_surgery_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        s_name    = st.text_input("Surgery Name",  placeholder="e.g. Appendectomy")
        s_surgeon = st.text_input("Surgeon",        placeholder="e.g. Dr. Ahmed")
        s_room    = st.selectbox("Room",            rooms)
    with col2:
        s_duration = st.slider("Duration (hrs)", 1, 8, 2)
        s_urgency  = st.selectbox("Urgency",    ["Elective", "Urgent"])
        s_slot     = st.number_input(
            "Start Slot (0 = 08:00)",
            min_value=0, max_value=int(total_slots) - 1,
            value=0, step=1,
            help="Each slot = 1 hour. 0=08:00, 1=09:00 ..."
        )

    s_equip = st.multiselect(
        "Equipment Needed",
        ["ventilator", "scalpel", "drill", "xray", "laser",
         "camera", "ecg_monitor", "anesthesia_machine", "cauterizer"]
    )
    s_icu   = st.checkbox("Requires ICU Bed", value=False)
    add_btn = st.form_submit_button("➕ Add Surgery", type="primary")

if add_btn:
    if not s_name.strip():
        sb.error("Surgery name is required.")
    elif not s_surgeon.strip():
        sb.error("Surgeon name is required.")
    else:
        sid     = f"S{st.session_state.surgery_counter}"
        urgency = 1 if s_urgency == "Elective" else 2

        # ── Fix 2: Proactive overflow check before full validation ────────────
        # Catches start_slot + duration > total_slots early and gives a clear,
        # specific message instead of the generic constraint-violation error.
        _end_slot = int(s_slot) + int(s_duration)
        if _end_slot > int(total_slots):
            sb.error(
                f"⚠️ Working-hours overflow: surgery starts at slot {int(s_slot)} "
                f"({slot_labels.get(int(s_slot), '')}) and lasts {int(s_duration)} hr(s), "
                f"finishing at slot {_end_slot} which exceeds the "
                f"{int(total_slots)}-hour day limit "
                f"(latest finish: {slot_labels.get(int(total_slots), '')}). "
                "Reduce duration or choose an earlier start slot."
            )
        else:
            candidate_surgery  = Surgery(
                surgery_id=sid, name=s_name.strip(), duration=int(s_duration),
                urgency=urgency, surgeon=s_surgeon.strip(),
                equipment=s_equip, requires_icu=s_icu
            )
            candidate_schedule = {**st.session_state.schedule,       sid: (s_room, int(s_slot))}
            candidate_dict     = {**st.session_state.surgeries_dict, sid: candidate_surgery}

            if is_valid_schedule(candidate_schedule, candidate_dict,
                                 rooms, int(total_slots), int(num_icu)):
                st.session_state.surgeries_dict[sid] = candidate_surgery
                st.session_state.schedule[sid]        = (s_room, int(s_slot))
                st.session_state.saved_rooms          = rooms
                st.session_state.saved_slots          = int(total_slots)
                st.session_state.saved_icu_beds       = int(num_icu)
                st.session_state.surgery_counter     += 1
                st.session_state.ml_probability       = None
                sb.success(f"✅ {s_name.strip()} added as {sid}")
            else:
                # ── Fix 4: Specific constraint failure messages ───────────────
                # Identify exactly which constraint was violated instead of
                # showing one generic error for all five constraints.
                _sched_existing = dict(st.session_state.schedule)
                _dict_existing  = dict(st.session_state.surgeries_dict)

                # Check room conflict
                _room_conflict = False
                for _oid, (_or, _os) in _sched_existing.items():
                    _o = _dict_existing.get(_oid)
                    if _o and _or == s_room:
                        _overlap = range(
                            max(int(s_slot), _os),
                            min(int(s_slot) + int(s_duration), _os + _o.duration)
                        )
                        if len(_overlap) > 0:
                            _room_conflict = True
                            break

                # Check surgeon conflict
                _surgeon_conflict = False
                for _oid, (_or, _os) in _sched_existing.items():
                    _o = _dict_existing.get(_oid)
                    if _o and _o.surgeon.strip().lower() == s_surgeon.strip().lower():
                        _overlap = range(
                            max(int(s_slot), _os),
                            min(int(s_slot) + int(s_duration), _os + _o.duration)
                        )
                        if len(_overlap) > 0:
                            _surgeon_conflict = True
                            break

                # Check equipment conflict
                _equip_conflict = False
                if s_equip:
                    for _oid, (_or, _os) in _sched_existing.items():
                        _o = _dict_existing.get(_oid)
                        if _o and _or != s_room:
                            _shared = set(s_equip) & set(_o.equipment)
                            if _shared:
                                _overlap = range(
                                    max(int(s_slot), _os),
                                    min(int(s_slot) + int(s_duration), _os + _o.duration)
                                )
                                if len(_overlap) > 0:
                                    _equip_conflict = True
                                    break

                # Check ICU conflict
                _icu_conflict = False
                if s_icu:
                    _icu_running = sum(
                        1 for _oid, (_or, _os) in _sched_existing.items()
                        if _dict_existing.get(_oid) and
                           _dict_existing[_oid].requires_icu and
                           len(range(max(int(s_slot), _os),
                                    min(int(s_slot) + int(s_duration),
                                        _os + _dict_existing[_oid].duration))) > 0
                    )
                    if _icu_running >= int(num_icu):
                        _icu_conflict = True

                if _room_conflict:
                    sb.error(
                        f"❌ Room conflict: {s_room} is already occupied during "
                        f"{slot_labels.get(int(s_slot), f'slot {s_slot}')} – "
                        f"{slot_labels.get(_end_slot, f'slot {_end_slot}')}. "
                        "Choose a different room or time slot."
                    )
                elif _surgeon_conflict:
                    sb.error(
                        f"❌ Surgeon conflict: {s_surgeon.strip()} is already "
                        "operating during this time window. "
                        "Assign a different surgeon or adjust the time slot."
                    )
                elif _equip_conflict:
                    sb.error(
                        f"❌ Equipment conflict: one or more items in "
                        f"{s_equip} are already in use in a different room "
                        "during this time window."
                    )
                elif _icu_conflict:
                    sb.error(
                        f"❌ ICU unavailable: all {int(num_icu)} ICU bed(s) are "
                        "occupied during this time window. "
                        "Add more ICU beds or choose a non-overlapping time slot."
                    )
                else:
                    sb.error(
                        "⚠️ Constraint violation detected. "
                        "Check working hours, room assignments, and equipment."
                    )

if st.session_state.surgeries_dict:
    sb.markdown("**Surgeries added:**")
    for sid, surg in st.session_state.surgeries_dict.items():
        room, slot = st.session_state.schedule.get(sid, ("?", "?"))
        icu_tag    = " 🏥" if surg.requires_icu else ""
        sb.caption(
            f"• {sid}: {surg.name}{icu_tag} | {room} | "
            f"{slot_labels.get(slot, slot)} | {surg.surgeon}"
        )

sb.divider()
# ── Step 3: Inject Emergency ──────────────────────────────────
sb.markdown("### 🚨 Step 3 — Inject Emergency Surgery")

with sb.form("emergency_form"):
    col1, col2 = st.columns(2)
    with col1:
        em_name    = st.text_input("Emergency Name",
                                    placeholder="e.g. Trauma Surgery")
        em_surgeon = st.text_input("Surgeon",
                                    placeholder="e.g. Dr. Sara")
    with col2:
        em_duration = st.slider("Duration (hrs)", 1, 8, 2)

    em_equip = st.multiselect(
        "Equipment Needed",
        ["ventilator", "scalpel", "drill", "xray", "laser",
         "camera", "ecg_monitor", "anesthesia_machine", "cauterizer"]
    )
    em_icu   = st.checkbox("Requires ICU Bed", value=True)
    run_btn  = st.form_submit_button("▶ Run A* Rescheduling", type="primary")

sb.divider()

# ── Autofill Test Data ────────────────────────────────────────
sb.markdown("### 🧪 Load Conflict Test Data")
sb.caption("5 surgeries with equipment + ICU conflicts — designed to push A* hard.")

if sb.button("⚡ Autofill Conflict Scenario"):
    st.session_state.surgeries_dict = {
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
    st.session_state.schedule = {
        "S1": ("Room 1", 0),
        "S2": ("Room 2", 0),
        "S3": ("Room 3", 0),
        "S4": ("Room 1", 5),
        "S5": ("Room 2", 6),
    }
    st.session_state.saved_rooms     = ["Room 1", "Room 2", "Room 3"]
    st.session_state.saved_slots     = 16
    st.session_state.saved_icu_beds  = 2
    st.session_state.surgery_counter = 6
    st.session_state.emergency_counter = 1
    st.session_state.goal_state      = None
    st.session_state.astar_logs      = []
    st.session_state.path_steps      = []
    st.session_state.emergency_ran   = False
    st.session_state.total_cost      = 0.0
    st.session_state.reschedule_count= 0
    st.session_state.ml_probability  = None
    sb.success(
        "✅ Loaded! Inject: Trauma Surgery | Dr. Zaid | 2 hrs | "
        "ventilator | ICU required"
    )
    st.rerun()

sb.divider()

# ── Reset ─────────────────────────────────────────────────────
if sb.button("🔄 Clear All & Reset"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = copy.deepcopy(v)
    st.rerun()


# =============================================================
# ML PREDICTION PANEL
# =============================================================

if st.session_state.schedule:
    prob, feat_df, risk_label, risk_class = predict_emergency_probability(
        st.session_state.schedule,
        st.session_state.surgeries_dict,
        st.session_state.saved_slots
    )
    st.session_state.ml_probability = prob
    st.session_state.ml_risk_label  = risk_label
    st.session_state.ml_risk_class  = risk_class

    st.markdown(
        '<p class="section-title">🤖 ML Emergency Prediction (Logistic Regression)</p>',
        unsafe_allow_html=True
    )

    model_note = (
        "✅ Trained Logistic Regression model loaded from emergency_model.pkl"
        if ml_model is not None
        else "⚠️ Model file not found — run train_model.py first."
    )

    st.markdown(
        f'<div class="{risk_class}">'
        f'<b>{risk_label}</b> &nbsp;|&nbsp; '
        f'Probability of another emergency in next 6 hours: <b>{prob:.1%}</b><br>'
        f'<small style="color:#555">{model_note}</small>'
        f'</div>',
        unsafe_allow_html=True
    )

    with st.expander("🔬 View ML features used for this prediction"):
        st.dataframe(
            feat_df.rename(columns={
                "hour_of_day"          : "Hour of Day",
                "is_weekend"           : "Weekend (1=Yes)",
                "num_surgeries"        : "Surgeries Scheduled",
                "num_urgent"           : "Urgent Surgeries",
                "room_occupancy_pct"   : "Room Occupancy",
                "avg_surgery_duration" : "Avg Duration (hrs)",
            }),
            use_container_width=True, hide_index=True
        )
        st.caption(
            "Features extracted automatically from the current schedule "
            "and fed into the trained Logistic Regression model."
        )

    st.markdown("<br>", unsafe_allow_html=True)


# =============================================================
# CONSTRAINT STATUS PANEL
# =============================================================

if st.session_state.schedule:
    st.markdown(
        '<p class="section-title">🛡️ Constraint Validation Status</p>',
        unsafe_allow_html=True
    )

    valid = is_valid_schedule(
        st.session_state.schedule,
        st.session_state.surgeries_dict,
        st.session_state.saved_rooms,
        st.session_state.saved_slots,
        st.session_state.saved_icu_beds
    )
    icu_using = sum(
        1 for sid, s in st.session_state.surgeries_dict.items()
        if s.requires_icu and sid in st.session_state.schedule
    )

    ok  = "background:#d4edda;border-radius:6px;padding:8px;text-align:center;font-size:13px"
    cv1, cv2, cv3, cv4, cv5 = st.columns(5)
    with cv1:
        st.markdown(f'<div style="{ok}">✔ Room Exclusivity<br><small>Active</small></div>',
                    unsafe_allow_html=True)
    with cv2:
        st.markdown(f'<div style="{ok}">✔ Surgeon Uniqueness<br><small>Active</small></div>',
                    unsafe_allow_html=True)
    with cv3:
        st.markdown(
            f'<div style="{ok}">✔ Working Hours<br>'
            f'<small>{int(st.session_state.saved_slots)} hrs/day</small></div>',
            unsafe_allow_html=True)
    with cv4:
        st.markdown(f'<div style="{ok}">✔ Equipment Conflict<br><small>Active</small></div>',
                    unsafe_allow_html=True)
    with cv5:
        st.markdown(
            f'<div style="{ok}">✔ ICU Availability<br>'
            f'<small>{icu_using} using / {st.session_state.saved_icu_beds} beds</small></div>',
            unsafe_allow_html=True)

    color   = "#d4edda" if valid else "#fdecea"
    overall = "✅ All constraints satisfied" if valid else "❌ Constraint violation detected"
    st.markdown(
        f'<div style="background:{color};border-radius:6px;padding:10px;'
        f'margin-top:10px;text-align:center;font-weight:bold">{overall}</div>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)


# =============================================================
# RUN A*
# =============================================================

if run_btn:
    if not st.session_state.schedule:
        st.error("⚠️ Add at least one elective surgery first (Step 2).")
    elif not em_name.strip():
        st.error("⚠️ Emergency surgery name is required.")
    elif not em_surgeon.strip():
        st.error("⚠️ Emergency surgeon name is required.")
    else:
        # ── Bug fix 1: unique ID per emergency (E1, E2, E3 ...) ──
        em_id     = f"E{st.session_state.emergency_counter}"

        emergency = Surgery(
            surgery_id   = em_id,
            name         = em_name.strip(),
            duration     = int(em_duration),
            urgency      = 3,
            surgeon      = em_surgeon.strip(),
            equipment    = em_equip,
            requires_icu = em_icu
        )

        # current schedule already includes all previously placed emergencies
        # (stored in st.session_state.schedule after each successful A* run)
        surgeries_with_em        = copy.deepcopy(st.session_state.surgeries_dict)
        surgeries_with_em[em_id] = emergency
        used_rooms               = st.session_state.saved_rooms
        used_slots               = st.session_state.saved_slots
        used_icu                 = st.session_state.saved_icu_beds

        em_prob = (st.session_state.ml_probability
                   if st.session_state.ml_probability is not None else 0.5)

        with st.spinner(f"⏳ A* placing {em_id} — checking all 5 constraints..."):
            goal, logs, tree_nodes = astar_search(
                initial_schedule      = st.session_state.schedule,
                surgeries_dict        = surgeries_with_em,
                rooms                 = used_rooms,
                total_slots           = used_slots,
                emergency_surgery     = emergency,
                emergency_probability = em_prob,
                num_icu_beds          = used_icu
            )

        if goal:
            st.session_state.pre_emergency_schedule = dict(st.session_state.schedule)
            st.session_state.pre_emergency_sdict    = copy.deepcopy(st.session_state.surgeries_dict)

            st.session_state.schedule         = goal.schedule
            st.session_state.surgeries_dict   = surgeries_with_em
            st.session_state.goal_state       = goal
            st.session_state.astar_logs       = logs
            st.session_state.astar_tree       = tree_nodes      # ← save tree
            st.session_state.path_steps       = trace_path(goal)
            st.session_state.emergency_ran    = True
            st.session_state.total_cost      += goal.g_cost
            st.session_state.reschedule_count += 1
            st.session_state.emergency_counter += 1
        else:
            st.error(
                f"❌ A* could not place {em_id}. "
                "Try more rooms, more ICU beds, or more working hours."
            )


# =============================================================
# METRICS ROW
# =============================================================

st.markdown('<p class="section-title">📊 Live Metrics</p>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{len(st.session_state.schedule)}</div>
        <div class="metric-label">Surgeries Scheduled</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{st.session_state.reschedule_count}</div>
        <div class="metric-label">Rescheduling Events</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{round(st.session_state.total_cost, 2)}</div>
        <div class="metric-label">Total Rescheduling Cost</div>
    </div>""", unsafe_allow_html=True)

with c4:
    prob_display = (f"{st.session_state.ml_probability:.1%}"
                    if st.session_state.ml_probability is not None else "—")
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="font-size:20px">{prob_display}</div>
        <div class="metric-label">ML Emergency Probability</div>
    </div>""", unsafe_allow_html=True)

with c5:
    # Compute PEAK concurrent ICU usage — i.e. the maximum number of
    # ICU-requiring surgeries running at the same time in any single slot.
    # This is the correct figure to compare against num_icu_beds.
    # (Total ICU surgeries across all time slots can legitimately exceed
    #  num_icu_beds as long as they don't overlap.)
    _icu_slots: dict = {}
    for _sid in st.session_state.schedule:
        _s = st.session_state.surgeries_dict.get(_sid)
        if _s and _s.requires_icu:
            _sr, _ss = st.session_state.schedule[_sid]
            for _slot in range(_ss, _ss + _s.duration):
                _icu_slots[_slot] = _icu_slots.get(_slot, 0) + 1
    peak_icu = max(_icu_slots.values()) if _icu_slots else 0
    icu_beds = st.session_state.saved_icu_beds
    icu_color = "color:#D0021B;" if peak_icu > icu_beds else ""
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="{icu_color}">{peak_icu} / {icu_beds}</div>
        <div class="metric-label">Peak Concurrent ICU Beds</div>
    </div>""", unsafe_allow_html=True)

with c6:
    status = "🔴 Emergency Active" if st.session_state.emergency_ran else "✅ Stable"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="font-size:16px">{status}</div>
        <div class="metric-label">System Status</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================
# SCHEDULE TABLE
# =============================================================

if st.session_state.schedule:
    st.markdown('<p class="section-title">🗂️ Schedule Table</p>',
                unsafe_allow_html=True)

    display_schedule = (
        st.session_state.goal_state.schedule
        if st.session_state.goal_state
        else st.session_state.schedule
    )

    used_slot_labels = {i: f"{8+i}:00"
                        for i in range(st.session_state.saved_slots + 1)}
    rows = []

    for sid, (room, slot) in sorted(display_schedule.items()):
        s = st.session_state.surgeries_dict.get(sid)
        if s:
            end = slot + s.duration
            rows.append({
                "ID"        : sid,
                "Surgery"   : s.name,
                "Room"      : room,
                "Start"     : used_slot_labels.get(slot, f"Slot {slot}"),
                "End"       : used_slot_labels.get(end,  f"Slot {end}"),
                "Duration"  : f"{s.duration} hr(s)",
                "Surgeon"   : s.surgeon,
                "Urgency"   : URGENCY_LABELS[s.urgency],
                "Equipment" : ", ".join(s.equipment) if s.equipment else "—",
                "ICU"       : "Yes" if s.requires_icu else "No",
            })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("<br>", unsafe_allow_html=True)


# =============================================================
# GANTT CHART
# =============================================================

if st.session_state.schedule:
    st.markdown('<p class="section-title">📅 Gantt Chart</p>',
                unsafe_allow_html=True)

    used_rooms = st.session_state.saved_rooms
    used_slots = st.session_state.saved_slots

    if st.session_state.emergency_ran and st.session_state.goal_state:
        st.markdown("""<div class="emergency-box">
            🚨 <b>Emergency received.</b> A* has computed the optimal rescheduling
            satisfying all 5 hard constraints. Compare below.
        </div>""", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔴 After Rescheduling (A* Output)",
                               "🔵 Original Schedule"])
        with tab1:
            st.plotly_chart(
                build_gantt(st.session_state.goal_state.schedule,
                            st.session_state.surgeries_dict,
                            used_rooms, used_slots,
                            "Rescheduled Plan — A* Output"),
                use_container_width=True
            )
        with tab2:
            # Use the saved pre-emergency snapshot — NOT st.session_state.schedule
            # which has already been overwritten with the A* result.
            _orig_schedule = (
                st.session_state.pre_emergency_schedule
                if st.session_state.emergency_ran
                else st.session_state.schedule
            )
            _orig_sdict = (
                st.session_state.pre_emergency_sdict
                if st.session_state.emergency_ran
                else st.session_state.surgeries_dict
            )
            st.plotly_chart(
                build_gantt(_orig_schedule,
                            _orig_sdict,
                            used_rooms, used_slots,
                            "Original Schedule (Before Emergency)"),
                use_container_width=True
            )
    else:
        st.markdown("""<div class="success-box">
            ✅ <b>No emergency active.</b> Schedule is stable.
            Use Step 3 in the sidebar to inject an emergency.
        </div>""", unsafe_allow_html=True)

        st.plotly_chart(
            build_gantt(st.session_state.schedule,
                        st.session_state.surgeries_dict,
                        used_rooms, used_slots,
                        "Current Surgery Schedule"),
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)


 # =============================================================
# A* TRANSPARENCY LOG
# =============================================================
 
if st.session_state.astar_logs:
    st.markdown('<p class="section-title">🔍 A* Search — Transparency Log</p>',
                unsafe_allow_html=True)
 
    with st.expander("View A* iteration log  (g, h, f values per step)"):
        st.dataframe(
            pd.DataFrame(st.session_state.astar_logs),
            use_container_width=True, hide_index=True
        )
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    st.markdown('<p class="section-title">🪜 Rescheduling Path — Step by Step</p>',
                unsafe_allow_html=True)
 
    with st.expander("View each action A* took to reach the final schedule"):
        for i, state in enumerate(st.session_state.path_steps):
            if state.action:
                st.markdown(f"""
                <div class="log-entry">
                    <b>Step {i}:</b> {state.action} &nbsp;|&nbsp;
                    g = {state.g_cost:.2f} &nbsp;&nbsp;
                    h = {state.h_cost:.2f} &nbsp;&nbsp;
                    f = {state.f_cost:.2f}
                </div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)

# =============================================================
# FOOTER
# =============================================================

st.divider()
st.markdown(
    "<center><small>"
    "Qadeer Raza (23i-3059) | Abdullah Tariq (23i-3011) | Malik Usman (23i-3020) "
    "| FAST NUCES Islamabad | AI-Driven Hospital Surgery Scheduling System"
    "</small></center>",
    unsafe_allow_html=True
)