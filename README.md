# 🏥 AI-Driven Hospital Surgery Scheduling System

**Group:[Qadeer Raza (23i-2059)], [Abdullah Tariq (23i-3011)], [Malik Usman (23i-3020)]**



## Project Overview

This project implements an AI-driven hospital surgery scheduling system that manages elective surgery planning and dynamically reschedules when emergency cases arrive. The core of Assignment 1 Part B is the **A\* Search Algorithm**, which finds the optimal rescheduling path with minimum cost and disruption when an emergency surgery is injected into an existing schedule. The entire system runs through an interactive **Streamlit web dashboard** — no console interaction needed.



## Project Structure


Assignment1_Qadeer Raza_Abdullah Tariq_Malik Usman/
│
├── app.py
├── train_model.py
├── search_Algorithm.py
├── emergency_model.pkl
├── i233059_i233011_i233020_A02.py
├── requirements.txt
└── README.md     

> All A\* search logic is embedded inside `app.py` so only one file is needed to run the full system.


## Dependencies

Make sure you have **Python 3.8 or above** installed. Then install the required libraries using:


pip install streamlit plotly pandas


To verify everything is installed correctly:

 
python --version
streamlit --version
 

 

##  Execution Instructions

Run the following command in the terminal from the folder where `app.py` is located:

streamlit run app.py

The dashboard will open automatically in your browser at:

http://localhost:8501


##  How to Use the Dashboard

The sidebar has **3 steps** to follow in order:


**Step 1 — Hospital Setup**

Set the number of operating rooms (1–6) and working hours per day (4–24). The system adapts everything to whatever you configure here.
<img width="291" height="575" alt="image" src="https://github.com/user-attachments/assets/1b483573-401a-404b-b78f-b6fc46268ee6" />


**Step 2 — Add Elective Surgeries**

Fill in the form for each surgery:
- Surgery name and surgeon name
- Assign a room and start slot
- Set duration and urgency level (Elective or Urgent)
- Select required equipment

Click **Add Surgery**. Each surgery gets a unique ID (S1, S2, S3...) and appears in a running list in the sidebar. Repeat this for as many surgeries as needed before injecting an emergency.


**Step 3 — Inject Emergency Surgery**

Enter the emergency surgery details:
- Name, surgeon, duration, and equipment
- Set the **ML Emergency Probability** slider — this simulates the output of the machine learning model, representing the probability of another emergency arriving in the next 6 hours

Click **Run A\* Rescheduling**. The A\* algorithm will compute the optimal rescheduling and the dashboard updates instantly.

---

**Reset**

Click **Clear All & Reset** at the bottom of the sidebar to wipe everything and start fresh.


## 📥 Sample Input

### Hospital Setup
- Operating Rooms: 3
- Working Hours: 12 (08:00 — 20:00)

### Elective Surgeries

| ID | Surgery | Room | Start | Duration | Surgeon | Urgency |
|----|---------|------|-------|----------|---------|---------|
| S1 | Appendectomy | Room 1 | 08:00 | 2 hrs | Dr. Ahmed | Elective |
| S2 | Hip Replacement | Room 2 | 08:00 | 3 hrs | Dr. Sara | Elective |
| S3 | Heart Bypass | Room 1 | 10:00 | 4 hrs | Dr. Khan | Urgent |
| S4 | Cataract Surgery | Room 2 | 11:00 | 1 hr | Dr. Ahmed | Elective |

### Emergency Surgery

| Field | Value |
|-------|-------|
| Name | Emergency Trauma |
| Duration | 2 hrs |
| Surgeon | Dr. Sara |
| Equipment | ventilator |
| ML Probability | 0.75 |

---

## 📤 Sample Output (UI)

After clicking **Run A\* Rescheduling**, the dashboard shows:

**Metrics Row**
- Surgeries Scheduled: 5 (4 elective + 1 emergency)
- Rescheduling Events: 1
- Total Rescheduling Cost: 0.00
- System Status: 🔴 Emergency Active

**Gantt Chart — Two Tabs**
- *After Rescheduling (A\* Output)* — shows the updated plan with Emergency Trauma placed in Room 3 at 08:00 without disturbing any existing surgery
- *Original Schedule* — shows the plan before the emergency arrived

**Schedule Table**

| ID | Surgery | Room | Start | End | Duration | Surgeon | Urgency |
|----|---------|------|-------|-----|----------|---------|---------|
| S1 | Appendectomy | Room 1 | 08:00 | 10:00 | 2 hr(s) | Dr. Ahmed | Elective |
| S2 | Hip Replacement | Room 2 | 08:00 | 11:00 | 3 hr(s) | Dr. Sara | Elective |
| S3 | Heart Bypass | Room 1 | 10:00 | 14:00 | 4 hr(s) | Dr. Khan | Urgent |
| S4 | Cataract Surgery | Room 2 | 11:00 | 12:00 | 1 hr(s) | Dr. Ahmed | Elective |
| E1 | Emergency Trauma | Room 3 | 08:00 | 10:00 | 2 hr(s) | Dr. Sara | Emergency |

**A\* Transparency Log**

| Iteration | Action | g(n) | h(n) | f(n) |
|-----------|--------|------|------|------|
| 1 | Start state | 0.00 | 3.75 | 3.75 |
| 2 | Insert E1 → Room 3 at slot 0 | 0.00 | 0.00 | 0.00 |

**Step-by-Step Rescheduling Path**
Step 1: Insert E1 → Room 3 at slot 0  |  g = 0.00  h = 0.00  f = 0.00


## 🧠 Technical Overview

### Problem Representation

The scheduling problem is modeled as a **graph search problem** where each node in the search graph is a complete hospital schedule configuration — a dictionary mapping each surgery ID to a (room, time slot) pair. Each edge between nodes represents one rescheduling action: either inserting the emergency surgery into a free slot, or delaying an existing surgery by 1 or 2 slots to create space.

### A\* Search

A\* is implemented using Python's `heapq` module as a min-heap priority queue. The algorithm always expands the node with the lowest f(n) = g(n) + h(n) first, guaranteeing that the first valid rescheduled schedule it reaches is the optimal one.

**g(n) — Path Cost**
Accumulates the cost of every rescheduling action taken so far. Delay cost is calculated as number of slots delayed multiplied by an urgency multiplier (1.0 for elective, 2.0 for urgent, 4.0 for emergency), plus a 0.5 penalty for changing rooms.

**h(n) — Heuristic**
Estimates future cost based on how late high-urgency surgeries are scheduled, combined with a weighted ML emergency probability score. The heuristic is admissible — it never overestimates actual future cost — which guarantees A\* finds the optimal rescheduling path.

### Constraint Validation

Before any state is accepted into the search, three hard constraints are checked:
- No two surgeries overlap in the same room at the same time
- No surgeon is assigned to two rooms simultaneously  
- Every surgery must finish within the total working hours of the day

### Key Components

**Surgery class** — stores all attributes: ID, name, duration, urgency, surgeon, and equipment.

**ScheduleState class** — represents one search node. Stores the current schedule, g cost, h cost, f cost, parent state, and the action that produced it.

**is_valid_schedule()** — constraint checker called before accepting any neighbor state.

**heuristic()** — computes h(n) using urgency-based delay estimation and ML probability score.

**get_neighbors()** — generates all valid next states by trying emergency insertion and surgery delays across all rooms and slots.

**astar_search()** — main A\* loop. Returns the goal state and a full log of every iteration for the transparency panel.


## 📊 Algorithm Properties

| Property         | Value                                 |
|------------------|---------------------------------------|
| State Space      | (R × T)^S — exponential               |
| Branching Factor | Up to R × T per node                  |
| Time Complexity  | O(b^d) reduced by heuristic           |
| Space Complexity | O(b^d) — open list nodes stored       |
| Completeness     | Yes — finds solution if one exists    |
| Optimality       | Yes — guaranteed with admissible h(n) |

Where R = rooms, T = time slots, S = surgeries, b = branching factor, d = solution depth.

-
## Notes

- The ML Emergency Probability slider simulates the output of a logistic regression model that will be fully implemented in the final project submission (deadline: 3rd May 2026)
- All input is entered dynamically through the UI — there is no hardcoded data
- The system supports up to 6 operating rooms and 24 working hours
- A\* delay actions are limited to 1 or 2 slots to keep the branching factor manageable for demonstration
