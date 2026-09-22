"""
traffic_grid_env.py
====================
A single, algorithm-agnostic Gymnasium environment for multi-junction SUMO
traffic-signal control, built to be shared across DQN, PPO, actor-critic and
critic-actor training runs (mirrors the role common/env.py played for the
2-junction project, generalized to an arbitrary grid).

Design goals
------------
1. ONE environment, many algorithms. The env exposes:
   - a joint MultiDiscrete action space (one binary "switch vote" per
     signalized junction) that PPO / vanilla actor-critic / critic-actor can
     consume directly (SB3's MultiDiscrete support, or a factored
     per-junction softmax head in a custom actor).
   - DQN does NOT need a flattened Discrete(2**N) action space (that blows
     up combinatorially for N=16). Instead, apply a single SHARED Q-network
     independently to every junction's local observation (parameter-sharing
     / independent-DQN, the standard way this is scaled in the traffic-MARL
     literature) and assemble the N resulting binary actions into the
     MultiDiscrete vector before calling env.step(). See train_dqn_stub.py.
2. Graph-native observations for GNN function approximators. Every reset()
   / step() returns a Dict observation containing:
       - "node_features": (N, F) float32   per-junction local features
       - "action_mask":   (N,)   float32   1 if that junction may legally
                                            switch this step, else 0
   The static graph connectivity (edge_index, built once from the actual
   .net.xml junction adjacency via sumolib) lives on `env.edge_index`
   (shape (2, E), int64) and is passed once to a GCN/GAT feature extractor
   -- it does not need to be re-sent every step since the grid topology is
   fixed within an episode.
   For plain-MLP baselines (tabular-ish DQN, vanilla PPO/AC without a GNN)
   wrap the env with `FlattenGraphObs` below to get a flat Box(N*F,).
3. Centralized-critic friendly. info["global_state"] is the flattened
   (N*F,) vector and info["agent_rewards"] is a per-junction reward array,
   so a centralized critic-actor scheme can bootstrap off the global reward
   returned by step() while a decentralized/factored actor can bootstrap
   off info["agent_rewards"] if you split credit per-junction instead.
4. Congestion-aware routing, opt-in via enable_routing=True. At every
   multi-way "decision edge" (an edge whose downstream junction offers more
   than one outgoing edge), the joint action additionally carries a
   discrete choice of which candidate downstream edge to route newly
   arriving vehicles toward (up to `routing_k` alternatives, ranked by free-
   flow travel time). This is intentionally a discrete, SB3-MultiDiscrete-
   compatible design rather than continuous edge-weight tuning, since a
   mixed Discrete+Box action space is not natively supported by SB3 and a
   pure Box space would be awkward for the tabular / DQN baselines this
   project also compares against. Treat this as a first, extensible pass --
   the exact routing action design is expected to be refined with Prashansa.

Requires: gymnasium, numpy, traci, sumolib (SUMO_HOME must be set / sumo on
PATH). Tested against the uploaded grid4x4_net.xml / routes_rou.xml.
"""

from __future__ import annotations

import os
import sys
import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)

import sumolib  # noqa: E402
import traci  # noqa: E402


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class GridEnvConfig:
    net_file: str
    route_file: str
    use_gui: bool = False
    sumo_binary: Optional[str] = None          # override to a specific sumo/sumo-gui path
    episode_seconds: int = 3600
    step_length: float = 1.0                   # seconds of SUMO sim time per traci.simulationStep()
    decision_interval: int = 5                 # seconds of sim time per RL step (action repeat)
    min_green: int = 10                        # seconds before a switch vote is honored
    yellow_time: int = 4                       # must match the yellow phases baked into the .net.xml
    max_green: int = 90                        # used only to normalize the "elapsed phase time" feature
    switch_penalty: float = 0.3                # discourages needless flip-flopping
    wasted_vote_penalty: float = 0.03           # small penalty for voting switch while ineligible
    enable_routing: bool = False
    routing_k: int = 3                         # candidate downstream edges considered per decision edge
    routing_interval: int = 10                 # seconds between rerouting passes
    seed: int = 0


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

class TrafficGridEnv(gym.Env):
    """Gymnasium environment wrapping a SUMO traffic grid via TraCI.

    Action space:  MultiDiscrete([2] * N_junctions [+ [routing_k] * N_decision_edges])
        First N_junctions entries: 1 = "vote to switch this junction's phase
        this step" (honored only if the junction is past min_green and not
        currently yellow), 0 = "hold". Any trailing entries (only present if
        enable_routing=True) select, per decision edge, which of that edge's
        top-`routing_k` downstream edges newly-arriving vehicles should be
        biased toward.

    Observation space: Dict(
        node_features: Box(N, F) float32,
        action_mask:   Box(N,)   float32   (1 = legal to switch now)
    )
    Static graph connectivity is available as env.edge_index (2, E) int64
    and does not change within/across episodes for a fixed network file.
    """

    metadata = {"render_modes": []}

    # Local per-junction feature layout (F = 11):
    #   [0:4]  normalized queue length,  approach order (N, E, S, W)
    #   [4:8]  normalized waiting time,  approach order (N, E, S, W)
    #   [8]    normalized current phase index
    #   [9]    is_yellow flag
    #   [10]   normalized elapsed time in current phase
    NUM_FEATURES = 11
    APPROACH_ORDER = ("N", "E", "S", "W")
    QUEUE_NORM = 20.0
    WAIT_NORM = 300.0

    def __init__(self, config: GridEnvConfig):
        super().__init__()
        self.cfg = config
        self._sumo_started = False
        self._episode_step = 0
        self._episode_arrived = 0

        # ---- Static topology discovery (no TraCI connection needed yet) ----
        net = sumolib.net.readNet(self.cfg.net_file)
        tl_nodes = sorted((n.getID() for n in net.getNodes() if n.getType() == "traffic_light"))
        if not tl_nodes:
            raise ValueError("No traffic_light junctions found in net file.")
        self.tl_ids = tl_nodes
        self.n_agents = len(self.tl_ids)
        self._tl_index = {tid: i for i, tid in enumerate(self.tl_ids)}

        self._approach_lanes = {}   # tl_id -> {approach: [lane_ids]}
        self._approach_edges = {}   # tl_id -> {approach: edge_id}  (for waiting-time queries)
        adjacency = set()
        for tid in self.tl_ids:
            node = net.getNode(tid)
            cx, cy = node.getCoord()
            approaches = {}
            approach_edge = {}
            for edge in node.getIncoming():
                fx, fy = edge.getFromNode().getCoord()
                dx, dy = cx - fx, cy - fy
                if abs(dx) > abs(dy):
                    direction = "E" if dx > 0 else "W"
                else:
                    direction = "N" if dy > 0 else "S"
                approaches.setdefault(direction, []).extend(
                    f"{edge.getID()}_{i}" for i in range(edge.getLaneNumber())
                )
                approach_edge[direction] = edge.getID()
                # grid adjacency: connect two TL junctions that share a direct edge
                other = edge.getFromNode().getID()
                if other in self._tl_index:
                    adjacency.add((self._tl_index[other], self._tl_index[tid]))
                    adjacency.add((self._tl_index[tid], self._tl_index[other]))
            self._approach_lanes[tid] = approaches
            self._approach_edges[tid] = approach_edge

        if adjacency:
            self.edge_index = np.array(sorted(adjacency), dtype=np.int64).T  # (2, E)
        else:
            self.edge_index = np.zeros((2, 0), dtype=np.int64)

        # ---- Routing decision points (optional) ----
        self._decision_edges = []
        self._decision_candidates = {}
        if self.cfg.enable_routing:
            for edge in net.getEdges():
                origin = edge.getFromNode().getID()
                out_edges = [
                    e
                    for e in edge.getToNode().getOutgoing()
                    if e.getID() != edge.getID() and e.getToNode().getID() != origin
                ]
                if len(out_edges) > 1:
                    ranked = sorted(out_edges, key=lambda e: e.getLength() / max(e.getSpeed(), 1e-3))
                    cands = [e.getID() for e in ranked[: self.cfg.routing_k]]
                    if len(cands) > 1:
                        self._decision_edges.append(edge.getID())
                        self._decision_candidates[edge.getID()] = cands
        self._rerouted_this_pass = set()

        # ---- Spaces ----
        nvec = [2] * self.n_agents + [len(self._decision_candidates[e]) for e in self._decision_edges]
        self.action_space = spaces.MultiDiscrete(nvec)
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=-1.0, high=1.0, shape=(self.n_agents, self.NUM_FEATURES), dtype=np.float32
                ),
                "action_mask": spaces.Box(low=0.0, high=1.0, shape=(self.n_agents,), dtype=np.float32),
            }
        )

        # per-junction runtime state, set on reset()
        self._time_in_phase = np.zeros(self.n_agents, dtype=np.float32)
        self._prev_wait = np.zeros(self.n_agents, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self._sumo_started:
            traci.close()
        binary = self.cfg.sumo_binary or ("sumo-gui" if self.cfg.use_gui else "sumo")
        traci.start(
            [
                binary,
                "-n", self.cfg.net_file,
                "-r", self.cfg.route_file,
                "--step-length", str(self.cfg.step_length),
                "--time-to-teleport", "300",
                "--collision.action", "warn",
                "--no-warnings",
                "--no-step-log",
                "--seed", str(seed if seed is not None else self.cfg.seed),
                "--start",
            ]
        )
        self._sumo_started = True
        self._episode_step = 0
        self._episode_arrived = 0
        self._time_in_phase[:] = 0.0
        self._rerouted_this_pass.clear()

        # Take control away from SUMO's built-in actuated logic: fix each TL
        # to phase 0 with an effectively infinite duration so only our
        # switch votes advance it (mirrors the fix used in the 2-junction env).
        for tid in self.tl_ids:
            traci.trafficlight.setPhase(tid, 0)
            traci.trafficlight.setPhaseDuration(tid, 9999)

        self._prev_wait[:] = self._per_junction_waiting()
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action)
        signal_votes = action[: self.n_agents]
        routing_choices = action[self.n_agents:]

        wasted_votes, forced_switches = self._apply_signal_votes(signal_votes)
        if self.cfg.enable_routing:
            self._apply_routing(routing_choices)

        for _ in range(self.cfg.decision_interval):
            traci.simulationStep()
            self._time_in_phase += self.cfg.step_length
            # Checked every step_length seconds (not just once per
            # decision_interval) so a yellow phase lasts exactly
            # cfg.yellow_time real seconds instead of always rounding up
            # to the next decision_interval boundary. See docstring below.
            self._advance_yellow_phases()
            # traci.simulation.getArrivedNumber() only reports vehicles
            # that arrived in the SINGLE most recent simulationStep() call
            # (confirmed in SUMO's own TraCI docs), not since the last
            # time it was queried. Reading it once after this whole
            # decision_interval loop -- as the previous version did --
            # silently discards arrivals from every tick except the last
            # one. Summing it after every individual tick is the only way
            # to not undercount.
            self._episode_arrived += traci.simulation.getArrivedNumber()
        self._episode_step += self.cfg.decision_interval

        wait_now = self._per_junction_waiting()
        agent_rewards = self._prev_wait - wait_now  # positive = less waiting than before
        agent_rewards -= self.cfg.switch_penalty * (signal_votes == 1) * (~wasted_votes)
        agent_rewards -= self.cfg.wasted_vote_penalty * wasted_votes
        self._prev_wait = wait_now

        reward = float(np.sum(agent_rewards))
        obs = self._get_obs()
        queue_now = self._per_junction_queue()

        terminated = False
        truncated = self._episode_step >= self.cfg.episode_seconds or traci.simulation.getMinExpectedNumber() <= 0
        info = {
            "global_state": obs["node_features"].reshape(-1),
            "agent_rewards": agent_rewards.astype(np.float32),
            # UNCAPPED grid totals -- deliberately NOT derived from
            # node_features, whose queue/wait columns are clipped to
            # [0,1] before QUEUE_NORM/WAIT_NORM norm ing (see _get_obs).
            # Any consumer that reconstructs a "total" by summing the
            # clipped features back up (obs[:, 0:4].sum()*QUEUE_NORM etc.)
            # silently caps every junction's contribution at QUEUE_NORM /
            # WAIT_NORM, which understates real congestion once any
            # junction exceeds it. Use these two fields instead whenever
            # an uncapped grid-wide magnitude is needed.
            "total_waiting_time": float(np.sum(wait_now)),
            "total_queue_length": float(np.sum(queue_now)),
            # Cumulative vehicles arrived so far THIS EPISODE (see the
            # accumulation above) -- not this step's arrivals alone. Every
            # existing caller (eval_common.run_episode, dqn/test.py) reads
            # info["throughput"] only once, after the episode ends, and
            # expects a whole-episode total; a per-step reset here would
            # silently reintroduce the undercount for them.
            "throughput": self._episode_arrived,
            "forced_switches": forced_switches,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._sumo_started:
            traci.close()
            self._sumo_started = False

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _advance_yellow_phases(self):
        """Auto-advances any junction that has been sitting in its yellow
        phase for >= cfg.yellow_time back to its paired green phase.

        This is intentionally independent of this step's signal_votes --
        yellow duration is a fixed network-file property, not something a
        vote should gate. Without this, _apply_signal_votes' `not is_yellow`
        eligibility guard means a junction that ever enters yellow (which
        is set with setPhaseDuration(tid, 9999), i.e. it will never expire
        on its own) can never become eligible again: it is permanently
        stuck yellow, action_mask[i] stays 0 forever, and it can never
        vote-switch or be forced again for the rest of the episode.

        Called once per traci.simulationStep() inside step()'s inner loop
        (i.e. every cfg.step_length seconds), NOT once per decision_interval.
        Checking only once per decision_interval would round every yellow
        phase up to a full decision_interval regardless of cfg.yellow_time
        (e.g. yellow_time=4 but decision_interval=5 would always yield a
        5s yellow) -- ticking it at step_length granularity instead means
        yellow lasts the configured cfg.yellow_time as long as yellow_time
        is a multiple of step_length. By the time the NEXT step()'s
        _apply_signal_votes runs, any junction whose yellow interval has
        already elapsed is back on a green phase with _time_in_phase reset
        (it just won't be min_green-eligible until it accumulates
        min_green seconds of its own, same as any other freshly-green
        junction)."""
        for i, tid in enumerate(self.tl_ids):
            state = traci.trafficlight.getRedYellowGreenState(tid).lower()
            is_yellow = "y" in state
            if is_yellow and self._time_in_phase[i] >= self.cfg.yellow_time:
                phase = traci.trafficlight.getPhase(tid)
                n_phases = len(traci.trafficlight.getAllProgramLogics(tid)[0].phases)
                next_phase = (phase + 1) % n_phases  # yellow -> its paired green phase
                traci.trafficlight.setPhase(tid, next_phase)
                traci.trafficlight.setPhaseDuration(tid, 9999)
                self._time_in_phase[i] = 0.0

    def _apply_signal_votes(self, votes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Applies each junction's binary switch vote subject to a
        min_green eligibility floor, AND enforces max_green as a hard
        ceiling: a junction that has been green for >= max_green seconds
        is switched regardless of its vote (a fail-safe every real
        fixed-time/actuated controller has). Every train/eval/plotting
        script in this project (eval_common.py's run_episode,
        dqn/ppo/ac's test.py, switch_timing_plot.py,
        analyze_switching_behavior.py) already reads info["forced_switches"]
        assuming this cap exists -- previously max_green was only used to
        NORMALIZE the elapsed-phase-time feature (see _get_obs) and was
        never actually enforced here, so info["forced_switches"] was never
        populated and any script touching it crashed with a KeyError.

        Returns (wasted, forced), both (n_agents,) bool arrays:
          wasted[i]: voted to switch but wasn't eligible (still yellow, or
                     hasn't held min_green yet) -- the vote had no effect.
          forced[i]: the environment switched this junction because it
                     hit max_green, independent of (or overriding) its vote.
        """
        wasted = np.zeros(self.n_agents, dtype=bool)
        forced = np.zeros(self.n_agents, dtype=bool)
        for i, tid in enumerate(self.tl_ids):
            is_yellow = "y" in traci.trafficlight.getRedYellowGreenState(tid).lower()
            eligible = (not is_yellow) and (self._time_in_phase[i] >= self.cfg.min_green)
            hit_cap = (not is_yellow) and (self._time_in_phase[i] >= self.cfg.max_green)
            wants_switch = votes[i] == 1

            if wants_switch and not eligible:
                wasted[i] = True
            switching_by_vote = wants_switch and eligible
            must_force = hit_cap and not switching_by_vote
            if must_force:
                forced[i] = True

            if switching_by_vote or must_force:
                phase = traci.trafficlight.getPhase(tid)
                n_phases = len(traci.trafficlight.getAllProgramLogics(tid)[0].phases)
                next_phase = (phase + 1) % n_phases  # green -> its paired yellow phase
                traci.trafficlight.setPhase(tid, next_phase)
                traci.trafficlight.setPhaseDuration(tid, 9999)
                self._time_in_phase[i] = 0.0
        return wasted, forced

    def _apply_routing(self, choices: np.ndarray):
        for choice, edge_id in zip(choices, self._decision_edges):
            candidates = self._decision_candidates[edge_id]
            target_edge = candidates[int(choice) % len(candidates)]
            for veh_id in traci.edge.getLastStepVehicleIDs(edge_id):
                if veh_id in self._rerouted_this_pass:
                    continue
                try:
                    dest = traci.vehicle.getRoute(veh_id)[-1]
                    route = traci.simulation.findRoute(target_edge, dest)
                    if route.edges:
                        traci.vehicle.setRoute(veh_id, [edge_id] + list(route.edges))
                    self._rerouted_this_pass.add(veh_id)
                except traci.exceptions.TraCIException:
                    # illegal reroute (e.g. no valid connection) - leave the
                    # vehicle on its existing route rather than crash the episode
                    self._rerouted_this_pass.add(veh_id)
                    continue
        if self._episode_step % self.cfg.routing_interval == 0:
            self._rerouted_this_pass.clear()

    def _per_junction_waiting(self) -> np.ndarray:
        out = np.zeros(self.n_agents, dtype=np.float32)
        for i, tid in enumerate(self.tl_ids):
            total = 0.0
            for lanes in self._approach_lanes[tid].values():
                for lane in lanes:
                    total += traci.lane.getWaitingTime(lane)
            out[i] = total
        return out

    def _per_junction_queue(self) -> np.ndarray:
        """Uncapped per-junction queue length (vehicles), the same raw
        quantity _get_obs() clips to [0,1]*QUEUE_NORM for the observation.
        Exposed separately (via info["total_queue_length"]) so evaluation/
        logging code has an unclipped source instead of reconstructing a
        capped total from node_features."""
        out = np.zeros(self.n_agents, dtype=np.float32)
        for i, tid in enumerate(self.tl_ids):
            total = 0.0
            for lanes in self._approach_lanes[tid].values():
                for lane in lanes:
                    total += traci.lane.getLastStepHaltingNumber(lane)
            out[i] = total
        return out

    def _get_obs(self) -> dict:
        feats = np.zeros((self.n_agents, self.NUM_FEATURES), dtype=np.float32)
        mask = np.zeros(self.n_agents, dtype=np.float32)
        for i, tid in enumerate(self.tl_ids):
            lanes_by_approach = self._approach_lanes[tid]
            for a_idx, approach in enumerate(self.APPROACH_ORDER):
                lanes = lanes_by_approach.get(approach, [])
                if lanes:
                    queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
                    wait = sum(traci.lane.getWaitingTime(l) for l in lanes)
                else:
                    queue, wait = 0.0, 0.0
                feats[i, a_idx] = np.clip(queue / self.QUEUE_NORM, 0, 1)
                feats[i, 4 + a_idx] = np.clip(wait / self.WAIT_NORM, 0, 1)

            phase = traci.trafficlight.getPhase(tid)
            n_phases = len(traci.trafficlight.getAllProgramLogics(tid)[0].phases)
            state = traci.trafficlight.getRedYellowGreenState(tid).lower()
            is_yellow = "y" in state
            feats[i, 8] = phase / max(n_phases - 1, 1)
            feats[i, 9] = 1.0 if is_yellow else 0.0
            feats[i, 10] = np.clip(self._time_in_phase[i] / self.cfg.max_green, 0, 1)

            mask[i] = 1.0 if (not is_yellow and self._time_in_phase[i] >= self.cfg.min_green) else 0.0
        return {"node_features": feats, "action_mask": mask}


# --------------------------------------------------------------------------- #
# Wrapper for plain-MLP baselines (DQN / vanilla PPO / vanilla actor-critic)
# --------------------------------------------------------------------------- #

class FlattenGraphObs(gym.ObservationWrapper):
    """Flattens the Dict(node_features, action_mask) observation into a
    single Box(N*F + N,) vector for algorithms that don't consume graph
    structure directly (SB3's default MlpPolicy). GNN-based critic-actor
    runs should use the raw TrafficGridEnv instead and read env.edge_index."""

    def __init__(self, env: TrafficGridEnv):
        super().__init__(env)
        n, f = env.n_agents, env.NUM_FEATURES
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(n * f + n,), dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([obs["node_features"].reshape(-1), obs["action_mask"]]).astype(np.float32)


if __name__ == "__main__":
    # Smoke test against the uploaded grid4x4 files: random policy rollout.
    cfg = GridEnvConfig(
    net_file="../sumo_4x4_network/grid4x4.net.xml",
    route_file="../sumo_4x4_network/routes.rou.xml",
    episode_seconds=300,
    decision_interval=5,
    )
    env = TrafficGridEnv(cfg)
    print("agents:", env.n_agents, "action_space:", env.action_space, "edges:", env.edge_index.shape[1])
    obs, _ = env.reset(seed=0)
    print("node_features shape:", obs["node_features"].shape, "mask sum:", obs["action_mask"].sum())
    total_reward = 0.0
    for t in range(20):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if t == 0:
            print("sample reward:", reward, "total_waiting_time:", info["total_waiting_time"])
        if term or trunc:
            break
    print("rollout ok, total_reward over", t + 1, "steps:", total_reward)
    env.close()