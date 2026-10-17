import enum
import random
import time
from typing import Dict, List, Optional, Tuple

class RaftRole(enum.Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
    DEAD = "dead"

class RaftNode:
    """
    Raft Consensus Node implementing distributed leader election,
    term tracking, and quorum voting across the swarm.
    """
    def __init__(self, node_id: int, total_nodes: int):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.role = RaftRole.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[int] = None
        self.votes_received = set()
        self.current_leader: Optional[int] = None

        # Randomized election timeout between 15 and 30 ticks
        self.election_timeout = random.randint(15, 30)
        self.timer = self.election_timeout
        self.heartbeat_interval = 5

    def reset_timer(self):
        self.election_timeout = random.randint(15, 30)
        self.timer = self.election_timeout

    def tick(self, active_cluster: Dict[int, 'RaftNode']) -> List[str]:
        """
        Clock tick for consensus state machine.
        Returns list of log messages describing consensus actions taken.
        """
        logs = []
        if self.role == RaftRole.DEAD:
            return logs

        if self.role == RaftRole.LEADER:
            # Broadcast periodic heartbeat to all active peers
            self.timer -= 1
            if self.timer <= 0:
                self.timer = self.heartbeat_interval
                for peer_id, peer in active_cluster.items():
                    if peer_id != self.node_id and peer.role != RaftRole.DEAD:
                        peer.receive_heartbeat(self.node_id, self.current_term)
            return logs

        # Follower or Candidate
        self.timer -= 1
        if self.timer <= 0:
            # Election timeout expired -> Start election
            logs.extend(self.start_election(active_cluster))

        return logs

    def start_election(self, active_cluster: Dict[int, 'RaftNode']) -> List[str]:
        logs = []
        self.role = RaftRole.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.votes_received = {self.node_id}
        self.current_leader = None
        self.reset_timer()

        alive_count = sum(1 for p in active_cluster.values() if p.role != RaftRole.DEAD)
        quorum = (alive_count // 2) + 1

        logs.append(f"Raft: Drone {self.node_id} started election for Term {self.current_term} (needs {quorum}/{alive_count} votes)")

        # Request votes from all alive peers
        for peer_id, peer in active_cluster.items():
            if peer_id != self.node_id and peer.role != RaftRole.DEAD:
                granted = peer.request_vote(self.node_id, self.current_term)
                if granted:
                    self.votes_received.add(peer_id)

        # Check if majority achieved
        if len(self.votes_received) >= quorum:
            self.role = RaftRole.LEADER
            self.current_leader = self.node_id
            logs.append(f"Raft: Drone {self.node_id} achieved quorum ({len(self.votes_received)}/{alive_count} votes) -> Promoted to LEADER (Term {self.current_term})")
            # Immediate heartbeat
            for peer_id, peer in active_cluster.items():
                if peer_id != self.node_id and peer.role != RaftRole.DEAD:
                    peer.receive_heartbeat(self.node_id, self.current_term)

        return logs

    def request_vote(self, candidate_id: int, term: int) -> bool:
        """Process RequestVote RPC from candidate."""
        if self.role == RaftRole.DEAD:
            return False

        if term > self.current_term:
            self.current_term = term
            self.role = RaftRole.FOLLOWER
            self.voted_for = None
            self.current_leader = None

        if term == self.current_term and (self.voted_for is None or self.voted_for == candidate_id):
            self.voted_for = candidate_id
            self.reset_timer()
            return True

        return False

    def receive_heartbeat(self, leader_id: int, term: int):
        """Process Heartbeat (AppendEntries) RPC from leader."""
        if self.role == RaftRole.DEAD:
            return

        if term >= self.current_term:
            self.current_term = term
            self.current_leader = leader_id
            self.role = RaftRole.FOLLOWER
            self.voted_for = None
            self.reset_timer()


class SwarmConsensusCluster:
    """Manages decentralized consensus across the entire drone fleet."""
    def __init__(self, n_drones: int):
        self.nodes = {i: RaftNode(i, n_drones) for i in range(n_drones)}
        # Elect initial leader (node 0)
        self.nodes[0].role = RaftRole.LEADER
        self.nodes[0].current_term = 1
        for i in range(n_drones):
            self.nodes[i].current_leader = 0
            self.nodes[i].current_term = 1

    @property
    def leader_id(self) -> int:
        for nid, node in self.nodes.items():
            if node.role == RaftRole.LEADER:
                return nid
        return -1

    def kill_node(self, node_id: int):
        if node_id in self.nodes:
            self.nodes[node_id].role = RaftRole.DEAD
            self.nodes[node_id].current_leader = None

    def tick(self) -> List[str]:
        logs = []
        for nid, node in self.nodes.items():
            if node.role != RaftRole.DEAD:
                node_logs = node.tick(self.nodes)
                logs.extend(node_logs)
        return logs
