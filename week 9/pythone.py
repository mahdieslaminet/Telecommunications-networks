Python 3.13.3 (tags/v3.13.3:6280bb5, Apr  8 2025, 14:47:33) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
# =========================================================
# UavNetSim-v1 : UAV Communication Network Simulator
# Based on the reference paper
# =========================================================

# -----------------------------
# Imports
# -----------------------------
import simpy
import random
import numpy as np
import math
import matplotlib.pyplot as plt

# -----------------------------
# Global Simulation Parameters
# -----------------------------
NUM_NODES = 10
AREA_SIZE = 500          # meters
TX_RANGE = 150           # meters
SIM_TIME = 200           # seconds
PACKET_RATE = 1          # packets per second
PROP_DELAY = 0.01        # propagation delay (sec)

# -----------------------------
# Packet Class
# -----------------------------
class Packet:
    def __init__(self, src, dst, time):
        self.src = src
        self.dst = dst
        self.creation_time = time
        self.hops = 0

# -----------------------------
# UAV Node Class
# -----------------------------
class UAVNode:
    def __init__(self, env, node_id, stats):
        self.env = env
        self.id = node_id
        self.stats = stats
        self.nodes = []

        # Initial 3D Position
        self.x = random.uniform(0, AREA_SIZE)
        self.y = random.uniform(0, AREA_SIZE)
        self.z = random.uniform(50, 150)

        # Start Processes
        self.env.process(self.mobility())
        self.env.process(self.generate_packets())

    # -----------------------------
    # Distance Calculation
    # -----------------------------
    def distance(self, other):
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    # -----------------------------
    # Mobility Model (Random Waypoint)
    # -----------------------------
    def mobility(self):
        while True:
            dx = random.uniform(0, AREA_SIZE)
            dy = random.uniform(0, AREA_SIZE)
            dz = random.uniform(50, 150)
            speed = random.uniform(5, 15)

            dist = math.sqrt(
                (dx - self.x) ** 2 +
                (dy - self.y) ** 2 +
                (dz - self.z) ** 2
            )

            travel_time = dist / speed if speed > 0 else 1
            yield self.env.timeout(travel_time)

            self.x, self.y, self.z = dx, dy, dz

    # -----------------------------
    # Packet Generation
    # -----------------------------
    def generate_packets(self):
        while True:
            yield self.env.timeout(random.expovariate(PACKET_RATE))
            dst_node = random.choice([n for n in self.nodes if n.id != self.id])
            pkt = Packet(self.id, dst_node.id, self.env.now)
            self.send_packet(pkt)

    # -----------------------------
    # Greedy Routing
    # -----------------------------
    def send_packet(self, packet):
        destination = self.nodes[packet.dst]
        next_hop = None
        min_distance = float("inf")

        for node in self.nodes:
            if node.id != self.id and self.distance(node) <= TX_RANGE:
                d = node.distance(destination)
                if d < min_distance:
                    min_distance = d
                    next_hop = node

        if next_hop is not None:
            packet.hops += 1
            self.env.process(self.transmit(packet, next_hop))
        else:
            self.stats["lost"] += 1

    # -----------------------------
    # Transmission (ALOHA-like MAC)
    # -----------------------------
    def transmit(self, packet, next_hop):
        yield self.env.timeout(PROP_DELAY)

        if next_hop.id == packet.dst:
            delay = self.env.now - packet.creation_time
            self.stats["received"] += 1
            self.stats["delay"].append(delay)
        else:
            next_hop.send_packet(packet)

# -----------------------------
# Simulation Runner
# -----------------------------
def run_simulation():
    env = simpy.Environment()

    stats = {
        "received": 0,
...         "lost": 0,
...         "delay": []
...     }
... 
...     nodes = [UAVNode(env, i, stats) for i in range(NUM_NODES)]
... 
...     for node in nodes:
...         node.nodes = nodes
... 
...     env.run(until=SIM_TIME)
...     return stats
... 
... # -----------------------------
... # Main Execution
... # -----------------------------
... if __name__ == "__main__":
...     stats = run_simulation()
... 
...     total_packets = stats["received"] + stats["lost"]
...     pdr = stats["received"] / total_packets if total_packets > 0 else 0
...     avg_delay = np.mean(stats["delay"]) if stats["delay"] else 0
... 
...     print("=================================")
...     print("UAV Network Simulation Results")
...     print("=================================")
...     print("Total Packets:", total_packets)
...     print("Received Packets:", stats["received"])
...     print("Lost Packets:", stats["lost"])
...     print("Packet Delivery Ratio (PDR):", round(pdr, 3))
...     print("Average End-to-End Delay:", round(avg_delay, 4), "sec")
... 
...     # Plot Delay Histogram
...     plt.figure()
...     plt.hist(stats["delay"], bins=20)
...     plt.xlabel("End-to-End Delay (sec)")
...     plt.ylabel("Number of Packets")
...     plt.title("Delay Distribution in UAV Network")
...     plt.grid(True)
...     plt.show()
