# CS472_Final_Project
Final Project for CS472 Unsupervised Machine Learning

# Network Sentinel: HDBSCAN-Powered Edge IPS
This project implements a real-time Intrusion Prevention System (IPS) designed to run on resource-constrained edge hardware, such as a Raspberry Pi or x86 Mini PC. It leverages Unsupervised Machine Learning to protect home networks and private game servers (e.g., Minecraft, Terraria) from malicious traffic.

# Behavioral Detection via HDBSCAN
Unlike traditional firewalls that rely on static "signatures," this system identifies threats based on mathematical density and statistical noise.

Dimensionality Reduction (PCA): High-dimensional network features (packet size, TTL, byte counts) are condensed using Principal Component Analysis (PCA) to remove redundant noise and enable efficient clustering.

Density-Based Clustering: The HDBSCAN algorithm groups incoming traffic into "stable" clusters representing legitimate behavior.

The "Noise" Signal: Malicious activity—such as DDoS floods, Port Scanning, or Brute Force attacks—manifests as "Noise" (-1) because it fails to fit into established, dense clusters of normal traffic.

Protocol Awareness: The model is trained to recognize the unique "heartbeat" of gaming protocols (UDP) and VoIP (Discord), preventing them from being falsely flagged as anomalies.

# Automated Mitigation (The "Auto-Kick")
The system creates a proactive feedback loop between the AI model and the Linux kernel to defend the network in real-time.

Asynchronous Inspection: To prevent game lag, the system uses non-polling, event-driven packet capture. It "mirrors" traffic batches for analysis rather than stalling every individual packet.

Dynamic Firewalling: When a specific IP is flagged as "Noise" with high probability, the Python backend executes a system call to the iptables or nftables firewall.

Instant Mitigation: Malicious IPs are "blackholed" at the Public IP gateway, dropping all future packets before they can penetrate the internal network or consume bandwidth.

# Deployment Architecture
Edge Node: Designed to run on a Mini PC or Raspberry Pi positioned as a transparent bridge between the ISP Modem and the internal Router.

Sandbox Testing: Initial baselines were captured in an isolated Mobile Hotspot environment to ensure high-fidelity "Normal" data without external network interference.

Tech Stack: * Language: Python (AI Logic), C (High-speed Packet Sniffing/Filtering).

Libraries: scikit-learn (HDBSCAN/PCA), Scapy (Packet Manipulation), pandas.

Firewall: Linux iptables.

# Practical Use Case: Private Game Hosting
This system is specifically tuned for home-hosted game servers:

DDoS Protection: Blocks massive request floods that would otherwise crash a home modem.

Scan Prevention: Identifies and blocks "Port Scouts" before they can find open game ports like 25565 or 7777.

Latency-Based Quality Control: Can be configured to automatically mitigate high-latency connections that degrade the gaming experience for other players.
