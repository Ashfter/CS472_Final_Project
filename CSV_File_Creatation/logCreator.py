from scapy.all import sniff, IP, TCP, UDP
import pandas as pd
import time

# string for application name
app_name = input("What application are you logging? (e.g., Terraria, Valorant, Minecraft, etc)")

# set packet count
packet_count = 150000

# bucket to store packetsM
packet_list = []

# file name creation
filename = f"baseline_{app_name}.csv"


def packet_inspector(pkt):
    # 1. We only care about IP traffic (ignore low-level hardware noise)
    if IP in pkt:
        # 2. Extract basic IP info
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        proto_num = pkt[IP].proto  # 6 for TCP, 17 for UDP
        
        # 3. Handle Ports (Logic: check if it's TCP or UDP first)
        src_port = 0
        dst_port = 0
        
        if TCP in pkt:
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif UDP in pkt:
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport

        # 4. Create the Row (Dictionary) to match UNSW-NB15
        # We add the "empty" attack columns here too!
        data_row = {
            'srcip': src_ip,
            'sport': src_port,
            'dstip': dst_ip,
            'dsport': dst_port,
            'proto': 'tcp' if TCP in pkt else 'udp',
            'sbytes': len(pkt),
            'timestamp': time.time(),
            'attack_cat': '',  # Auto-fill Normal
            'label': 0         # Auto-fill Normal
        }

        # 5. Throw it in the bucket
        packet_list.append(data_row)

# Print sniffing initiation
print(f"Sniffing {packet_count} packets. Start your game now!")

# 'prn' is the callback
# 'count' is your limit
# 'store=0' tells Scapy not to keep a second copy in its own memory
sniff(prn=packet_inspector, count=packet_count, store=0)

# raw datafram
raw_df = pd.DataFrame(packet_list)

print("\nCapture complete! Formatting data...")

# format to UNSW 
flow_df = raw_df.groupby(['srcip', 'sport', 'dstip', 'dsport', 'proto']).agg({
    'sbytes': 'sum', 'timestamp':['min', 'max']}).reset_index()

# fix column names
flow_df.columns = ['srcip', 'sport', 'dstip', 'dsport',
                    'proto', 'sbytes', 'start_time', 'end_time']
flow_df['dur'] = flow_df['end_time'] - flow_df['start_time']

flow_df['attack_cat'] = ''
flow_df['label'] = 0

# re order columns
final_df = flow_df[['srcip', 'sport', 'dstip', 'dsport',
                     'proto', 'dur', 'sbytes', 'attack_cat', 'label']]

# Save to CSV
final_df.to_csv(filename, index=False, header=False)

print(f"Success! {len(final_df)} rows saved to {filename}")