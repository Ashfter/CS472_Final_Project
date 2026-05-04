import os
import time

import pandas as pd
from scapy.all import sniff, IP, TCP, UDP


def capture_baseline(app_name: str, packet_count: int = 150000) -> pd.DataFrame:
    packet_list = []

    def packet_inspector(pkt):
        if IP not in pkt:
            return
        src_port = dst_port = 0
        if TCP in pkt:
            src_port, dst_port = pkt[TCP].sport, pkt[TCP].dport
        elif UDP in pkt:
            src_port, dst_port = pkt[UDP].sport, pkt[UDP].dport

        packet_list.append({
            'srcip': pkt[IP].src,
            'sport': src_port,
            'dstip': pkt[IP].dst,
            'dsport': dst_port,
            'proto': 'tcp' if TCP in pkt else 'udp',
            'sbytes': len(pkt),
            'timestamp': time.time(),
            'attack_cat': '',
            'label': 0,
        })

    print(f"Sniffing {packet_count} packets. Start your game now!")
    sniff(prn=packet_inspector, count=packet_count, store=0)
    return pd.DataFrame(packet_list)


def build_flow_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    print("\nCapture complete! Formatting data...")
    flow_df = raw_df.groupby(
        ['srcip', 'sport', 'dstip', 'dsport', 'proto']
    ).agg({'sbytes': 'sum', 'timestamp': ['min', 'max']}).reset_index()

    flow_df.columns = [
        'srcip', 'sport', 'dstip', 'dsport', 'proto', 'sbytes', 'start_time', 'end_time'
    ]
    flow_df['dur'] = flow_df['end_time'] - flow_df['start_time']
    flow_df['attack_cat'] = ''
    flow_df['label'] = 0
    return flow_df[['srcip', 'sport', 'dstip', 'dsport', 'proto', 'dur', 'sbytes', 'attack_cat', 'label']]


def main():
    app_name = input("What application are you logging? (e.g., Terraria, Valorant, Minecraft, etc): ")

    raw_df = capture_baseline(app_name)
    final_df = build_flow_df(raw_df)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(project_root, "CSV_File_Creation", f"baseline_{app_name}.csv")
    final_df.to_csv(out_path, index=False, header=False)
    print(f"Success! {len(final_df)} rows saved to {out_path}")


if __name__ == "__main__":
    main()
