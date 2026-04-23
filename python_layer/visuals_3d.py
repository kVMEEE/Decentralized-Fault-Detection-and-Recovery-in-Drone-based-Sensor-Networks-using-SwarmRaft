import socket
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Configure UDP Listen Server
UDP_IP = "127.0.0.1"
UDP_PORT = 9000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

# C-Struct Matching 
PACK_FMT = '<i b i i d d d d d d d d d d d d ' + ('d' * 20) + ' i'
STRUCT_SIZE = struct.calcsize(PACK_FMT)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
drones_data = {}

def update(frame):
    while True:
        try:
            data, addr = sock.recvfrom(2048)
            if len(data) == STRUCT_SIZE:
                unpacked = struct.unpack(PACK_FMT, data)
                d_id = unpacked[0]
                t_x, t_y, t_z = unpacked[4:7]    # True
                g_x, g_y, g_z = unpacked[7:10]   # GNSS (Spoofed/INS)
                r_x, r_y, r_z = unpacked[13:16]  # Recovered
                
                drones_data[d_id] = {'true': (t_x, t_y, t_z), 'gnss': (g_x, g_y, g_z), 'rec': (r_x, r_y, r_z)}
        except BlockingIOError:
            break
            
    ax.clear()
    ax.set_title("Live SwarmRaft 3D Telemetry Feed")
    ax.set_xlim([0, 100]); ax.set_ylim([0, 100]); ax.set_zlim([0, 100])
    
    for d_id, data in drones_data.items():
        ax.scatter(*data['true'], color='green', marker='o', s=300, alpha=0.5, label='True' if d_id==0 else "")
        ax.scatter(*data['gnss'], color='red', marker='x', s=50, label='Spoofed/INS' if d_id==0 else "")
        ax.scatter(*data['rec'], color='gold', marker='*', s=150, edgecolor='black', label='Recovered' if d_id==0 else "")
        ax.plot([data['true'][0], data['gnss'][0]], [data['true'][1], data['gnss'][1]], [data['true'][2], data['gnss'][2]], color='gray')
                
    if drones_data:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys())

ani = animation.FuncAnimation(fig, update, interval=33, cache_frame_data=False)
if __name__ == "__main__":
    plt.show()
