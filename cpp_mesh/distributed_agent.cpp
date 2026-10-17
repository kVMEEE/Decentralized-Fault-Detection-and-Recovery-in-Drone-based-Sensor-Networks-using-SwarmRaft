#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>
#include <cmath>
#include <map>
#include <set>
#include <random>

#ifdef _WIN32
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <unistd.h>
#endif

#define MAX_BUFFER 2048
#define BASE_PORT 8000
#define VIZ_PORT 9000

#pragma pack(push, 1)
struct DroneTelemetry {
    int32_t drone_id;
    int8_t is_leader;
    
    // Raft
    int32_t current_term;
    int32_t voted_for;
    
    // Position
    double true_x, true_y, true_z;
    double gnss_x, gnss_y, gnss_z;
    double ins_x, ins_y, ins_z;
    double rec_x, rec_y, rec_z;
    
    double distances[20]; 
    int32_t active_neighbors;
};
#pragma pack(pop)

class DroneAgent {
private:
    int id;
    int total_drones;
    int f_threshold; 
    bool running;
    DroneTelemetry current_state;
    std::mutex state_mutex;
    int udp_socket;

    // Cluster tracking
    std::map<int, DroneTelemetry> cluster_state;
    std::map<int, std::chrono::steady_clock::time_point> last_heard;
    std::set<int> votes_granted;
    
    int election_timeout;
    
    // Kinematic state for Dead Reckoning
    double vel_x, vel_y, vel_z;
    double prev_rec_x, prev_rec_y, prev_rec_z;

public:
    DroneAgent(int id, int total) : id(id), total_drones(total), running(true) {
        f_threshold = (total - 1) / 2; 
        vel_x = 0.05; vel_y = -0.02; vel_z = 0.0;

#ifdef _WIN32
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
        udp_socket = socket(AF_INET, SOCK_DGRAM, 0);
        
        // Setup Non-Blocking Receive
        u_long mode = 1;
        ioctlsocket(udp_socket, FIONBIO, &mode);
        
        sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(BASE_PORT + id);
        bind(udp_socket, (struct sockaddr *)&server_addr, sizeof(server_addr));
        
        current_state.drone_id = id;
        current_state.is_leader = (id == 0) ? 1 : 0;
        current_state.current_term = 1;
        current_state.voted_for = (id == 0) ? 0 : -1;
        
        // 3D Staggered Cluster Formation
        double angle = (2.0 * 3.1415926535 * id) / total;
        double radius = 25.0;
        double start_x = 50.0 + radius * std::cos(angle);
        double start_y = 50.0 + radius * std::sin(angle);
        double start_z = 25.0 + 6.0 * std::sin(angle * 2);

        current_state.true_x = start_x; current_state.true_y = start_y; current_state.true_z = start_z;
        current_state.gnss_x = start_x; current_state.gnss_y = start_y; current_state.gnss_z = start_z;
        current_state.ins_x = start_x;  current_state.ins_y = start_y;  current_state.ins_z = start_z;
        current_state.rec_x = start_x;  current_state.rec_y = start_y;  current_state.rec_z = start_z;
        
        prev_rec_x = start_x; prev_rec_y = start_y; prev_rec_z = start_z;
        
        current_state.active_neighbors = 0;
        std::srand(std::time(nullptr) + id * 100);
        election_timeout = 20 + (std::rand() % 20);
    }

    void receive_loop() {
        char buffer[MAX_BUFFER];
        sockaddr_in client_addr;
        int client_len = sizeof(client_addr);
        
        while (running) {
            int n = recvfrom(udp_socket, buffer, MAX_BUFFER, 0, (struct sockaddr*)&client_addr, &client_len);
            if (n > 0) {
                std::lock_guard<std::mutex> lock(state_mutex);
                DroneTelemetry* ns = (DroneTelemetry*)buffer;
                int n_id = ns->drone_id;
                cluster_state[n_id] = *ns;
                last_heard[n_id] = std::chrono::steady_clock::now();
                
                // Raft Term tracking
                if (ns->current_term > current_state.current_term) {
                    current_state.current_term = ns->current_term;
                    current_state.voted_for = -1;
                    current_state.is_leader = 0;
                    votes_granted.clear();
                }

                // If candidate receives vote for self
                if (current_state.is_leader == 0 && ns->voted_for == id && ns->current_term == current_state.current_term) {
                    votes_granted.insert(n_id);
                    int quorum = (total_drones / 2) + 1;
                    if ((int)votes_granted.size() + 1 >= quorum) {
                        current_state.is_leader = 1;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    void broadcast_loop() {
        while (running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 10Hz Mesh Loop
            std::lock_guard<std::mutex> lock(state_mutex);
            
            auto now = std::chrono::steady_clock::now();
            int active = 0;
            
            // Clean dead nodes & Synthesize Range Distances (UWB/Radio TOF)
            for (auto it = cluster_state.begin(); it != cluster_state.end();) {
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_heard[it->first]).count() > 1000) {
                    it = cluster_state.erase(it);
                } else {
                    active++;
                    double dx = current_state.true_x - it->second.true_x;
                    double dy = current_state.true_y - it->second.true_y;
                    double dz = current_state.true_z - it->second.true_z;
                    double noise = (((std::rand() % 100) / 100.0) - 0.5) * 0.4; // +/- 0.2m noise
                    current_state.distances[it->first] = std::sqrt(dx*dx + dy*dy + dz*dz) + noise;
                    it++;
                }
            }
            current_state.active_neighbors = active;
            
            // Raft Leader Election Phase (Quorum Based)
            if (current_state.is_leader == 0) {
                bool heard_leader = false;
                for (auto& cs : cluster_state) {
                    if (cs.second.is_leader == 1 && cs.second.current_term >= current_state.current_term) {
                        heard_leader = true;
                    }
                }
                
                if (!heard_leader) {
                    election_timeout--;
                    if (election_timeout <= 0) {
                        current_state.current_term++;
                        current_state.voted_for = id;
                        votes_granted.clear();
                        election_timeout = 25 + (std::rand() % 25);
                    }
                } else {
                    election_timeout = 25 + (std::rand() % 25);
                }
            }

            // Dynamic Cross-Ranging Byzantine Detection & Multilateration
            int honest_anchors = 0;
            std::vector<int> trusted_peers;

            for (auto& cs : cluster_state) {
                double nx = cs.second.gnss_x;
                double ny = cs.second.gnss_y;
                double nz = cs.second.gnss_z;
                
                double dx = current_state.rec_x - nx;
                double dy = current_state.rec_y - ny;
                double dz = current_state.rec_z - nz;
                double estimated_dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                double measured_dist = current_state.distances[cs.first];

                double residual = std::abs(estimated_dist - measured_dist);
                
                // Dynamic Byzantine Filter: Residual threshold check (no hardcoded IDs)
                if (residual < 3.0) {
                    trusted_peers.push_back(cs.first);
                    honest_anchors++;
                }
            }

            if (honest_anchors < 3 && total_drones > 3) {
                // Kinematic Dead-Reckoning Fallback
                current_state.ins_x += vel_x * 0.1;
                current_state.ins_y += vel_y * 0.1;
                current_state.ins_z += vel_z * 0.1;

                current_state.rec_x = current_state.ins_x;
                current_state.rec_y = current_state.ins_y;
                current_state.rec_z = current_state.ins_z;
            } else if (!trusted_peers.empty()) {
                // Full 3D Multilateration Gradient Descent
                double lr = 0.04;
                for (int iter = 0; iter < 12; iter++) {
                    double grad_x = 0, grad_y = 0, grad_z = 0;
                    for (int peer_id : trusted_peers) {
                        auto& cs = cluster_state[peer_id];
                        double nx = cs.gnss_x;
                        double ny = cs.gnss_y;
                        double nz = cs.gnss_z;
                        
                        double dx = current_state.rec_x - nx;
                        double dy = current_state.rec_y - ny;
                        double dz = current_state.rec_z - nz;
                        double dist = std::sqrt(dx*dx + dy*dy + dz*dz) + 1e-6;
                        double error = dist - current_state.distances[peer_id];
                        
                        grad_x += error * (dx / dist);
                        grad_y += error * (dy / dist);
                        grad_z += error * (dz / dist);
                    }
                    current_state.rec_x -= lr * grad_x;
                    current_state.rec_y -= lr * grad_y;
                    current_state.rec_z -= lr * grad_z;
                }

                // Update estimated kinematic velocity for dead reckoning
                vel_x = (current_state.rec_x - prev_rec_x) / 0.1;
                vel_y = (current_state.rec_y - prev_rec_y) / 0.1;
                vel_z = (current_state.rec_z - prev_rec_z) / 0.1;
                prev_rec_x = current_state.rec_x;
                prev_rec_y = current_state.rec_y;
                prev_rec_z = current_state.rec_z;

                current_state.ins_x = current_state.rec_x;
                current_state.ins_y = current_state.rec_y;
                current_state.ins_z = current_state.rec_z;
            }

            // Artificially Spoof GPS coordinates on Node 0 and Node 1 to simulate attack
            if (id == 0) {
                current_state.gnss_x += 0.05;
                current_state.gnss_y -= 0.03;
            }
            if (id == 1) {
                current_state.gnss_x -= 0.04;
                current_state.gnss_y += 0.05;
            }

            // Mesh Broadcast to Peers
            for (int i = 0; i < total_drones; i++) {
                if (i == id) continue;
                sockaddr_in target_addr;
                target_addr.sin_family = AF_INET;
                target_addr.sin_port = htons(BASE_PORT + i);
                target_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
                sendto(udp_socket, (char*)&current_state, sizeof(DroneTelemetry), 0, (struct sockaddr*)&target_addr, sizeof(target_addr));
            }
            
            // Broadcast to 3D Visualizer
            sockaddr_in viz_addr;
            viz_addr.sin_family = AF_INET;
            viz_addr.sin_port = htons(VIZ_PORT);
            viz_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
            sendto(udp_socket, (char*)&current_state, sizeof(DroneTelemetry), 0, (struct sockaddr*)&viz_addr, sizeof(viz_addr));
        }
    }

    void start() {
        std::thread rx_thread(&DroneAgent::receive_loop, this);
        std::thread tx_thread(&DroneAgent::broadcast_loop, this);
        rx_thread.join();
        tx_thread.join();
    }
};

int main(int argc, char* argv[]) {
    if (argc < 3) return 1;
    DroneAgent agent(std::stoi(argv[1]), std::stoi(argv[2]));
    agent.start();
    return 0;
}
