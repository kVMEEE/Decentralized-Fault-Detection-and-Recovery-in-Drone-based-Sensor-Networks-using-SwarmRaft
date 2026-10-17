#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif

extern "C" {

/**
 * 3D Non-Linear Least Squares (Levenberg-Marquardt) Multilateration.
 * Solves for position (x, y, z) that minimizes sum of squared residuals:
 *   S(x, y, z) = sum_i ( ||p - p_i|| - d_i )^2
 */
EXPORT_API int solve_multilateration_3d(
    const double* anchor_coords,  // Array of size n_anchors * 3: [x0, y0, z0, x1, y1, z1, ...]
    const double* distances,      // Array of size n_anchors
    int n_anchors,
    double* out_pos,              // In/Out array of size 3: [x, y, z]
    int max_iters,
    double tol
) {
    if (n_anchors < 4 || !anchor_coords || !distances || !out_pos) {
        return -1; // Insufficient anchors for full 3D solution
    }

    double x = out_pos[0];
    double y = out_pos[1];
    double z = out_pos[2];
    double lambda = 1e-3;

    for (int iter = 0; iter < max_iters; ++iter) {
        // Accumulate Normal Equations: (J^T J + lambda I) * delta = -J^T r
        double JtJ[3][3] = {{0}};
        double Jtr[3] = {0};
        double total_sq_err = 0.0;

        for (int i = 0; i < n_anchors; ++i) {
            double ax = anchor_coords[i * 3 + 0];
            double ay = anchor_coords[i * 3 + 1];
            double az = anchor_coords[i * 3 + 2];
            double d_meas = distances[i];

            double dx = x - ax;
            double dy = y - ay;
            double dz = z - az;
            double est_dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (est_dist < 1e-6) est_dist = 1e-6;

            double r = est_dist - d_meas;
            total_sq_err += r * r;

            double Jx = dx / est_dist;
            double Jy = dy / est_dist;
            double Jz = dz / est_dist;

            JtJ[0][0] += Jx * Jx;
            JtJ[0][1] += Jx * Jy;
            JtJ[0][2] += Jx * Jz;

            JtJ[1][0] += Jy * Jx;
            JtJ[1][1] += Jy * Jy;
            JtJ[1][2] += Jy * Jz;

            JtJ[2][0] += Jz * Jx;
            JtJ[2][1] += Jz * Jy;
            JtJ[2][2] += Jz * Jz;

            Jtr[0] += Jx * r;
            Jtr[1] += Jy * r;
            Jtr[2] += Jz * r;
        }

        // Add Levenberg-Marquardt damping
        JtJ[0][0] += lambda;
        JtJ[1][1] += lambda;
        JtJ[2][2] += lambda;

        // Invert 3x3 matrix via Cramer's rule
        double det = JtJ[0][0] * (JtJ[1][1] * JtJ[2][2] - JtJ[1][2] * JtJ[2][1]) -
                     JtJ[0][1] * (JtJ[1][0] * JtJ[2][2] - JtJ[1][2] * JtJ[2][0]) +
                     JtJ[0][2] * (JtJ[1][0] * JtJ[2][1] - JtJ[1][1] * JtJ[2][0]);

        if (std::abs(det) < 1e-12) {
            lambda *= 10.0;
            continue;
        }

        double invDet = 1.0 / det;
        double inv[3][3];
        inv[0][0] = (JtJ[1][1] * JtJ[2][2] - JtJ[1][2] * JtJ[2][1]) * invDet;
        inv[0][1] = (JtJ[0][2] * JtJ[2][1] - JtJ[0][1] * JtJ[2][2]) * invDet;
        inv[0][2] = (JtJ[0][1] * JtJ[1][2] - JtJ[0][2] * JtJ[1][1]) * invDet;

        inv[1][0] = (JtJ[1][2] * JtJ[2][0] - JtJ[1][0] * JtJ[2][2]) * invDet;
        inv[1][1] = (JtJ[0][0] * JtJ[2][2] - JtJ[0][2] * JtJ[2][0]) * invDet;
        inv[1][2] = (JtJ[0][2] * JtJ[1][0] - JtJ[0][0] * JtJ[1][2]) * invDet;

        inv[2][0] = (JtJ[1][0] * JtJ[2][1] - JtJ[1][1] * JtJ[2][0]) * invDet;
        inv[2][1] = (JtJ[0][1] * JtJ[2][0] - JtJ[0][0] * JtJ[2][1]) * invDet;
        inv[2][2] = (JtJ[0][0] * JtJ[1][1] - JtJ[0][1] * JtJ[1][0]) * invDet;

        double delta_x = -(inv[0][0] * Jtr[0] + inv[0][1] * Jtr[1] + inv[0][2] * Jtr[2]);
        double delta_y = -(inv[1][0] * Jtr[0] + inv[1][1] * Jtr[1] + inv[1][2] * Jtr[2]);
        double delta_z = -(inv[2][0] * Jtr[0] + inv[2][1] * Jtr[1] + inv[2][2] * Jtr[2]);

        x += delta_x;
        y += delta_y;
        z += delta_z;

        double step_norm = std::sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
        if (step_norm < tol) {
            break;
        }
    }

    out_pos[0] = x;
    out_pos[1] = y;
    out_pos[2] = z;
    return 0;
}

/**
 * Collective Byzantine Detection via Spatial Residual Cross-Validation.
 * Compares claimed peer coordinates against pairwise ranging measurements.
 * Drones exceeding residual threshold against a quorum of peers are flagged as Byzantine.
 */
EXPORT_API int detect_byzantine_nodes(
    const double* reported_coords, // Array of size n_nodes * 3
    const double* pairwise_dist,   // Array of size n_nodes * n_nodes
    int n_nodes,
    double threshold,
    int* out_is_byzantine          // Output array of size n_nodes (1 = byzantine, 0 = honest)
) {
    if (n_nodes < 3 || !reported_coords || !pairwise_dist || !out_is_byzantine) {
        return -1;
    }

    std::vector<int> suspect_votes(n_nodes, 0);

    for (int i = 0; i < n_nodes; ++i) {
        out_is_byzantine[i] = 0;
        double xi = reported_coords[i * 3 + 0];
        double yi = reported_coords[i * 3 + 1];
        double zi = reported_coords[i * 3 + 2];

        for (int j = i + 1; j < n_nodes; ++j) {
            double xj = reported_coords[j * 3 + 0];
            double yj = reported_coords[j * 3 + 1];
            double zj = reported_coords[j * 3 + 2];

            double geom_dist = std::sqrt((xi - xj) * (xi - xj) +
                                         (yi - yj) * (yi - yj) +
                                         (zi - zj) * (zi - zj));
            double meas_dist = pairwise_dist[i * n_nodes + j];

            double residual = std::abs(geom_dist - meas_dist);
            if (residual > threshold) {
                // Inconsistency detected between node i and node j
                suspect_votes[i]++;
                suspect_votes[j]++;
            }
        }
    }

    // A node is Byzantine if flagged by a quorum of peers
    int vote_threshold = (n_nodes - 1) / 2;
    if (vote_threshold < 1) vote_threshold = 1;

    for (int i = 0; i < n_nodes; ++i) {
        if (suspect_votes[i] >= vote_threshold) {
            out_is_byzantine[i] = 1;
        }
    }

    return 0;
}

/**
 * 3D Geometric Dilution of Precision (GDOP).
 * Computes sqrt(trace((H^T * H)^-1)) for target relative to drone constellation.
 */
EXPORT_API double calculate_gdop_3d(
    const double* positions, // Array of size n_nodes * 3
    const double* target,    // Array of size 3
    int n_nodes
) {
    if (n_nodes < 4 || !positions || !target) {
        return 99.9;
    }

    double tx = target[0];
    double ty = target[1];
    double tz = target[2];

    // Compute H^T * H (4x4 matrix)
    double HtH[4][4] = {{0}};

    for (int i = 0; i < n_nodes; ++i) {
        double dx = tx - positions[i * 3 + 0];
        double dy = ty - positions[i * 3 + 1];
        double dz = tz - positions[i * 3 + 2];
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < 1e-6) dist = 1e-6;

        double uv[4] = { dx / dist, dy / dist, dz / dist, 1.0 };

        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                HtH[r][c] += uv[r] * uv[c];
            }
        }
    }

    // Invert 4x4 matrix using Gauss-Jordan with partial pivoting
    double aug[4][8] = {{0}};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            aug[i][j] = HtH[i][j];
        }
        aug[i][i + 4] = 1.0;
    }

    for (int i = 0; i < 4; ++i) {
        // Find pivot
        int pivot = i;
        double max_val = std::abs(aug[i][i]);
        for (int k = i + 1; k < 4; ++k) {
            if (std::abs(aug[k][i]) > max_val) {
                max_val = std::abs(aug[k][i]);
                pivot = k;
            }
        }

        if (max_val < 1e-9) {
            return 99.9; // Singular / collinear geometry
        }

        if (pivot != i) {
            for (int j = 0; j < 8; ++j) {
                std::swap(aug[i][j], aug[pivot][j]);
            }
        }

        double div = aug[i][i];
        for (int j = 0; j < 8; ++j) {
            aug[i][j] /= div;
        }

        for (int k = 0; k < 4; ++k) {
            if (k != i) {
                double factor = aug[k][i];
                for (int j = 0; j < 8; ++j) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    double trace = aug[0][4] + aug[1][5] + aug[2][6] + aug[3][7];
    if (trace <= 0.0 || std::isnan(trace)) {
        return 99.9;
    }

    double gdop = std::sqrt(trace);
    return std::min(gdop, 99.9);
}

} // extern "C"
