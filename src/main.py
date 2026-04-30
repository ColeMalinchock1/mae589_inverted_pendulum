"""

Script to simulate a LQR 

"""

import time
import numpy as np
from scipy import linalg
import mujoco
import mujoco.viewer
import csv

from cart_manipulator import CartManipulator
from lib.constants import (
    G,
    U_MIN, U_MAX,
    MC, M1, M2, L1, LC1, LC2, I1, I2,
    TARGETS, THRESHOLD,
    K_ENERGY, K_CART_P, K_CART_D,
    x0, x_dot0, q10, q1dot0, q20, q2dot0 # INITIAL STATES
)


class MuJoCoRunner():

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("../world/scene.xml")
        self.data  = mujoco.MjData(self.model)
        self.cart  = CartManipulator(self.model, self.data)
        self.dt    = self.model.opt.timestep

        # --- Logging setup ---
        self.log_file = open("sim_log.csv", "w", newline="")
        self.logger   = csv.writer(self.log_file)
        self.logger.writerow([
            "t", "phase", "target_x",
            "x", "x_dot", "theta1_deg", "theta2_deg",
            "theta1_dot", "theta2_dot", "u"
        ])
        self.log_every  = 10
        self.step_count = 0

        # Set initial state (pendulums hanging down)
        self.data.qpos[self.model.joint('slider').qposadr[0]] = x0
        self.data.qpos[self.model.joint('hinge1').qposadr[0]] = q10  # = pi
        self.data.qpos[self.model.joint('hinge2').qposadr[0]] = q20  # = pi
        self.data.qvel[self.model.joint('slider').dofadr[0]]  = x_dot0
        self.data.qvel[self.model.joint('hinge1').dofadr[0]]  = q1dot0
        self.data.qvel[self.model.joint('hinge2').dofadr[0]]  = q2dot0
        mujoco.mj_forward(self.model, self.data)

        self.setup()

    def setup(self):
        # ---------------------------------------------------------------
        # Precompute inertia/gravity constants (same h-params as the
        # GEKKO tutorial — used in both the energy swing-up and the LQR
        # linearization).
        # ---------------------------------------------------------------
        h1 = MC + M1 + M2
        h2 = M1*LC1 + M2*L1
        h3 = M2*LC2
        h4 = M1*LC1**2 + M2*L1**2 + I1
        h5 = M2*LC2*L1
        h6 = M2*LC2**2 + I2
        h7 = M1*LC1*G + M2*L1*G
        h8 = M2*LC2*G

        # Store the ones needed at runtime by the swing-up controller
        self.h4 = h4
        self.h5 = h5
        self.h6 = h6
        self.h7 = h7
        self.h8 = h8

        # ---------------------------------------------------------------
        # Build linearized A, B matrices around the upright equilibrium
        # (theta1 = theta2 = 0).  At the top, sin(q) ≈ q and cos(q) ≈ 1,
        # so M is constant and the gravity terms linearize to +h7*q1, +h8*q2.
        # ---------------------------------------------------------------
        M_lin = np.array([[h1, h2, h3],
                          [h2, h4, h5],
                          [h3, h5, h6]])
        M_inv = np.linalg.inv(M_lin)

        # Column vectors: how gravity and input map through the inertia
        t1c = M_inv @ np.array([0, h7, 0])   # effect of link-1 gravity
        t2c = M_inv @ np.array([0,  0, h8])  # effect of link-2 gravity
        bc  = M_inv @ np.array([1,  0,  0])  # effect of cart force

        # State vector: [x, xdot, theta1, theta1dot, theta2, theta2dot]
        self.A = np.array([
            [0, 1, 0,      0, 0,      0],
            [0, 0, t1c[0], 0, t2c[0], 0],
            [0, 0, 0,      1, 0,      0],
            [0, 0, t1c[1], 0, t2c[1], 0],
            [0, 0, 0,      0, 0,      1],
            [0, 0, t1c[2], 0, t2c[2], 0]
        ])
        self.B = np.array([[0], [bc[0]], [0], [bc[1]], [0], [bc[2]]])

        # ---------------------------------------------------------------
        # Solve the LQR Riccati equation to get the optimal gain K.
        # Q penalises state errors; R penalises control effort.
        # Increase Q diagonal entries to make the controller care more
        # about that state; increase R to limit how hard it pushes.
        # ---------------------------------------------------------------
        Q = np.diag([10.0, 1.0, 100.0, 1.0, 100.0, 1.0])
        R = np.array([[1.0]])
        P      = linalg.solve_continuous_are(self.A, self.B, Q, R)
        self.K = linalg.inv(R) @ self.B.T @ P

        print("LQR gain K computed:")
        print(self.K)

        self.sim_time   = 0.0
        self.phase      = "swingup"   # starts here, switches to "lqr"
        self.target_idx = 0

    # -------------------------------------------------------------------
    # SWING-UP: Energy-pumping controller
    # -------------------------------------------------------------------
    def energy_swingup(self, x, x_dot, theta1, theta2, theta1_dot, theta2_dot):
        """
        Pumps mechanical energy into the two-link pendulum until it has
        enough to reach the upright position, then lets LQR take over.

        Total energy of the two links relative to the upright equilibrium:

            KE = 0.5 * (h4*θ1dot² + 2*h5*cos(θ1-θ2)*θ1dot*θ2dot + h6*θ2dot²)
            PE = h7*(cos(θ1) - 1) + h8*(cos(θ2) - 1)
            E  = KE + PE

        At upright rest:  E = 0   (desired)
        At bottom at rest: E = -2*h7 - 2*h8  (very negative)

        Control law:
            u_swing = K_ENERGY * θ1dot * cos(θ1) * (0 - E)
            u_cart  = -K_CART_P * x - K_CART_D * x_dot   (keeps cart centered)
            u = u_swing + u_cart
        """
        KE = 0.5 * (
            self.h4 * theta1_dot**2
            + 2 * self.h5 * np.cos(theta1 - theta2) * theta1_dot * theta2_dot
            + self.h6 * theta2_dot**2
        )
        PE = self.h7 * (np.cos(theta1) - 1) + self.h8 * (np.cos(theta2) - 1)
        E  = KE + PE

        # Only pump when below target energy — clamp off when E >= 0 to prevent over-spinning.
        # When the pendulum already has enough energy to reach the top, stop pushing.
        if E < 0:
            u_swing = K_ENERGY * theta1_dot * np.cos(theta1) * (0.0 - E)
        else:
            u_swing = 0.0

        u_cart  = -K_CART_P * x - K_CART_D * x_dot   # soft spring to keep cart near center
        return u_swing + u_cart

    # -------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------
    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            try:
                next_time = time.perf_counter()
                while viewer.is_running():
                    arm_angles     = self.cart.get_arm_angles()
                    arm_velocities = self.cart.get_arm_velocities()
                    cart_position  = self.cart.get_cart_position()
                    cart_velocity  = self.cart.get_cart_velocity()

                    u = self.controls(cart_position, cart_velocity,
                                      arm_angles, arm_velocities)

                    self.cart.apply_force(u)
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()

                    self.sim_time += self.dt
                    next_time     += self.dt
                    sleep_time = next_time - time.perf_counter()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        next_time = time.perf_counter()

            except KeyboardInterrupt:
                print("Simulation stopped.")
            finally:
                self.log_file.close()
                print("Log saved to sim_log.csv")

    # -------------------------------------------------------------------
    # CONTROL DISPATCH
    # -------------------------------------------------------------------
    def controls(self, cart_position, cart_velocity, arm_angles, arm_velocities):
        # CartManipulator now returns radians directly — no conversion needed
        x          = float(np.asarray(cart_position).flat[0])
        x_dot      = float(np.asarray(cart_velocity).flat[0])
        theta1     = float(np.asarray(arm_angles).flat[0])
        theta2     = float(np.asarray(arm_angles).flat[1])
        theta1_dot = float(np.asarray(arm_velocities).flat[0])
        theta2_dot = float(np.asarray(arm_velocities).flat[1])

        state    = {"x": x, "x_dot": x_dot,
                    "theta1": theta1, "theta1_dot": theta1_dot,
                    "theta2": theta2, "theta2_dot": theta2_dot}
        target_x = TARGETS[self.target_idx]

        if self.phase == "swingup":
            u = self.energy_swingup(x, x_dot, theta1, theta2, theta1_dot, theta2_dot)

            # Hand off to LQR once both links are near the top
            if abs(theta1) < THRESHOLD and abs(theta2) < THRESHOLD:
                print(f"[t={self.sim_time:.2f}s] Switching to LQR | "
                      f"theta1={np.degrees(theta1):.1f}° "
                      f"theta2={np.degrees(theta2):.1f}°")
                self.phase = "lqr"

        else:  # "lqr"
            u = self.lqr_control(state, target_x)

            # Once stable at current target, advance to the next one
            if (abs(x - target_x) < 0.05 and abs(theta1) < 0.05 and
                    abs(theta2) < 0.05 and abs(x_dot) < 0.05):
                if self.target_idx < len(TARGETS) - 1:
                    self.target_idx += 1
                    print(f"[t={self.sim_time:.2f}s] Moving to endpoint "
                          f"{self.target_idx + 1}: x = {TARGETS[self.target_idx]}")

        u = float(np.clip(u, U_MIN, U_MAX))

        # --- Log every N steps ---
        self.step_count += 1
        if self.step_count % self.log_every == 0:
            self.logger.writerow([
                f"{self.sim_time:.4f}", self.phase, f"{target_x:.2f}",
                f"{x:.4f}", f"{x_dot:.4f}",
                f"{np.degrees(theta1):.2f}", f"{np.degrees(theta2):.2f}",
                f"{theta1_dot:.4f}", f"{theta2_dot:.4f}",
                f"{u:.4f}"
            ])

        if self.step_count % 50 == 0:
            print(f"t={self.sim_time:6.2f}s | phase={self.phase:10s} | "
                  f"x={x:6.3f} | θ1={np.degrees(theta1):7.2f}° | "
                  f"θ2={np.degrees(theta2):7.2f}° | u={u:7.3f}")

        return u

    # -------------------------------------------------------------------
    # LQR CONTROLLER
    # -------------------------------------------------------------------
    def lqr_control(self, state, target_x):
        """
        u = -K * error_state

        error_state = current state minus the upright equilibrium,
        with x offset by the current target position.
        """
        x_vec = np.array([
            [float(state['x'])          - float(target_x)],
            [float(state['x_dot'])                       ],
            [float(state['theta1'])                      ],
            [float(state['theta1_dot'])                  ],
            [float(state['theta2'])                      ],
            [float(state['theta2_dot'])                  ]
        ], dtype=float)
        return float(-(self.K @ x_vec).item())


if __name__ == "__main__":
    runner = MuJoCoRunner()
    runner.run()