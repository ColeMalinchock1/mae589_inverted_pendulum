"""

Script to simulate a double inverted pendulum to be swung up and to reach a target x.
Uses LQR to swing up and maintain holding the upward position of the double pendulum.

Inspired from the work fromTheodore Bounds: https://apmonitor.com/do/index.php/Main/DoubleInvertedPendulum

@author Cole Malinchock (malinchc@gmail.com)
@date 4/30/2026

"""

# Import necessary libraries
import time
import numpy as np
from scipy import linalg
import mujoco
import mujoco.viewer
import csv

# Import custom libraries
from lib.cart_manipulator import CartManipulator
from lib.constants import (
    G,
    U_MIN, U_MAX,
    MC, M1, M2, L1, LC1, LC2, I1, I2,
    TARGET_X_1, TARGET_X_2, THRESHOLD,
    K_ENERGY, K_CART_P, K_CART_D,
    LOG_RATE,
    x0_1, x0_2, x_dot0, q10, q1_dot0, q20_1, q20_2, q2_dot0 # INITIAL STATES
)

# Minimum center-to-center separation before repulsion kicks in.
# Cart half-width is 0.5 m, so full width = 1.0 m; 1.2 m gives a 0.2 m buffer.
SAFE_DIST = 1.2   # [m]
K_REPULSE = 30.0  # repulsive gain [N/m]


class MuJoCoRunner():
    """ Class for running MuJoCo and the controller """

    def __init__(self):
        """ Initialization sequence of the model """

        # Creates the model from the xml
        self.model = mujoco.MjModel.from_xml_path("../world/scene.xml")
        print(self.model)

        # Gets the data of the model
        self.data = mujoco.MjData(self.model)

        # Creates the cart object
        self.carts = CartManipulator(self.model, self.data)

        # Gets the timestep for the model
        self.dt = self.model.opt.timestep

        # --- Logging setup ---
        self.log_file = open("sim_log.csv", "w", newline="")
        self.logger = csv.writer(self.log_file)
        self.logger.writerow([
            "cart_id", "t", "phase", "target_x",
            "x", "x_dot", "theta1_deg", "theta2_deg",
            "theta1_dot", "theta2_dot", "u"
        ])
        self.log_every = int(1 / LOG_RATE)
        self.step_count = 0

        # Set initial states for both carts
        self.carts.set_state(x0_1, x_dot0, q10, q20_1, q1_dot0, q2_dot0, cart_id=1)
        self.carts.set_state(x0_2, x_dot0, q10, q20_2, q1_dot0, q2_dot0, cart_id=2)

        mujoco.mj_forward(self.model, self.data)

        # Complete the setup
        self.setup()


    def setup(self):
        """ Precompute the h parameters and LQR gain """

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

        # Build linearized A, B matrices around the upright equilibrium
        M_lin = np.array([[h1, h2, h3],
                          [h2, h4, h5],
                          [h3, h5, h6]])
        M_inv = np.linalg.inv(M_lin)

        # Column vectors: how gravity and input map through the inertia
        t1c = M_inv @ np.array([0, h7, 0])
        t2c = M_inv @ np.array([0,  0, h8])
        bc  = M_inv @ np.array([1,  0,  0])

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

        # Solve the LQR Riccati equation to get the optimal gain K.
        # Q penalises state errors; R penalises control effort.
        Q = np.diag([10.0, 1.0, 100.0, 1.0, 100.0, 1.0])
        R = np.array([[1.0]])
        P = linalg.solve_continuous_are(self.A, self.B, Q, R)
        self.K = linalg.inv(R) @ self.B.T @ P

        print("LQR gain K computed:")
        print(self.K)

        # Initialize the simulator state
        self.sim_time = 0.0
        self.phase_1  = "swingup"
        self.phase_2  = "swingup"


    def run(self):
        """ Main loop to be ran for the simulation and controls """

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:

            try:

                next_time = time.perf_counter()
                while viewer.is_running():

                    # Get the current values of both carts
                    cart_1_arm_angles     = self.carts.get_arm_angles(cart_id=1)
                    cart_1_arm_velocities = self.carts.get_arm_velocities(cart_id=1)
                    cart_1_position       = self.carts.get_cart_position(cart_id=1)
                    cart_1_velocity       = self.carts.get_cart_velocity(cart_id=1)

                    cart_2_arm_angles     = self.carts.get_arm_angles(cart_id=2)
                    cart_2_arm_velocities = self.carts.get_arm_velocities(cart_id=2)
                    cart_2_position       = self.carts.get_cart_position(cart_id=2)
                    cart_2_velocity       = self.carts.get_cart_velocity(cart_id=2)

                    # Compute controller inputs for both carts
                    u1, u2 = self.controls(
                        cart_1_position, cart_1_velocity, cart_1_arm_angles, cart_1_arm_velocities,
                        cart_2_position, cart_2_velocity, cart_2_arm_angles, cart_2_arm_velocities
                    )

                    # Apply forces and step the simulation
                    self.carts.apply_force(u1, u2)
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()

                    # Increment the sim time
                    self.sim_time += self.dt
                    next_time     += self.dt

                    # Real-time pacing
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


    def controls(self,
                 cart_1_position: float,   cart_1_velocity: float,
                 cart_1_arm_angles: tuple, cart_1_arm_velocities: tuple,
                 cart_2_position: float,   cart_2_velocity: float,
                 cart_2_arm_angles: tuple, cart_2_arm_velocities: tuple
                 ) -> tuple[float, float]:
        """ Performs the control algorithms to output u1 and u2

        Args:
            cart_1_position (float): Cart 1 x position
            cart_1_velocity (float): Cart 1 x velocity
            cart_1_arm_angles (tuple): Cart 1 arm angles (theta1, theta2)
            cart_1_arm_velocities (tuple): Cart 1 arm angular velocities
            cart_2_position (float): Cart 2 x position
            cart_2_velocity (float): Cart 2 x velocity
            cart_2_arm_angles (tuple): Cart 2 arm angles (theta1, theta2)
            cart_2_arm_velocities (tuple): Cart 2 arm angular velocities

        Returns:
            tuple[float, float]: Control inputs (u1, u2) for cart 1 and cart 2
        """

        # Unpack scalars
        x_1          = float(np.asarray(cart_1_position).flat[0])
        x_dot_1      = float(np.asarray(cart_1_velocity).flat[0])
        theta1_1     = float(np.asarray(cart_1_arm_angles).flat[0])
        theta2_1     = float(np.asarray(cart_1_arm_angles).flat[1])
        theta1_dot_1 = float(np.asarray(cart_1_arm_velocities).flat[0])
        theta2_dot_1 = float(np.asarray(cart_1_arm_velocities).flat[1])

        x_2          = float(np.asarray(cart_2_position).flat[0])
        x_dot_2      = float(np.asarray(cart_2_velocity).flat[0])
        theta1_2     = float(np.asarray(cart_2_arm_angles).flat[0])
        theta2_2     = float(np.asarray(cart_2_arm_angles).flat[1])
        theta1_dot_2 = float(np.asarray(cart_2_arm_velocities).flat[0])
        theta2_dot_2 = float(np.asarray(cart_2_arm_velocities).flat[1])

        # Build state dicts
        state_1 = {"x": x_1, "x_dot": x_dot_1,
                   "theta1": theta1_1, "theta1_dot": theta1_dot_1,
                   "theta2": theta2_1, "theta2_dot": theta2_dot_1}

        state_2 = {"x": x_2, "x_dot": x_dot_2,
                   "theta1": theta1_2, "theta1_dot": theta1_dot_2,
                   "theta2": theta2_2, "theta2_dot": theta2_dot_2}

        target_x_1 = TARGET_X_1
        target_x_2 = TARGET_X_2

        # --- Cart 1 control (handled independently of cart 2's phase) ---
        if self.phase_1 == "swingup":
            u_1 = self.energy_swingup(state_1)
            if abs(theta1_1) < THRESHOLD and abs(theta2_1) < THRESHOLD:
                print(f"[t={self.sim_time:.2f}s] Cart 1 switching to LQR | "
                      f"theta1={np.degrees(theta1_1):.1f}° "
                      f"theta2={np.degrees(theta2_1):.1f}°")
                self.phase_1 = "lqr"
        else:
            u_1 = self.lqr_control(state_1, target_x_1)

        # --- Cart 2 control (handled independently of cart 1's phase) ---
        if self.phase_2 == "swingup":
            u_2 = self.energy_swingup(state_2)
            if abs(theta1_2) < THRESHOLD and abs(theta2_2) < THRESHOLD:
                print(f"[t={self.sim_time:.2f}s] Cart 2 switching to LQR | "
                      f"theta1={np.degrees(theta1_2):.1f}° "
                      f"theta2={np.degrees(theta2_2):.1f}°")
                self.phase_2 = "lqr"
        else:
            u_2 = self.lqr_control(state_2, target_x_2)

        # --- Collision avoidance (applied regardless of phase) ---
        u_1, u_2 = self._collision_avoidance(x_1, x_dot_1, u_1, x_2, x_dot_2, u_2)

        # Clip to actuator limits
        u_1 = float(np.clip(u_1, U_MIN, U_MAX))
        u_2 = float(np.clip(u_2, U_MIN, U_MAX))

        # --- Logging ---
        self.step_count += 1
        if self.step_count % self.log_every == 0:
            self.logger.writerow([
                1, f"{self.sim_time:.4f}", self.phase_1, f"{target_x_1:.2f}",
                f"{x_1:.4f}", f"{x_dot_1:.4f}",
                f"{np.degrees(theta1_1):.2f}", f"{np.degrees(theta2_1):.2f}",
                f"{theta1_dot_1:.4f}", f"{theta2_dot_1:.4f}",
                f"{u_1:.4f}"
            ])
            self.logger.writerow([
                2, f"{self.sim_time:.4f}", self.phase_2, f"{target_x_2:.2f}",
                f"{x_2:.4f}", f"{x_dot_2:.4f}",
                f"{np.degrees(theta1_2):.2f}", f"{np.degrees(theta2_2):.2f}",
                f"{theta1_dot_2:.4f}", f"{theta2_dot_2:.4f}",
                f"{u_2:.4f}"
            ])

        # Print every 50 steps
        if self.step_count % 50 == 0:
            print(f"t={self.sim_time:6.2f}s | "
                  f"C1 [{self.phase_1:7s}] x={x_1:6.3f} θ1={np.degrees(theta1_1):7.2f}° θ2={np.degrees(theta2_1):7.2f}° u={u_1:7.3f} | "
                  f"C2 [{self.phase_2:7s}] x={x_2:6.3f} θ1={np.degrees(theta1_2):7.2f}° θ2={np.degrees(theta2_2):7.2f}° u={u_2:7.3f}")

        return u_1, u_2


    def lqr_control(self, state: dict, target_x: float) -> float:
        """ The LQR controller on the cart to hold the double pendulum upwards

        Args:
            state (dict): The current state of the cart and arms
            target_x (float): The target x position

        Returns:
            float: The input value for the cart
        """

        x_vec = np.array([
            [float(state['x'])          - float(target_x)],
            [float(state['x_dot'])],
            [float(state['theta1'])],
            [float(state['theta1_dot'])],
            [float(state['theta2'])],
            [float(state['theta2_dot'])]
        ], dtype=float)

        return float(-(self.K @ x_vec).item())


    def energy_swingup(self, state: dict) -> float:
        """ Pumps mechanical energy into the two-link pendulum until it has enough
        to reach the upright position, then lets LQR take over.

        Args:
            state (dict): The current state of the cart and arms

        Returns:
            float: The swing-up force input
        """

        x          = state['x']
        x_dot      = state['x_dot']
        theta1     = state['theta1']
        theta2     = state['theta2']
        theta1_dot = state['theta1_dot']
        theta2_dot = state['theta2_dot']

        # Total kinetic energy of the two-link system
        KE = 0.5 * (
            self.h4 * theta1_dot**2
            + 2 * self.h5 * np.cos(theta1 - theta2) * theta1_dot * theta2_dot
            + self.h6 * theta2_dot**2
        )

        # Total potential energy (zero reference at hanging-down equilibrium)
        PE = self.h7 * (np.cos(theta1) - 1) + self.h8 * (np.cos(theta2) - 1)

        E = KE + PE

        # Pump energy in when the system is below the target energy level
        if E < 0:
            u_swing = K_ENERGY * theta1_dot * np.cos(theta1) * (-E)
        else:
            u_swing = 0.0

        # PD term to keep the cart near the origin during swing-up
        u_cart = -K_CART_P * x - K_CART_D * x_dot

        return u_swing + u_cart


    def _collision_avoidance(self,
                              x_1: float, x_dot_1: float, u_1: float,
                              x_2: float, x_dot_2: float, u_2: float
                              ) -> tuple[float, float]:
        """ Adds a repulsive force to each cart when they get too close.

        When the separation drops below SAFE_DIST, a force proportional to the
        overlap is added to push both carts away from each other.

        Args:
            x_1 (float): Cart 1 position
            x_dot_1 (float): Cart 1 velocity (unused, reserved for damping extension)
            u_1 (float): Cart 1 control input before avoidance
            x_2 (float): Cart 2 position
            x_dot_2 (float): Cart 2 velocity (unused, reserved for damping extension)
            u_2 (float): Cart 2 control input before avoidance

        Returns:
            tuple[float, float]: Adjusted (u_1, u_2) with repulsion applied
        """

        separation = x_1 - x_2   # positive when cart 1 is to the right of cart 2
        dist = abs(separation)

        if dist < SAFE_DIST:
            overlap   = SAFE_DIST - dist
            f_repulse = K_REPULSE * overlap

            if separation >= 0:
                # Cart 1 is right of cart 2 — push cart 1 further right, cart 2 further left
                u_1 += f_repulse
                u_2 -= f_repulse
            else:
                # Cart 1 is left of cart 2 — push cart 1 further left, cart 2 further right
                u_1 -= f_repulse
                u_2 += f_repulse

        return u_1, u_2


if __name__ == "__main__":

    # Create the MuJoCo Runner
    runner = MuJoCoRunner()

    # Run the system
    runner.run()
