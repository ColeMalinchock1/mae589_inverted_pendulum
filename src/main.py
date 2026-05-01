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
    TARGETS, THRESHOLD,
    K_ENERGY, K_CART_P, K_CART_D,
    LOG_RATE,
    x0, x_dot0, q10, q1dot0, q20, q2dot0 # INITIAL STATES
)


class MuJoCoRunner():
    """ Class for running MuJoCo and the controller """

    def __init__(self):
        """ Initialization sequence of the model """

        # Creates the model from the xml
        self.model = mujoco.MjModel.from_xml_path("../world/scene.xml")
        
        # Gets the data of the model
        self.data = mujoco.MjData(self.model)
        
        # Creates the cart object
        self.cart = CartManipulator(self.model, self.data)

        # Gets the timestep for the model
        self.dt = self.model.opt.timestep

        # --- Logging setup ---
        self.log_file = open("sim_log.csv", "w", newline="")
        self.logger = csv.writer(self.log_file)
        self.logger.writerow([
            "t", "phase", "target_x",
            "x", "x_dot", "theta1_deg", "theta2_deg",
            "theta1_dot", "theta2_dot", "u", "ke", "pe"
        ])
        self.log_every = 1/LOG_RATE
        self.step_count = 0

        # Set initial state (pendulums hanging down)
        self.data.qpos[self.model.joint('slider').qposadr[0]] = x0
        self.data.qpos[self.model.joint('hinge1').qposadr[0]] = q10  # = pi
        self.data.qpos[self.model.joint('hinge2').qposadr[0]] = q20  # = pi
        self.data.qvel[self.model.joint('slider').dofadr[0]]  = x_dot0
        self.data.qvel[self.model.joint('hinge1').dofadr[0]]  = q1dot0
        self.data.qvel[self.model.joint('hinge2').dofadr[0]]  = q2dot0
        mujoco.mj_forward(self.model, self.data)

        # Complete the setup of the 
        self.setup()


    def setup(self):
        """ Precompute the h parameters """

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
        bc = M_inv @ np.array([1,  0,  0])

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

        # Print the LQR k
        print("LQR gain K computed:")
        print(self.K)
        
        # Initialize the values for the simulator
        self.sim_time = 0.0
        self.phase = "swingup"
        self.target_idx = 0
        self.pe = self.ke = 0.0


    def run(self):
        """ Main loop to be ran for the simulation and controls """

        # Launch the MuJoCo simulator viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:

            try:

                # Get the timer
                next_time = time.perf_counter()
                while viewer.is_running():

                    # Get the current values of the cart
                    arm_angles = self.cart.get_arm_angles()
                    arm_velocities = self.cart.get_arm_velocities()
                    cart_position = self.cart.get_cart_position()
                    cart_velocity = self.cart.get_cart_velocity()

                    # Compute the controller input on the cart
                    u = self.controls(cart_position, cart_velocity, arm_angles, arm_velocities)

                    # Apply the input on the cart
                    self.cart.apply_force(u)
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()

                    # Increment the sim time
                    self.sim_time += self.dt
                    next_time += self.dt

                    # Check if the system needs to sleep
                    sleep_time = next_time - time.perf_counter()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        next_time = time.perf_counter()

            # Check for keyboard interrupt
            except KeyboardInterrupt:
                print("Simulation stopped.")

            # Log the results
            finally:
                self.log_file.close()
                print("Log saved to sim_log.csv")


    def controls(self, cart_position:float, cart_velocity:float, arm_angles:tuple[float, float], arm_velocities:tuple[float, float]) -> float:
        """ Performs the control algorithms to output u, the input into the system

        Args:
            cart_position (float): The cart x position
            cart_velocity (float): The cart y position
            arm_angles (tuple[float, float]): The arm angles
            arm_velocities (tuple[float, float]): The arm angular velocities

        Returns:
            float: The input u for the cart
        """

        # Get the current values of the cart
        x = float(np.asarray(cart_position).flat[0])
        x_dot = float(np.asarray(cart_velocity).flat[0])
        theta1 = float(np.asarray(arm_angles).flat[0])
        theta2 = float(np.asarray(arm_angles).flat[1])
        theta1_dot = float(np.asarray(arm_velocities).flat[0])
        theta2_dot = float(np.asarray(arm_velocities).flat[1])

        # Create the current state of the cart
        state = {"x": x, "x_dot": x_dot,
                "theta1": theta1, "theta1_dot": theta1_dot,
                "theta2": theta2, "theta2_dot": theta2_dot}
        
        # Get the target x position
        target_x = TARGETS[self.target_idx]

        # Check what the current phase is
        if self.phase == "swingup":

            # Get the energy to swing up
            u = self.energy_swingup(x, x_dot, theta1, theta2, theta1_dot, theta2_dot)

            # Hand off to LQR once both links are near the top
            if abs(theta1) < THRESHOLD and abs(theta2) < THRESHOLD:
                print(f"[t={self.sim_time:.2f}s] Switching to LQR | "
                      f"theta1={np.degrees(theta1):.1f}° "
                      f"theta2={np.degrees(theta2):.1f}°")
                
                # Change the phase to lqr
                self.phase = "lqr"

        else:

            # Get the lqr input
            u = self.lqr_control(state, target_x)

            # Once stable at current target, advance to the next one
            if (abs(x - target_x) < 0.05 and abs(theta1) < 0.05 and
                    abs(theta2) < 0.05 and abs(x_dot) < 0.05):
                
                # Increment to the next target if there is another one
                if self.target_idx < len(TARGETS) - 1:
                    self.target_idx += 1
                    print(f"[t={self.sim_time:.2f}s] Moving to endpoint "
                          f"{self.target_idx + 1}: x = {TARGETS[self.target_idx]}")

        # Make sure the u is between the min and max
        u = float(np.clip(u, U_MIN, U_MAX))

        # Log every step
        self.step_count += 1
        if self.step_count % self.log_every == 0:
            self.logger.writerow([
                f"{self.sim_time:.4f}", self.phase, f"{target_x:.2f}",
                f"{x:.4f}", f"{x_dot:.4f}",
                f"{np.degrees(theta1):.2f}", f"{np.degrees(theta2):.2f}",
                f"{theta1_dot:.4f}", f"{theta2_dot:.4f}",
                f"{u:.4f}",
                f"{self.ke:.4f}",
                f"{self.pe:.4f}"
            ])

        # Print every 50
        if self.step_count % 50 == 0:
            print(f"t={self.sim_time:6.2f}s | phase={self.phase:10s} | "
                  f"x={x:6.3f} | θ1={np.degrees(theta1):7.2f}° | "
                  f"θ2={np.degrees(theta2):7.2f}° | u={u:7.3f}")

        return u
    

    def lqr_control(self, state:dict, target_x:float) -> float:
        """ The LQR controller on the cart to hold the double pendulum upwards

        Args:
            state (dict): The current state of the cart and arms
            target_x (float): The target x position

        Returns:
            float: The input value for the cart
        """

        # The x vector
        x_vec = np.array([
            [float(state['x']) - float(target_x)],
            [float(state['x_dot'])],
            [float(state['theta1'])],
            [float(state['theta1_dot'])],
            [float(state['theta2'])],
            [float(state['theta2_dot'])]
        ], dtype=float)


        return float(-(self.K @ x_vec).item())
    

    def energy_swingup(self, x:float, x_dot:float, theta1:float, theta2:float, theta1_dot:float, theta2_dot:float) -> float:
        """ Pumps mechanical energy into the two-link pendulum until it has enough to reach the upright position, then lets LQR take over.

        Args:
            x (float): Cart x position
            x_dot (float): Cart x velocity
            theta1 (float): Arm 1 angle
            theta2 (float): Arm 2 angle
            theta1_dot (float): Arm 1 angular velocity
            theta2_dot (float): Arm 2 angular velocity

        Returns:
            float: The swing up force input
        """

        # The total kinetic energy of the system
        KE = 0.5 * (
            self.h4 * theta1_dot**2
            + 2 * self.h5 * np.cos(theta1 - theta2) * theta1_dot * theta2_dot
            + self.h6 * theta2_dot**2
        )

        # The total potential energy of the system
        PE = self.h7 * (np.cos(theta1) - 1) + self.h8 * (np.cos(theta2) - 1)

        # The total energy of the system
        E  = KE + PE

        self.ke = KE
        self.pe = PE

        # Check that the energy is not too high and over-spinning
        if E < 0:

            # Calculate the input from the kinetic energy and angular values of the first arm
            u_swing = K_ENERGY * theta1_dot * np.cos(theta1) * (-E)
        else:
            u_swing = 0.0

        # Get the cart force input from the displacement and velocity
        u_cart  = -K_CART_P * x - K_CART_D * x_dot

        return u_swing + u_cart


if __name__ == "__main__":
    
    # Create the MuJoCo Runner
    runner = MuJoCoRunner()

    # Run the system
    runner.run()