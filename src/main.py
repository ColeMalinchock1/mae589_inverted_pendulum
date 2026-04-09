import time
import mujoco
import mujoco.viewer

from cart_manipulator import CartManipulator

model = mujoco.MjModel.from_xml_path("../world/scene.xml")
data = mujoco.MjData(model)
cart = CartManipulator(model, data)

dt = model.opt.timestep  # simulation timestep

with mujoco.viewer.launch_passive(model, data) as viewer:
    try:
        next_time = time.perf_counter()

        while viewer.is_running():
            # controller computes command for this step
            cart.apply_force(100.0)

            # advance simulation exactly one step
            mujoco.mj_step(model, data)

            # render occasionally
            viewer.sync()

            # pace to real time
            next_time += dt
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # fell behind, reset clock so lag doesn't accumulate forever
                next_time = time.perf_counter()

    except KeyboardInterrupt:
        print("Simulation stopped.")