from lib.constants import PI

class CartManipulator:


    def __init__(self, model=None, data=None):
        self.model = model
        self.data = data

        self.cart_1_id = self.model.body('cart_1').id
        self.arm1_1_id = self.model.body('arm1_1').id
        self.mass1_1_id = self.model.body('mass1_1').id
        self.arm2_1_id = self.model.body('arm2_1').id
        self.mass2_1_id = self.model.body('mass2_1').id

        self.cart_2_id = self.model.body('cart_2').id
        self.arm1_2_id = self.model.body('arm1_2').id
        self.mass1_2_id = self.model.body('mass1_2').id
        self.arm2_2_id = self.model.body('arm2_2').id
        self.mass2_2_id = self.model.body('mass2_2').id


    def set_state(self, x0:float, x_dot0:float, q10:float, q20:float, q1_dot0:float, q2_dot0:float, cart_id:int=None):
        
        if cart_id is None:
            cart_id = [1, 2]
        else:
            cart_id = [cart_id]
        
        for id in cart_id:
            self.data.qpos[self.model.joint(f'slider_{id}').qposadr[0]] = x0
            self.data.qpos[self.model.joint(f'hinge1_{id}').qposadr[0]] = q10
            self.data.qpos[self.model.joint(f'hinge2_{id}').qposadr[0]] = q20
            self.data.qvel[self.model.joint(f'slider_{id}').dofadr[0]] = x_dot0
            self.data.qvel[self.model.joint(f'hinge1_{id}').dofadr[0]] = q1_dot0
            self.data.qvel[self.model.joint(f'hinge2_{id}').dofadr[0]] = q2_dot0


    def apply_force(self, force_1:float, force_2:float):

        # Apply horizontal force to the center of mass of the cart
        self.data.xfrc_applied[self.cart_1_id, 0] = force_1
        self.data.xfrc_applied[self.cart_2_id, 0] = force_2

    
    def get_cart_position(self, cart_id:int):

        # Get the position of the cart
        if cart_id == 1:
            return self.data.xpos[self.cart_1_id]
        elif cart_id == 2:
            return self.data.xpos[self.cart_2_id]
        else:
            print("Cart ID must be 1 or 2")
            return None


    def get_cart_velocity(self, cart_id:int):

        # Get the velocity of the cart
        if cart_id == 1:
            return self.data.qvel[self.model.joint('slider_1').dofadr]
        elif cart_id == 2:
            return self.data.qvel[self.model.joint('slider_2').dofadr]
        else:
            print("Cart ID must be 1 or 2")
            return None


    def get_arm_angles(self, cart_id:int):
        # Get the angles of the two arms
        if cart_id == 1:
            theta1_rad = self.data.qpos[self.model.joint('hinge1_1').qposadr]
            theta2_rad = self.data.qpos[self.model.joint('hinge2_1').qposadr]
        elif cart_id == 2:
            theta1_rad = self.data.qpos[self.model.joint('hinge1_2').qposadr]
            theta2_rad = self.data.qpos[self.model.joint('hinge2_2').qposadr]
        else:
            print("Cart ID must be 1 or 2")
            return None

        theta1_rad, theta2_rad = self._rad_bounding(theta1_rad, theta2_rad + theta1_rad)

        return (theta1_rad, theta2_rad)
    

    def get_arm_velocities(self, cart_id:int):
        # Get the angular velocities of the two arms
        if cart_id == 1:
            theta1_vel_rad = self.data.qvel[self.model.joint('hinge1_1').dofadr]
            theta2_vel_rad = self.data.qvel[self.model.joint('hinge2_1').dofadr]
        elif cart_id == 2:
            theta1_vel_rad = self.data.qvel[self.model.joint('hinge1_2').dofadr]
            theta2_vel_rad = self.data.qvel[self.model.joint('hinge2_2').dofadr]
        else:
            print("Cart ID must be 1 or 2")
            return None

        theta1_vel_rad, theta2_vel_rad = self._rad_bounding(theta1_vel_rad, theta2_vel_rad + theta1_vel_rad)

        return (theta1_vel_rad, theta2_vel_rad)
    

    def _rad_bounding(self, theta1_rad, theta2_rad):
        # Convert radians to degrees
        theta1_rad_bounded = (theta1_rad[0] + PI) % (2 * PI) - PI
        theta2_rad_bounded = (theta2_rad[0] + PI) % (2 * PI) - PI

        return theta1_rad_bounded, theta2_rad_bounded