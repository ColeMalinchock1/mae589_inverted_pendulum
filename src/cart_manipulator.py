from lib.constants import PI

class CartManipulator:
    def __init__(self, model=None, data=None):
        self.model = model
        self.data = data

        self.cart_id = self.model.body('cart').id
        self.arm1_id = self.model.body('arm1').id
        self.mass1_id = self.model.body('mass1').id
        self.arm2_id = self.model.body('arm2').id
        self.mass2_id = self.model.body('mass2').id


    def apply_force(self, force):
        # Apply horizontal force to the center of mass of the cart

        self.data.xfrc_applied[self.cart_id, 0] = force

    
    def get_cart_position(self):

        # Get the position of the cart
        return self.data.xpos[self.cart_id]


    def get_cart_velocity(self):

        # Get the velocity of the cart
        return self.data.qvel[self.model.joint('slider').dofadr]


    def get_arm_angles(self):
        # Get the angles of the two arms

        theta1_rad = self.data.qpos[self.model.joint('hinge1').qposadr]
        theta2_rad = self.data.qpos[self.model.joint('hinge2').qposadr]

        theta1_rad, theta2_rad = self._rad_bounding(theta1_rad, theta2_rad + theta1_rad)

        return (theta1_rad, theta2_rad)
    

    def get_arm_velocities(self):
        # Get the angular velocities of the two arms

        theta1_vel_rad = self.data.qvel[self.model.joint('hinge1').dofadr]
        theta2_vel_rad = self.data.qvel[self.model.joint('hinge2').dofadr]

        theta1_vel_rad, theta2_vel_rad = self._rad_bounding(theta1_vel_rad, theta2_vel_rad + theta1_vel_rad)

        return (theta1_vel_rad, theta2_vel_rad)
    

    def _rad_bounding(self, theta1_rad, theta2_rad):
        # Convert radians to degrees
        theta1_rad_bounded = (theta1_rad[0] + PI) % (2 * PI) - PI
        theta2_rad_bounded = (theta2_rad[0] + PI) % (2 * PI) - PI

        return theta1_rad_bounded, theta2_rad_bounded