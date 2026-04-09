class CartManipulator:
    def __init__(self, model=None, data=None):
        self.model = model
        self.data = data

        self.cart_id = self.model.body('cart').id

    def apply_force(self, force):
        # Apply horizontal force to the center of mass of the cart
        
        self.data.xfrc_applied[self.cart_id, 0] = force
        print("Applying force:", force)