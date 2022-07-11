

# class Config(DefaultConfig):
class Config():

    def __init__(self):
        # super(Config, self).__init__()

        self.IMAGE_WIDTH  = 60
        self.IMAGE_HEIGHT = 160
        self.INPUT_SIZE   = [self.IMAGE_HEIGHT, self.IMAGE_WIDTH]  # HxW
