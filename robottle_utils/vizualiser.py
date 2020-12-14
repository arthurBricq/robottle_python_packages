import matplotlib.pyplot as plt

class ImageVizualiser():
    def __init__(self):
        fig = plt.figure(figsize = (5,5))
        self.ax = fig.gca()

    def display(self, img):
        plt.pause(0.001)
        self.ax.imshow(img)
        plt.draw()
        plt.pause(0.01)
        


