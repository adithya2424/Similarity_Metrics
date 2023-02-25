import nrrd
import matplotlib.pyplot as plt
import numpy as np

# To change slices using key presses, it is a good idea to create a class that will have img and slc as
# member variables and the callback function as a member function to access and modify those variables.
class displayvolume:
    def __init__(self, img, voxsz, direction=None, slc=None, contrast=None, level=None, interpolation='bilinear'):
        self.img = img
        self.voxsz = voxsz
        self.direction = direction
        self.slc = slc
        # set to direction 2 for default
        if self.direction is None:
            self.direction = 2
        else:
            self.direction = direction

        if contrast is not None:
            self.contrast = contrast
        else:
            self.contrast = 1000

        if level is not None:
            self.level = level
        else:
            self.level = 0

        self.interpolation = interpolation
        # set to middle slc for default
        if self.slc is None:
            self.slc = int((np.shape(self.img)[self.direction]) / 2)

        self.fig, self.ax = plt.subplots()
        plt.connect('key_press_event', self.on_key_press)
        plt.ion()
        self.display(slc)

    def display(self, slc=None, contrast=None, level=None):
        if contrast is not None:
            self.contrast = contrast

        if level is not None:
            self.level = level

        if slc is not None:
            self.slc = slc

        # Display sagittal slice 'slc' in grayscale with custom window/level and interpolation.
        # Use transpose so that 'x' is posterior-to-anterior and 'y' is inferior-to-superior
        if self.direction == 0:
            self.ax.imshow(self.img[self.slc, :, :].T, 'gray', vmin=self.level - self.contrast / 2, vmax=self.level + self.contrast / 2,
                      interpolation=self.interpolation)
            plt.xlabel('y')
            plt.ylabel('z')
            plt.title(f'Sagittal slice: Contrast={self.contrast:.1f}, Level={self.level:.1f}, x = {self.slc}')
            self.ax.set_xlim(left=0, right=np.shape(self.img)[1] - 1)
            self.ax.set_ylim(bottom=0, top=np.shape(self.img)[2] - 1)
            self.ax.set_aspect(self.voxsz[2] / self.voxsz[1])
            plt.show()

        if self.direction == 1:
            self.ax.imshow(self.img[:, self.slc, :].T, 'gray', vmin=self.level - self.contrast / 2,
                           vmax=self.level + self.contrast / 2,
                           interpolation=self.interpolation)
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title(f'Coronal slice: Contrast={self.contrast:.1f}, Level={self.level:.1f}, x = {self.slc}')
            self.ax.set_xlim(left=0, right=np.shape(self.img)[0] - 1)
            self.ax.set_ylim(bottom=0, top=np.shape(self.img)[2] - 1)
            self.ax.set_aspect(self.voxsz[2] / self.voxsz[0])
            plt.show()

        if self.direction == 2:
            self.ax.imshow(self.img[:, :, self.slc].T, 'gray', vmin=self.level - self.contrast / 2,
                           vmax=self.level + self.contrast / 2,
                           interpolation=self.interpolation)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Axial slice: Contrast={self.contrast:.1f}, Level={self.level:.1f}, x = {self.slc}')
            self.ax.set_aspect(self.voxsz[1] / self.voxsz[0])
            plt.show()

    def on_key_press(self, event):
        if (event.key == 'a' or event.key == 'up') and self.slc < np.shape(self.img)[self.direction]-1:
            self.slc += 1
        elif (event.key == 'z' or event.key == 'down') and self.slc > 0:
            self.slc -= 1
        elif event.key == 'd':
            self.level += 0.1 * self.contrast
        elif event.key == 'x':
            self.level -= 0.1 * self.contrast
        elif event.key == 'c':
            self.contrast *= 1.1
        elif event.key == 'v':
            self.contrast /= 1.1
        else:
            return

        self.display()


# # load a CT image to play with
# img, imgh = nrrd.read('/Users/adithyapamulaparthy/Desktop/Courses_Spring2023/MedicalImageSegmentation/0522c0001/img.nrrd')
# voxsz = [imgh['space directions'][0][0], imgh['space directions'][1][1], imgh['space directions'][2][2]]
#
#
# d = displayvolume(img, voxsz, 2)
# d.display(20, 900, 0)
#
#
# while (1):
#     d.fig.canvas.draw_idle()
#     d.fig.canvas.start_event_loop(0.3)