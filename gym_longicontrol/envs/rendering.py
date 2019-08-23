import io
import numpy as np

try:
    import pyglet
except ImportError:
    raise ImportError('''
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    ''')


class Viewer(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(vsync=True, *args, **kwargs)

        self.batch = pyglet.graphics.Batch()
        self.background = pyglet.graphics.OrderedGroup(0)
        self.foreground = pyglet.graphics.OrderedGroup(1)

        self.isopen = True
        self.pause = False
        self.plot_fig = False

        self.components = {}

        self.history = {}
        pyglet.gl.glClearColor(1, 1, 1, 1)

    def render(self, return_rgb_array=False):
        self.clear()
        self.dispatch_event('on_draw')
        self.dispatch_events()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.data, dtype=np.uint8)
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.flip()
        return arr if return_rgb_array else self.isopen

    def get_array(self):
        self.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer(
        ).get_image_data()
        self.flip()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1, :, 0:3]

    def on_draw(self):
        self.batch.draw()

    def on_key_press(self, key, modifiers):
        if key == pyglet.window.key.SPACE:
            self.pause = True if self.pause is False else False
            print('Rollout is paused. Press SPACE again to continue.'
                  if self.pause is True else 'Rollout continued.')
            while self.pause:
                self.render()
        if key == pyglet.window.key.ENTER:
            self.plot_fig = True if self.plot_fig is False else False
            print('Plotting enabled. Press ENTER again to disable plotting.'
                  if self.plot_fig is True else 'Plotting disabled.')

    def on_close(self):
        self.isopen = False
        self.close()


class Image(pyglet.sprite.Sprite):
    def __init__(self,
                 filename,
                 rel_anchor_x=0.5,
                 rel_anchor_y=0.5,
                 *args,
                 **kwargs):
        image = pyglet.image.load(filename)
        image.anchor_x = int(image.width * rel_anchor_x)
        image.anchor_y = int(image.height * rel_anchor_y)
        super().__init__(image, *args, **kwargs)


class Figure(pyglet.sprite.Sprite):
    def __init__(self,
                 mlp_fig,
                 rel_anchor_x=0.5,
                 rel_anchor_y=0.5,
                 *args,
                 **kwargs):
        self.rel_anchor_x = rel_anchor_x
        self.rel_anchor_y = rel_anchor_y
        image = figure_to_image(mlp_fig, self.rel_anchor_x, self.rel_anchor_y)
        super().__init__(image, *args, **kwargs)

    @property
    def figure(self):
        return (self.image)

    @figure.setter
    def figure(self, mlp_fig):
        self.image = figure_to_image(mlp_fig, self.rel_anchor_x,
                                     self.rel_anchor_y)


def figure_to_image(mlp_fig, rel_anchor_x=0.5, rel_anchor_y=0.5):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(mlp_fig)
    pic_data = io.BytesIO()
    canvas.print_raw(pic_data, dpi=mlp_fig.dpi)
    width, height = mlp_fig.get_size_inches() * mlp_fig.dpi
    width = int(width)
    height = int(height)
    image = pyglet.image.ImageData(width, height, 'RGBA', pic_data.getvalue(),
                                   -4 * width)
    image.anchor_x = int(image.width * rel_anchor_x)
    image.anchor_y = int(image.height * rel_anchor_y)
    return image


class Label(pyglet.text.Label):
    def __init__(self, anchor_x='center', anchor_y='center', *args, **kwargs):
        super().__init__(anchor_x=anchor_x, anchor_y=anchor_y, *args, **kwargs)

    @property
    def position(self):
        return (self.x, self.y)

    @position.setter
    def position(self, xy_tuple):
        self.x = xy_tuple[0]
        self.y = xy_tuple[1]
