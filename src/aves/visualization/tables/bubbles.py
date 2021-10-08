import pymunk
from pymunk.vec2d import Vec2d
import pymunk.matplotlib_util
import numpy as np

from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import seaborn as sns


def bubble_plot(
    ax,
    df,
    position_column,
    radius_column,
    label_column=None,
    color_column=None,
    palette="plasma",
    n_bins=10,
    num_steps=50,
    x_position_scaling=800,
    min_label_size=4,
    max_label_size=64,
    starting_y_range=None,
    margin=2,
):
    df = df.reset_index()
    space = pymunk.Space()
    space.gravity = (0, 0)

    radius = np.sqrt(df[radius_column].values) + margin

    if starting_y_range is None:
        starting_y_range = int(np.sqrt(df.shape[0]))

    for idx, row in df.iterrows():
        x = row[position_column] * x_position_scaling
        y = np.random.randint(-starting_y_range, starting_y_range)
        mass = 10
        r = radius[idx]

        moment = pymunk.moment_for_circle(mass, 0, r, (0, 0))

        body = pymunk.Body(mass, moment)

        body.position = x, y
        body.start_position = Vec2d(*body.position)

        shape = pymunk.Circle(body, r)
        shape.elasticity = 0.9999999

        space.add(body, shape)

    for i in range(num_steps):
        space.step(1)

    value_range = (min(radius), max(radius))

    def scale(value):
        result = (value - value_range[0]) / (value_range[1] - value_range[0])
        return result

    ax.set_aspect("equal")

    cmap = ListedColormap(sns.color_palette(palette, n_colors=n_bins))

    collection = []
    values = []
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    for body, (idx, row) in zip(space.bodies, df.iterrows()):
        circle = next(iter(body.shapes))
        c = Circle(np.array(body.position), circle.radius - margin)
        collection.append(c)
        values.append(row[position_column])

        body_x = body.position.x
        body_min_x = body_x - circle.radius
        body_max_x = body_x + circle.radius

        body_y = body.position.y
        body_min_y = body_y - circle.radius
        body_max_y = body_y + circle.radius

        if body_min_y < min_y:
            min_y = body_min_y

        if body_min_x < min_x:
            min_x = body_min_x

        if body_max_y > max_y:
            max_y = body_max_y

        if body_max_x > max_x:
            max_x = body_max_x

        if label_column is not None:
            label_size = int(scale(radius[idx]) * max_label_size)
            if label_size < min_label_size:
                continue
            ax.annotate(
                row[label_column],
                np.array(body.position),
                ha="center",
                va="center",
                fontsize=label_size,
            )

    collection = PatchCollection(collection, color="pink", edgecolor="none", cmap=cmap)
    collection.set_array(np.array(values))
    ax.add_collection(collection)

    ax.set_aspect("equal")
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])

    return space, collection
