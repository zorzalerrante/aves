import matplotlib.patheffects as path_effects


def text(ax, *args, **kwargs):
    outline = kwargs.pop("outline", None)
    t = ax.text(*args, **kwargs)

    if not outline is None:
        if type(outline) != dict:
            outline = dict(width=outline)

        t.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=outline.get("width", 1),
                    foreground=outline.get("color", "black"),
                ),
                path_effects.Normal(),
            ]
        )

    return t
