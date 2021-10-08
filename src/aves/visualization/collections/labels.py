import matplotlib.patheffects as path_effects
from adjustText import adjust_text


class LabelCollection(object):
    def __init__(self, default_ha="center", default_va="center"):
        self.elements = []
        self.ha = default_ha
        self.va = default_va

    def add_text(self, s, x, y, ha=None, va=None):
        self.elements.append((s, x, y, ha, va))

    def render(
        self,
        ax,
        fig=None,
        tight_figure=True,
        avoid_collisions=False,
        outline=False,
        adjustment_args={},
        outline_args={"linewidth": 2, "foreground": "black"},
        **kwargs
    ):
        rendered = []

        for s, x, y, ha, va in self.elements:
            if ha is None:
                ha = self.ha
            if va is None:
                va = self.va

            text = ax.text(x, y, s, ha=ha, va=va, **kwargs)

            if outline:
                text.set_path_effects(
                    [
                        path_effects.Stroke(**outline_args),
                        path_effects.Normal(),
                    ]
                )

            rendered.append(text)

        if fig is not None and tight_figure:
            fig.tight_layout()

        if avoid_collisions:
            adjust_text(rendered, ax=ax, **adjustment_args)
