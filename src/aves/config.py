import matplotlib as mpl
import seaborn as sns


def setup_style(
    style="whitegrid",
    context="paper",
    dpi=192,
    font_family="Inter",
    formatter_limits=True,
):
    sns.set_style(style)
    sns.set_context(context)
    # esto configura la calidad de la imagen. dependerá de tu resolución. el valor por omisión es 80
    mpl.rcParams["figure.dpi"] = dpi
    # esto depende de las fuentes que tengas instaladas en el sistema.
    if font_family in mpl.font_manager.get_font_names():
        mpl.rcParams["font.family"] = font_family

    if formatter_limits:
        mpl.rcParams["axes.formatter.limits"] = (-99, 99)
