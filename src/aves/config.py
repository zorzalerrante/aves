import matplotlib as mpl
import seaborn as sns
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager




def setup_style(
    style="whitegrid",
    context="paper",
    dpi=192,
    font_family=None,
    formatter_limits=True,
    font_scale=0.9,
):
    sns.set_style(style)
    sns.set_context(context, font_scale=font_scale)
    # esto configura la calidad de la imagen. dependerá de tu resolución. el valor por omisión es 80
    mpl.rcParams["figure.dpi"] = dpi
    
    if font_family:
        # esto depende de las fuentes que tengas instaladas en el sistema.
        if font_family in mpl.font_manager.get_font_names():
            mpl.rcParams["font.family"] = font_family
    else:
        font_path = Path(os.path.dirname(__file__)) / 'assets' / 'RobotoCondensed-Regular.ttf'
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)

        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = prop.get_name()

    if formatter_limits:
        mpl.rcParams["axes.formatter.limits"] = (-99, 99)
