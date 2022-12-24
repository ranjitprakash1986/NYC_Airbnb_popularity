# Author: Ranjit Sundaramurthi
# Date: 2022-12-2022

# code adapted from DSCI-531 Data Visualization I

import vl_convert as vlc


def save_chart(chart, filename, scale_factor=1):
    """
    Save an Altair chart using vl-convert

    Parameters
    ----------
    chart : altair.Chart
        Altair chart to save
    filename : str
        The path to save the chart to
    scale_factor: int or float
        The factor to scale the image resolution by.
        E.g. A value of `2` means two times the default resolution.
    """
    if filename.split(".")[-1] == "svg":
        with open(filename, "w", encoding='utf-8') as f:
            f.write(vlc.vegalite_to_svg(chart.to_dict()))
    elif filename.split(".")[-1] == "png":
        with open(filename, "wb") as f:
            f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
    else:
        raise ValueError("Only svg and png formats are supported")
