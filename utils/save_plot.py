# Author: Ranjit Sundaramurthi
# Date: 2022-12-2022

"""
This is a script that contains a plot saving function for the Altair plot.
It will be used in the general_EDA.py or other related script for plot saving.
 
Options:
<None>
"""


# code adapted from DSCI-531 Data Visualization I


def save_chart(chart, filename, scale_factor=1):
    """
    Save an Altair chart using altair saver

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
    from altair_saver import save  # to save the altair charts in svg format
    import altair as alt

    alt.renderers.enable("mimetype")
    alt.data_transformers.disable_max_rows()

    if filename.split(".")[-1] == "svg":
        save(chart, filename)
    else:
        raise ValueError("Only svg formats is supported")
