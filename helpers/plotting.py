"""
Functions to organise and calculate quality criteria from catalogue

"""

import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
import numpy as np

from obspy import Catalog, Inventory

from mpl_toolkits.axes_grid1 import make_axes_locatable

from shapely.geometry import Point, Polygon

from typing import Iterable


def plot_catalog(
    lats: Iterable[float],
    lons: Iterable[float],
    depths: Iterable[float],
    mags: Iterable[float],
    poly: Polygon,
    region: str,
    bblats: Iterable[float],
    bblons: Iterable[float],
    splats: Iterable[float],
    splons: Iterable[float],
) -> plt.Figure:
    """
    Plot a catalog for a region on a map. Earthquakes coloured by depth and scaled by magnitude.

    Params
    ------
    lats:
        Latitudes of events in catalog
    lons:
        Longitudes of events in catalog
    depths:
        Depths (km) of events in catalog
    mags:
        Magnitudes of events in catalog
    poly:
        Polygon outline for selected region
    region:
        Region name (used only for title of plot)
    bblats:
        Broadband station latitudes (or instrument type 1)
    bblons:
        Broadband station longitudes (or instrument type 1)
    splats:
        Short-period station latitudes (or instrument type 2)
    splons:
        Short-period station longitudes (or instrument type 2)

    Returns
    -------
    Map plot
    """

    # define map region
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 1, 1, projection=crs.Mercator())
    ax1.set_extent([min(lons) - 0.5, max(lons) + 0.5, min(lats) - 0.5, max(lats) + 0.5])
    ax1.add_feature(cfeature.COASTLINE)
    # plot earthquakes
    mappable = ax1.scatter(
        lons, lats, s=mags, c=depths, cmap="plasma_r", transform=crs.Geodetic()
    )
    ax1.scatter(
        bblons,
        bblats,
        s=35,
        color="black",
        marker="^",
        transform=crs.Geodetic(),
        label="Existing GeoNet Broadband",
    )
    ax1.scatter(
        splons,
        splats,
        s=35,
        color="black",
        marker="v",
        transform=crs.Geodetic(),
        label="Existing GeoNet Short Period",
    )
    # plot polygon of region
    polyx, polyy = poly.exterior.xy
    ax1.plot(polyx, polyy, c="gray", transform=crs.Geodetic())
    # add legend
    fig.colorbar(mappable, label="Earthquake Depth (km)")
    plt.title(str(len(lons)) + " events in " + region + " catalogue")

    return fig


def plot_azimuthal_map(
    lats: Iterable[float],
    lons: Iterable[float],
    max_gap: Iterable[float],
    poly: Polygon,
    region: str,
    bblats: Iterable[float],
    bblons: Iterable[float],
    splats: Iterable[float],
    splons: Iterable[float],
) -> plt.Figure:
    """
    Make a map plot of an earthquake catalog coloured by azimuthal gap

    Params
    ------
    lats:
        Latitudes of events in catalog
    lons:
        Longitudes of events in catalog
    max_gap:
        Maximum azimuthal gap for events in catalog
    poly:
        Polygon outline for selected region
    region:
        Region name (used only for title of plot)
    bblats:
        Broadband station latitudes (or instrument type 1)
    bblons:
        Broadband station longitudes (or instrument type 1)
    splats:
        Short-period station latitudes (or instrument type 2)
    splons:
        Short-period station longitudes (or instrument type 2)

    Returns
    -------
    Map plot

    """

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 1, 1, projection=crs.Mercator())
    # hack to force colour bar between 0 and 360
    # minaz_c = [0, 360]
    # max_gapc = minaz_c + max_gap[2:]
    # plot data
    mappable = ax1.scatter(
        lons,
        lats,
        s=10,
        c=max_gap,
        cmap="coolwarm",
        vmin=0,
        vmax=360,
        transform=crs.Geodetic(),
    )
    ax1.scatter(
        bblons,
        bblats,
        s=35,
        color="black",
        marker="^",
        transform=crs.Geodetic(),
        label="Existing GeoNet Broadband",
    )
    ax1.scatter(
        splons,
        splats,
        s=35,
        color="black",
        marker="v",
        transform=crs.Geodetic(),
        label="Existing GeoNet Short Period",
    )
    # plot scale
    fig.colorbar(mappable, label=r"Maximum Azimuthal Gap, ($^\circ$)")
    ax1.legend(loc="upper left")
    ax1.set_extent([min(lons) - 1.5, max(lons) + 0.5, min(lats) - 0.2, max(lats) + 0.6])
    ax1.add_feature(cfeature.COASTLINE)
    # plot polygon of region
    polyx, polyy = poly.exterior.xy
    ax1.plot(polyx, polyy, c="gray", transform=crs.Geodetic())
    count = sum(1 for a in max_gap if a <= 180)
    plt.title(
        str(round(count / len(max_gap) * 100, 1))
        + "% of events in "
        + region
        + " region satisfy the azimuthal gap criterion"
    )

    return fig


def plot_depth_scatter(
    min_dist: Iterable[float],
    s_picks: Iterable[bool],
    depths: Iterable[float],
    fixed: Iterable[bool],
    region: str,
) -> plt.Figure:
    """
    Plot depth scatter and stacked histogram for events.

    All inputs should be ordered the same way

    Params
    ------
    min_dist:
        Minimum distance for picks
    s_picks:
        Whether S picks exist for the event within 1.4x depth
    depths:
        Depth of events
    fixed:
        Whether event depths are fixed or not
    region:
        Name of region plotted
    """

    max_depth = max(depths)
    max_plot_depth = (10 * (max_depth // 10)) + 10

    good_poly = Polygon([
        (0, 0), 
        (max_depth + 0.1, max_depth + 0.1), 
        (0, max_depth + 0.1)])
    okay_poly = Polygon([
        (0, 0), 
        (1.4 * (max_depth + 0.1), max_depth + 0.1), 
        (max_depth + 0.1, max_depth + 0.1)])

    mindistg, depthsg, fixedg = [], [], []  # Green - good
    mindistS, depthsS, fixedS = [], [], []  # Light pink - okay
    mindistSg, depthsSg, fixedSg = [], [], []  # Dark green
    mindistbad, depthsbad, fixedbad = [], [], []  # Pink - bad
    for i, d in enumerate(min_dist):
        loc = Point(d, depths[i])
        if good_poly.contains(loc) and not s_picks[i]:
            # Good - If point is within swath of acceptable depth-distance relation
            mindistg.append(d)
            depthsg.append(depths[i])
            fixedg.append(fixed[i])
        elif good_poly.contains(loc) and s_picks[i]:
            # Best - within swath and with P and S picks
            mindistSg.append(d)
            depthsSg.append(depths[i])
            fixedSg.append(fixed[i])
        elif okay_poly.contains(loc) and s_picks[i]:
            # Okay - Point is within okay swath and there is an S-pick
            mindistS.append(d)
            depthsS.append(depths[i])
            fixedS.append(fixed[i])
        else:
            # Bad - Outside of all swaths, or within secondary swath but without an S pick
            if d < 20 and depths[i] > 40:
                print(f"Oddly dodgy event {i} at {depths[i]} km depth and min dist {d} km")
            mindistbad.append(d)
            depthsbad.append(depths[i])
            fixedbad.append(fixed[i])

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_aspect("equal")
    ax.scatter(
        mindistbad, depthsbad, c="#B3589A", marker="x", s=8, label="bad: depth < min dist"
    )
    ax.scatter(
        mindistS, depthsS, c="#F6D3E8", marker="o", s=8, label="OK: S-phase contained"
    )
    ax.scatter(
        mindistg, depthsg, c="#BBD4A6", marker="o", s=12, label="good: depth > min dist"
    )
    ax.scatter(
        mindistSg,
        depthsSg,
        c="#9BBF85",
        marker="o",
        s=12,
        label="Best: P+S constraints",
    )
    
    ax.plot([0, max_plot_depth], [0, max_plot_depth], color="k", label="1:1 (P picks)")
    ax.plot([0, max_plot_depth * 1.4], [0, max_plot_depth], color="k", linestyle="dashed", label="1:1.4 (S picks)")
    ax.set_xlabel("Minimum distance, km")
    ax.set_ylabel("Depth, km")
    ax.set_xlim(0, 100)
    ax.set_ylim(max_plot_depth, 0)
    ax.legend(loc="lower right")
    ax.set_title(
        str(round(len(depthsg) / len(depths) * 100, 1))
        + "% of earthquakes in "
        + region
        + " have an acceptable minimum distance (less than depth)"
    )

    divider = make_axes_locatable(ax)
    ax_histy = divider.append_axes("right", 4, pad=0.5, sharey=ax)

    bin_width = 5
    bins = np.arange(0, max_plot_depth + bin_width, bin_width)

    ax_histy.hist(
        [depthsSg, depthsg, depthsS, depthsbad],
        bins=bins,
        orientation="horizontal",
        color=[
            "#9BBF85",
            "#BBD4A6", 
            "#F6D3E8", 
            "#B3589A", 
            ],
        label=[
            "best: P+S constraints",
            "good: depth > min dist",
            "OK: S-phase contained",
            "bad: depth < min dist",
            ],
        stacked=True,
    )
    ax_histy.set_xlabel("Cumulative number of earthquakes")

    ax_histy.legend(loc="lower right")
    ax_histy.set_title(
        str(round(fixed.count(1) / len(fixed) * 100, 1))
        + "% of events have fixed depth"
    )

    return fig


def plot_quality_criteria_scores(
    fixed_inv: Iterable[bool],
    cat_poly: Catalog,
    s_picks: Iterable[bool],
    min_dist_bi: Iterable[bool],
    ps: Iterable[bool],
    min_picks: Iterable[bool],
    min_az_bi: Iterable[bool],
    region: str,
) -> plt.Figure:
    """
    Plot bars of events meeting different quality criteria.

    All inputs should be ordered the same

    Params
    ------
    fixed_inv:
        Whether events have free depths
    cat_poly:
        Catalog of events
    s_picks:
        Whether S-picks are included
    min_dist_bi:
        Whether minimum distance criteria is met
    ps:
        Whether P and S pick count criteria are met
    min_picks:
        Whether minimum picks criteria are met
    min_az_bi:
        Whether minimum azimuth criteria is met
    region:
        Name of region
    """

    criteria = [
        "Not fixed",
        r"Min_Sdist$\leq$1.4*depth",
        r"Min_dist$\leq$depth",
        r"Min_Spicks$\geq1$",
        r"Min_picks$\geq$8",
        r"Min_az$\leq$180$^\circ$",
    ]
    # TODO: Convert to bools and just use sum?
    counts = [
        fixed_inv.count(1) / len(cat_poly) * 100,
        s_picks.count(1) / len(cat_poly) * 100,
        min_dist_bi.count(1) / len(cat_poly) * 100,
        ps.count(1) / len(cat_poly) * 100,
        min_picks.count(1) / len(cat_poly) * 100,
        min_az_bi.count(1) / len(cat_poly) * 100,
    ]
    countsfull = [100, 100, 100, 100, 100, 100]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(criteria, countsfull, color="#fee6ce", label="Not Satisfied")
    ax.barh(criteria, counts, color="#fdae6b", label="Satisfied")
    ax.legend(bbox_to_anchor=(1, 1.15))
    plt.title(region)

    ax.set_ylabel("Quality criteria")
    ax.set_xlabel("Percentage of events")
    ax.set_xlim([0, 100])
    plt.xticks(rotation=45)
    plt.show()

    return fig, counts, criteria


def plot_quality_score_map(
    quals,
    counts: Iterable[int],
    lons: Iterable[float],
    lats: Iterable[float],
    region: str,
    poly: Polygon,
    cat_poly: Catalog,
    sums,
    bblons: Iterable[float],
    bblats: Iterable[float],
    splons: Iterable[float],
    splats: Iterable[float],
):
    """
    I don't know what some of these input are.
    """

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 1, 1, projection=crs.Mercator())
    cmap = plt.get_cmap("coolwarm_r", np.max(quals) - np.min(quals) + 1)
    # plot data
    mappable = ax1.scatter(
        lons,
        lats,
        s=5,
        c=sums,
        cmap=cmap,
        vmin=np.min(quals) - 0.5,
        vmax=np.max(quals) + 0.5,
        transform=crs.Geodetic(),
    )
    ax1.scatter(
        bblons,
        bblats,
        s=35,
        color="black",
        marker="^",
        transform=crs.Geodetic(),
        label="Existing GeoNet Broadband",
    )
    ax1.scatter(
        splons,
        splats,
        s=35,
        color="black",
        marker="v",
        transform=crs.Geodetic(),
        label="Existing GeoNet Short Period",
    )
    ax1.legend(loc="lower right")
    ax1.set_extent([min(lons) - 0.5, max(lons) + 0.5, min(lats) - 0.5, max(lats) + 0.5])
    # add coastline, title and scale
    ax1.add_feature(cfeature.COASTLINE)
    plt.title(
        "In "
        + region
        + " "
        + str(round(counts[-1] / len(cat_poly) * 100, 1))
        + "% of events have a Quality Score = 6",
        loc="right",
    )
    fig.colorbar(
        mappable,
        ticks=np.arange(np.min(quals), np.max(quals) + 1),
        label="Quality Score",
    )
    # plot polygon of region
    polyx, polyy = poly.exterior.xy
    ax1.plot(polyx, polyy, c="gray", transform=crs.Geodetic())
    labels = [str(n) for n in quals]
    # add in pie chart
    ax2 = fig.add_axes([0.055, 0.48, 0.4, 0.4])  # left bottom width height
    colours = [cmap(n) for n in quals]
    ax2.pie(
        counts,
        labels=labels,
        labeldistance=0.6,
        colors=colours,
        startangle=90,
        shadow=False,
    )

    return fig


def plot_quality_score_bar(region, countspc_cum, quals):

    fig, ax = plt.subplots(figsize=(10, 1))
    cmap = plt.get_cmap("coolwarm_r", np.max(quals) - np.min(quals) + 1)
    colours = [cmap(n) for n in quals]
    ax.barh(region, countspc_cum[6], color=colours[6], label="QS6")
    ax.barh(region, countspc_cum[5], color=colours[5], label="QS5")
    ax.barh(region, countspc_cum[4], color=colours[4], label="QS4")
    ax.barh(region, countspc_cum[3], color=colours[3], label="QS3")
    ax.barh(region, countspc_cum[2], color=colours[2], label="QS2")
    ax.barh(region, countspc_cum[1], color=colours[1], label="QS1")
    ax.barh(region, countspc_cum[0], color=colours[0], label="QS0")
    ax.set_xlim([0, 100])
    ax.legend(bbox_to_anchor=(1, 1.15))

    return fig


if __name__ == "__main__":
    print("These are not the snakes you are looking for.")
