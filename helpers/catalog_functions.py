"""
Functions to organise and calculate quality criteria from catalogue

"""

import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
import numpy as np
import warnings

from dataclasses import dataclass

from math import radians, cos, sin, asin, sqrt

from obspy import Catalog, Inventory, UTCDateTime
from obspy.core.event import Event
from obspy.geodetics import gps2dist_azimuth

from mpl_toolkits.axes_grid1 import make_axes_locatable

from shapely.geometry import Point, Polygon

from typing import Iterable, List, Tuple


@dataclass(frozen=True, order=True)
class Coords:
    latitude: float
    longitude: float
    elevation: float
    key: str


def quality_measures_check(catalog: Catalog, inventory: Inventory) -> Catalog:
    """
    Check that events have the required quality measures and add them as needed
    """
    cat_checked = Catalog()
    for ev_in in catalog:
        ev = ev_in.copy()
        origin = ev.preferred_origin() or ev.origins[-1]
        if not origin.quality.minimum_distance:
            # Compute the minimum distance in degrees
            print(f"Event at {origin.time} missing minimum distance, computing")
            min_dist = get_min_dist_degrees(ev, inventory=inventory)
            origin.quality.minimum_distance = min_dist
        if not origin.quality.azimuthal_gap:
            # Compute the azimuthal gap
            print(f"Event at {origin.time} missing azimuthal gap, computing")
            azi_gap = get_azimuthal_gap(ev, inventory=inventory)
            origin.quality.azimuthal_gap = azi_gap
        cat_checked.append(ev)
    return cat_checked


def get_azimuthal_gap(event: Event, inventory: Inventory) -> float:
    """
    Compute azimuthal gap for an event.
    """
    origin = event.preferred_origin() or event.origins[-1]
    azimuths = [
        arr.azimuth or compute_azimuth(arr, event=event, inventory=inventory)
        for arr in origin.arrivals
    ]
    max_gap = 0.0
    for i, azimuth in enumerate(azimuths):
        # Rotate all azimuths so that azimuth is at 0
        other_azimuths = np.array([_azi for j, _azi in enumerate(azimuths) if j != i])
        other_azimuths -= azimuth
        other_azimuths %= 360
        cw_dist = min(other_azimuths)
        ccw_dist = min(360 - other_azimuths)
        gap = max(cw_dist, ccw_dist)
        if gap > max_gap:
            max_gap = gap
    return max_gap


def get_min_dist_degrees(event: Event, inventory: Inventory) -> float:
    """
    Lookup or compute the mimnum distance in degrees between event origin and stations.
    """
    origin = event.preferred_origin() or event.origins[-1]
    try:
        arrivals = sorted(origin.arrivals, key=lambda arr: arr.distance)
    except Exception as e:
        arrivals = None
    if arrivals:
        # If we have those arrivals, all is good.
        return arrivals[0].distance
    # Otherwise we need to work it out ourselves
    print(f"Pre-computed distances not found, computing")
    min_dist = 360.0
    for arrival in origin.arrivals:
        pick = arrival.pick_id.get_referred_object()
        if pick is None:
            raise NotImplementedError("No matching pick for arrival")
        pick_coords = lookup_station_coords(
            inventory=inventory,
            network=pick.waveform_id.network_code,
            station=pick.waveform_id.station_code,
            time=pick.time,
        )
        dist = haversine(
            lon1=origin.longitude,
            lat1=origin.latitude,
            lon2=pick_coords.longitude,
            lat2=pick_coords.latitude,
        )
        if dist < min_dist:
            min_dist = dist
    return min_dist


def lookup_station_coords(
    inventory: Inventory,
    network: str,
    station: str,
    time: UTCDateTime,
    channel: str = None,
) -> Coords:
    """Find a satisfactory co-ordinate for a station for a given time"""
    selected = inventory.select(
        network=network,
        station=station,
        starttime=time - 120,
        endtime=time + 120,
        channel=channel,
    )
    assert len(selected), f"Could not find {network}.{station} at {time}"
    # Coords is hashable, so we can make a set. If there is more than one item
    # in the set, then we have multiple possible locations.
    channel_locations = {
        Coords(c.latitude, c.longitude, c.elevation, f"{n.code}.{s.code}")
        for n in selected
        for s in n
        for c in s
    }
    if len(channel_locations) > 1:
        raise NotImplementedError(
            f"Multiple possible locations for {network}.{station}:\n"
            f"{channel_locations}"
        )
    return channel_locations.pop()


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great circle distance in degrees between two points
    on the earth (specified in decimal degrees)

    Params
    ------
    lat1:
        Latitude of point 1
    lon1:
        Longitude of point 1
    lat2:
        Latitude of point 2
    lon2:
        Longitude of point 2
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return c


def min_dist_calculator(min_dist_rad: float, ev: Event, inventory: Inventory) -> float:
    """
    Calculate km distance to station corresponding to closest arrival from event.

    Params
    ------
    min_dist_rad:
        Input distance in radians - will find the arrival that corresponds to this
    ev:
        Event to use as source location
    inventory:
        Inventory of stations (at least the stations picked in ev)
    """
    origin = ev.preferred_origin() or ev.origins[-1]
    arrivals = sorted(origin.arrivals, key=lambda arr: arr.distance, reverse=False)
    if round(arrivals[0].distance - min_dist_rad, 2) != 0.0:
        warnings.warn(
            f"Smallest distance arrival ({arrivals[0].distance}) "
            f"does not match {min_dist_rad})"
        )
        try:
            arr = [a for a in arrivals if a.distance == min_dist_rad][0]
        except IndexError:
            raise NotImplementedError("No arrival matching the min distance")
    else:
        arr = arrivals[0]
    try:
        pick = arr.pick_id.get_referred_object()
    except Exception as e:
        raise NotImplementedError("Arrival not associated with a pick")

    chan_coords = lookup_station_coords(
        inventory=inventory,
        network=pick.waveform_id.network_code,
        station=pick.waveform_id.station_code,
        time=pick.time,
        channel=f"{pick.waveform_id.channel_code[0:-1]}?",
    )
    min_dist_m, _, _ = gps2dist_azimuth(
        lat1=origin.latitude,
        lon1=origin.longitude,
        lat2=chan_coords.latitude,
        lon2=chan_coords.longitude,
    )

    return min_dist_m / 1000


def assign_variables(cat_poly: Catalog, inventory: Inventory):
    """
    Construct minimum azimuth, depths, minimum distances, magnitudes, latitudes,
    longitudes and fixed depth lists.

    Params
    ------
    cat_poly:
        Input catalog for a specific region
    inventory:
        Inventory of stations used for cat_poly

    Returns:
    --------
    A series of lists ordered as:
    - event azimuthal gaps (list of float)
    - event depths in km (list of float)
    - event distances to closest picked station in km (list of float)
    - event magnitudes (list of float)
    - event latitude (list of float)
    - event longitudes (list of float)
    - event depth fixing (list of bool)

    """

    max_gap, depths, min_dist, mags, lats, lons, fixed = [], [], [], [], [], [], []

    for ev in cat_poly:
        max_gap.append((ev.preferred_origin() or ev.origins[-1]).quality.azimuthal_gap)
        depths.append((ev.preferred_origin() or ev.origins[-1]).depth / 1000)
        # calculate min_distance
        min_dist_rad = (
            ev.preferred_origin() or ev.origins[-1]
        ).quality.minimum_distance
        # find distance in km to station which corresponds to this arrival
        min_dist.append(
            min_dist_calculator(ev=ev, min_dist_rad=min_dist_rad, inventory=inventory)
        )
        mags.append((ev.preferred_magnitude() or ev.magnitude[-1]).mag)
        lats.append((ev.preferred_origin() or ev.origins[-1]).latitude)
        lons.append((ev.preferred_origin() or ev.origins[-1]).longitude)
        if (ev.preferred_origin() or ev.origins[-1]).depth_type == "operator assigned":
            fixed.append(1)
        else:
            fixed.append(0)

    return max_gap, depths, min_dist, mags, lats, lons, fixed


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
    plt.title(str(len(cat)) + " events in " + region + " catalogue")

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
        c=min_gap,
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


def min_spick_dist(cat_poly: Catalog) -> List[bool]:

    # TODO: This should just be a pre-filled np array of False?
    s_picks = []
    for j, ev in enumerate(cat_poly):
        # flags = []
        for ar in (ev.preferred_origin() or ev.origins[-1]).arrivals:
            ##### TODO: THIS IS A FUDGE UNTIL THE BUG IN THE DISTANCE CALC USING THE INVENTORY IS FIXED
            if (
                ar.phase == "S"
                and ar.distance * 111.1
                < 1.4 * (ev.preferred_origin() or ev.origins[-1]).depth / 1000
            ):
                s_picks.append(True)
                # TODO: This could break out of the arrivals loop here.
                # flags.append(True)
        # if True in flags:
        #    s_picks.append(True)
        # else:
        #    s_picks.append(False)
        else:
            s_picks.append(False)

    return s_picks


def plot_depth_scatter(
    min_dist: Iterable[float],
    s_picks: Iterable[bool],
    depths: Iterable[float],
    fixed: Iterable[bool],
    region: str,
) -> plt.Figure:

    p = Polygon([(0, 0), (50, 50), (0, 50)])
    mindistg, depthsg, fixedg = [], [], []
    mindistS, depthsS, fixedS = [], [], []
    mindistSg, depthsSg, fixedSg = [], [], []
    for i, d in enumerate(min_dist):
        # TODO: This is likely the source of points plotting in the wrong places, check logic
        if p.contains(Point(d, depths[i])):
            mindistg.append(d)
            depthsg.append(depths[i])
            fixedg.append(fixed[i])
        if s_picks[i]:
            mindistS.append(d)
            depthsS.append(depths[i])
        if p.contains(Point(d, depths[i])) and s_picks[i] > 0:
            mindistSg.append(d)
            depthsSg.append(depths[i])

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_aspect("equal")
    ax.scatter(
        min_dist, depths, c="#B3589A", marker="x", s=8, label="bad: depth < min dist"
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
    ax.plot([0, 50], [0, 50], color="k", label="1:1 (P picks)")
    ax.plot([0, 70], [0, 50], color="k", linestyle="dashed", label="1:1.4 (S picks)")
    ax.set_xlabel("Minimum distance, km")
    ax.set_ylabel("Depth, km")
    ax.set_xlim(0, 100)
    ax.set_ylim(50, 0)
    plt.legend(loc="lower right")
    plt.title(
        str(round(len(depthsg) / len(depths) * 100, 1))
        + "% of earthquakes in "
        + region
        + " have an acceptable minimum distance (less than depth)"
    )

    divider = make_axes_locatable(ax)
    ax_histy = divider.append_axes("right", 4, pad=0.5, sharey=ax)

    bins = np.arange(0, 50, 5)
    ax_histy.hist(
        depths,
        bins=bins,
        orientation="horizontal",
        color="#B3589A",
        label="bad: depth < min dist",
    )
    ax_histy.hist(
        depthsS,
        bins=bins,
        orientation="horizontal",
        color="#F6D3E8",
        label="OK: S-phase contained",
    )
    ax_histy.hist(
        depthsg,
        bins=bins,
        orientation="horizontal",
        color="#BBD4A6",
        label="good: depth > min dist",
    )
    ax_histy.hist(
        depthsSg,
        bins=bins,
        orientation="horizontal",
        color="#9BBF85",
        label="best: P+S constraints",
    )

    ax_histy.set_xlabel("Number of earthquakes")
    plt.legend(loc="lower right")
    plt.title(
        str(round(fixed.count(1) / len(fixed) * 100, 1))
        + "% of events have fixed depth"
    )

    return fig


def binary_counts(
    max_gap: Iterable[float],
    cat_poly: Catalog,
    depths: Iterable[float],
    min_dist: Iterable[float],
    fixed: Iterable[bool],
):

    # TODO: Why are these all ints rather than bools?

    # min_azimuth requirement:
    min_az_bi = []
    for az in max_gap:
        if az <= 180:
            min_az_bi.append(1)
        else:
            min_az_bi.append(0)
    # min picks and one spick requirement
    ps, min_picks = [], []
    for i, ev in enumerate(cat_poly):
        if len((ev.preferred_origin() or ev.origins[-1]).arrivals) < 8:
            min_picks.append(0)
        else:
            min_picks.append(1)
        rt = [arr.phase for arr in (ev.preferred_origin() or ev.origins[-1]).arrivals]
        if rt.count("S") < 1:
            ps.append(0)
        else:
            ps.append(1)
    # min_dist
    min_dist_bi = []
    for i, d in enumerate(depths):
        if min_dist[i] <= d:
            min_dist_bi.append(1)
        else:
            min_dist_bi.append(0)
    # plus fixed and_spicks
    fixed_inv = []
    for f in fixed:
        if f == 0:
            fixed_inv.append(1)
        else:
            fixed_inv.append(0)

    return min_az_bi, min_picks, ps, min_dist_bi, fixed_inv


def plot_quality_criteria_scores(
    fixed_inv: Inventory,
    cat_poly: Catalog,
    s_picks: Iterable[bool],
    min_dist_bi: Iterable[bool],
    ps: Iterable[bool],
    min_picks: Iterable[bool],
    min_az_bi: Iterable[bool],
    region: str,
):

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
    counts,
    lons,
    lats,
    region,
    poly,
    cat_poly,
    sums,
    bblons,
    bblats,
    splons,
    splats,
):

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
