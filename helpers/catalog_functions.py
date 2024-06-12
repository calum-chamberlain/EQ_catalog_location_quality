"""
Functions to organise and calculate quality criteria from catalogue

"""

import numpy as np
import warnings

from dataclasses import dataclass

from math import radians, cos, sin, asin, sqrt

from obspy import Catalog, Inventory, UTCDateTime
from obspy.core.event import Event, Arrival, Pick, Origin
from obspy.geodetics import gps2dist_azimuth

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
        origin = _get_origin(ev)
        if origin is None:
            continue
        if not origin.quality.minimum_distance:
            # Compute the minimum distance in degrees
            print(
                f"Event {ev.resource_id.id} at {origin.time} missing minimum distance, computing"
            )
            min_dist = get_min_dist_degrees(ev, inventory=inventory)
            origin.quality.minimum_distance = min_dist
        if not origin.quality.azimuthal_gap:
            # Compute the azimuthal gap
            print(
                f"Event {ev.resource_id.id} at {origin.time} missing azimuthal gap, computing"
            )
            azi_gap = get_azimuthal_gap(ev, inventory=inventory)
            origin.quality.azimuthal_gap = azi_gap
        cat_checked.append(ev)
    return cat_checked


def get_azimuthal_gap(event: Event, inventory: Inventory) -> float:
    """
    Compute azimuthal gap for an event.
    """
    origin = _get_origin(event)
    if origin is None:
        return 360.0
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


def compute_azimuth(arrival: Arrival, event: Event, inventory: Inventory) -> float:
    """
    Compute the azimuth from an event to a station given by an arrival.

    Params
    ------
    arrival:
        The arrival to compute the azimuth for
    event:
        The event with at least an origin
    inventory:
        Inventory with at least station locations

    Returns
    -------
    Azimuth (in degrees) of a direct line from event to station.
    """
    pick = arrival.pick_id.get_referred_object()
    if pick is None:
        raise NotImplementedError(f"Arrival not associated with pick: {arrival}\n{event}")
    sta_loc = lookup_station_coords(
        inventory=inventory, network=pick.waveform_id.network_code, 
        station=pick.waveform_id.station_code, time=pick.time,
        channel=f"{pick.waveform_id.channel_code[0:2]}?")
    if sta_loc is None:
        return None
    ori = event.preferred_origin() or event.origins[-1]
    ev_lat, ev_lon = ori.latitude, ori.longitude
    dist, az, baz = gps2dist_azimuth(
        lat1=ev_lat, lon1=ev_lon, lat2=sta_loc.latitude, lon2=sta_loc.longitude)
    return az


def get_min_dist_degrees(event: Event, inventory: Inventory) -> float:
    """
    Lookup or compute the mimnum distance in degrees between event origin and stations.
    """
    origin = _get_origin(event)
    if origin is None:
        return 360.0
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
        if pick_coords is None:
            continue
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
    if len(selected) == 0:
        warnings.warn(f"Could not find {network}.{station} at {time}")
        return None
    # Coords is hashable, so we can make a set. If there is more than one item
    # in the set, then we have multiple possible locations.
    channel_locations = {
        Coords(c.latitude, c.longitude, c.elevation, f"{n.code}.{s.code}")
        for n in selected
        for s in n
        for c in s
    }
    if len(channel_locations) > 1:
        warnings.warn(
            f"Multiple possible locations for {network}.{station}:\n"
            f"{channel_locations}, using zeroth"
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
    origin = _get_origin(ev)
    if origin is None:
        return None
    arrivals = sorted(origin.arrivals, key=lambda arr: arr.distance, reverse=False)
    if round(arrivals[0].distance - min_dist_rad, 2) != 0.0:
        warnings.warn(
            f"Smallest distance arrival ({arrivals[0].distance}) "
            f"does not match ({min_dist_rad}) for {ev.resource_id.id}"
        )
        try:
            arr = [a for a in arrivals if a.distance == min_dist_rad][0]
        except IndexError:
            warnings.warn(
                f"No arrival matching the min distance ({min_dist_rad}) for event id "
                f"{ev.resource_id.id}, using the closest distance arrival"
            )
            arr = arrivals[0]
    else:
        arr = arrivals[0]
    try:
        pick = arr.pick_id.get_referred_object()
    except Exception as e:
        raise NotImplementedError("Arrival not associated with a pick")
    return _get_distance_for_pick(pick=pick, origin=origin, inventory=inventory)


def _get_distance_for_pick(pick: Pick, origin: Origin, inventory: Inventory):
    """ Internal func for convenience. """

    chan_coords = lookup_station_coords(
        inventory=inventory,
        network=pick.waveform_id.network_code,
        station=pick.waveform_id.station_code,
        time=pick.time,
        channel=f"{pick.waveform_id.channel_code[0:-1]}?",
    )
    if chan_coords is None:
        return None
    min_dist_m, _, _ = gps2dist_azimuth(
        lat1=origin.latitude,
        lon1=origin.longitude,
        lat2=chan_coords.latitude,
        lon2=chan_coords.longitude,
    )

    return min_dist_m / 1000


def _get_origin(event: Event) -> Origin:
    """ Get the origin of an event. """
    try:
        ori = event.preferred_origin() or event.origins[-1]
    except IndexError:
        warnings.warn(f"No origins for {event.resource_id.id}")
        return None
    return ori


def min_spick_dist(cat_poly: Catalog, inventory: Inventory) -> List[bool]:
    """
    Check if the closest used S-pick meets the minimum distance-depth requirements.

    Params
    ------
    cat_poly:
        Catalogue to check events for
    inventory:
        Inventory used for picking events

    Returns
    -------
    list of bool ordered as cat_poly where True events meet the 1.4xdepth requirement
    """
    
    s_picks = list(np.zeros(len(cat_poly), dtype=bool))

    for i, ev in enumerate(cat_poly):
        # Find closest S-arrival
        ori = _get_origin(ev)
        if ori is None:
            continue
        min_dist = 999999
        for arr in ori.arrivals:
            if not arr.phase.upper().startswith("S"):
                # Not s-pick, requirement not met for this arrival
                continue
            # Calculate distance
            pick = arr.pick_id.get_referred_object()
            if pick is None:
                warnings.warn(f"Arrival for {ev.resource_id.id} not linked to pick. Skipping")
                # Cannot check - requirement not met for this arrival
                continue
            dist = _get_distance_for_pick(pick=pick, origin=ori, inventory=inventory)
            if dist and dist < min_dist:
                # Overload min_dist to get the true minimum distance S-pick
                min_dist = dist
        # print(f"Min dist for S pick for event {ev.resource_id.id}: {min_dist:.2f} km. Depth: {ori.depth / 1000:.2f}")
        if min_dist < 1.4 * (ori.depth / 1000):
            s_picks[i] = True
    return s_picks


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
        ori = _get_origin(ev)
        max_gap.append(ori.quality.azimuthal_gap)
        depths.append(ori.depth / 1000)
        # calculate min_distance
        min_dist_rad = ori.quality.minimum_distance
        # find distance in km to station which corresponds to this arrival
        min_dist.append(
            min_dist_calculator(ev=ev, min_dist_rad=min_dist_rad, inventory=inventory)
        )
        mags.append((ev.preferred_magnitude() or ev.magnitude[-1]).mag)
        lats.append(ori.latitude)
        lons.append(ori.longitude)
        if ori.depth_type == "operator assigned":
            fixed.append(True)
        else:
            fixed.append(False)

    return max_gap, depths, min_dist, mags, lats, lons, fixed


def binary_counts(
    max_gap: Iterable[float],
    cat_poly: Catalog,
    depths: Iterable[float],
    min_dist: Iterable[float],
    fixed: Iterable[bool],
):
    """
    Check quality criteria for individual events.

    All inputs must be the same size and ordered the same.

    Params
    ------
    max_gap:
        List of maximum azimuthal gaps for events
    cat_poly:
        Catalog of events
    depths:
        Depths for all events
    min_dist:
        Minimum pick distances for all events
    fixed:
        Whether event depths are fixed for all events

    Returns
    -------
    lists of bools of whether quality criteria have been met, ordered as:
    - maximum azimuthal gap < 180.0
    - minimum number of picks >= 8
    - whether at least one pick is an S
    - whether the minimum distance is within one focal depth
    - whether the detpth is free
    """
    assert len(max_gap) == len(cat_poly)
    assert len(depths) == len(cat_poly)
    assert len(min_dist) == len(cat_poly)
    assert len(fixed) == len(cat_poly)
    
    # min_azimuth requirement:
    min_az_bi = [az <= 180 for az in max_gap]
    #for az in max_gap:
    #    if az <= 180:
    #        min_az_bi.append(True)
    #    else:
    #        min_az_bi.append(False)
    # min picks and one spick requirement
    ps, min_picks = [], []
    for i, ev in enumerate(cat_poly):
        ori = _get_origin(ev)
        if ori and len(ori.arrivals) < 8:
            min_picks.append(False)
        else:
            min_picks.append(True)
        rt = []
        if ori:
            rt = [arr.phase for arr in ori.arrivals]
        if rt.count("S") < 1:
            ps.append(False)
        else:
            ps.append(True)
    # min_dist
    min_dist_bi = [min_d <= d for min_d, d in zip(min_dist, depths)]
    #for i, d in enumerate(depths):
    #    if min_dist[i] <= d:
    #        min_dist_bi.append(True)
    #    else:
    #        min_dist_bi.append(False)
    # plus fixed and_spicks
    fixed_inv = [not f for f in fixed]

    return min_az_bi, min_picks, ps, min_dist_bi, fixed_inv


if __name__ == "__main__":
    print("These are not the snakes you are looking for.")
