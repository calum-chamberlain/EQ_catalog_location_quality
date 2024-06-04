# Functions to organise and calculate quality criteria from catalogue

from obspy import Catalog
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point, Polygon
from math import radians, cos, sin, asin, sqrt


def Quality_measures_check(catalog):
    ts=Catalog()
    for ev in catalog:
        if not (ev.preferred_origin() or ev.origins[-1]).quality.minimum_distance:
            #NEEDS TO CALCULATE MINIMUM DISTANCE AND ADD IT IN IF IT DOESN'T EXIST
            continue
        else:
            ts.append(ev)
    cat_checked=ts
    ts=Catalog()
    for ev in cat_checked:
        if not (ev.preferred_origin() or ev.origins[-1]).quality.azimuthal_gap:
            #NEEDS TO CALCULATE AZIMUTHAL GAP AND ADD IT IN IF IT DOESN'T EXIST
            continue
        else:
            ts.append(ev)
    cat_checked=ts
    return cat_checked
    
    
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r
    
def min_dist_calculator(min_dist_rad, ev, inventory):
    
    # calculate km distance to station which corresponds to closest arrival from event.
    # input distance in radians
    # find station from pick which corresponds to this arrival
    for arr in (ev.preferred_origin() or ev.origins[-1]).arrivals:
        if arr.distance != min_dist_rad:
            continue
        else:
            for p in ev.picks:
                if p.resource_id == arr.pick_id:
                    pc=p
    #now calculate distance from event to that station:
    CHAN=inventory.get_coordinates(ev.picks[0].waveform_id.network_code + '.' + ev.picks[0].waveform_id.station_code + '.' 
                        + ev.picks[0].waveform_id.location_code + '.' + ev.picks[0].waveform_id.channel_code)
    min_dist_km = (haversine(CHAN['longitude'], CHAN['latitude'], (ev.preferred_origin() or ev.origins[-1]).latitude, (ev.preferred_origin() or ev.origins[-1]).longitude))/1000
    
    return min_dist_km 
    
def assign_variables(cat_poly, inventory):

    min_az, depths, min_dist, mags, lats, lons, fixed = [], [], [], [], [], [], []

    for ev in cat_poly:
        min_az.append((ev.preferred_origin() or ev.origins[-1]).quality.azimuthal_gap)
        depths.append((ev.preferred_origin() or ev.origins[-1]).depth/1000)
        #calculate min_distance
        min_dist_rad=(ev.preferred_origin() or ev.origins[-1]).quality.minimum_distance
        #find distance in km to station which corresponds to this arrival
        min_dist.append(min_dist_calculator(ev=ev, min_dist_rad=min_dist_rad, inventory=inventory))
        mags.append((ev.preferred_magnitude()or ev.magnitude[-1]).mag)
        lats.append((ev.preferred_origin() or ev.origins[-1]).latitude)
        lons.append((ev.preferred_origin() or ev.origins[-1]).longitude)
        if (ev.preferred_origin() or ev.origins[-1]).depth_type == 'operator assigned':
            fixed.append(1)
        else:
            fixed.append(0)
            
    return min_az, depths, min_dist, mags, lats, lons, fixed
    
def plot_catalogue(cat, lats, lons, depths, mags, poly, bblons, bblats, splons, splats, region):
    
    #define map region
    fig = plt.figure(figsize=(12, 8))
    ax1=fig.add_subplot(1,1,1, projection=crs.Mercator())
    ax1.set_extent([min(lons)-0.5, max(lons)+0.5, min(lats)-0.5, max(lats)+0.5])
    ax1.add_feature(cfeature.COASTLINE)
    # plot earthquakes
    mappable = ax1.scatter(lons, lats, s=mags, c=depths, cmap='plasma_r', transform=crs.Geodetic())
    ax1.scatter(bblons, bblats, s=35, color='black', marker='^', transform=crs.Geodetic(), label='Existing GeoNet Broadband')
    ax1.scatter(splons, splats, s=35, color='black', marker='v', transform=crs.Geodetic(), label='Existing GeoNet Short Period')
    # plot polygon of region
    polyx, polyy = poly.exterior.xy
    ax1.plot(polyx, polyy, c='gray', transform=crs.Geodetic())
    # add legend
    fig.colorbar(mappable, label='Earthquake Depth (km)')
    plt.title(str(len(cat)) + ' events in ' + region + ' catalogue')
    
    return fig
    
def plot_azimuthal_map(lats, lons, min_az, bblons, bblats, splons, splats, poly, region):
 
    fig = plt.figure(figsize=(12, 8))
    ax1=fig.add_subplot(1,1,1, projection=crs.Mercator())
    #hack to force colour bar between 0 and 360
    minaz_c=[0,360]
    minazc=minaz_c+min_az[2:]
    #plot data
    mappable = ax1.scatter(lons, lats, s=10, c=minazc, cmap='coolwarm', transform=crs.Geodetic())
    ax1.scatter(bblons, bblats, s=35, color='black', marker='^', transform=crs.Geodetic(), label='Existing GeoNet Broadband')
    ax1.scatter(splons, splats, s=35, color='black', marker='v', transform=crs.Geodetic(), label='Existing GeoNet Short Period')
    #plot scale
    fig.colorbar(mappable, label='Maximum Azimuthal Gap, ($^\circ$)')
    ax1.legend(loc='upper left')
    ax1.set_extent([min(lons)-1.5, max(lons)+0.5, min(lats)-0.2, max(lats)+0.6])
    ax1.add_feature(cfeature.COASTLINE)
    # plot polygon of region
    polyx, polyy = poly.exterior.xy
    ax1.plot(polyx, polyy, c='gray', transform=crs.Geodetic())
    count = sum(1 for a in min_az if a<=180)
    plt.title(str(round(count/len(min_az)*100, 1)) + '% of events in ' + region + ' region satisfy the azimuthal gap criterion')
    
    return fig

def Min_Spick_dist(cat_poly):
    
    s_picks=[]
    for j, ev in enumerate(cat_poly):
        flags=[]
        for ar in (ev.preferred_origin() or ev.origins[-1]).arrivals:
            ##### THIS IS A FUDGE UNTIL THE BUG IN THE DISTANCE CALC USING THE INVENTORY IS FIXED
            if ar.phase =='S' and ar.distance*111.1 < 1.4*(ev.preferred_origin() or ev.origins[-1]).depth/1000:
                flags.append(1)
        if 1 in flags:
            s_picks.append(1)
        else:
            s_picks.append(0)
    
    return s_picks
    
def plot_depth_scatter(min_dist, s_picks, depths, fixed, region):
        
    p=Polygon([(0,0), (50,50), (0,50)])
    mindistg, depthsg, fixedg = [], [], []
    mindistS, depthsS, fixedS = [], [], []
    mindistSg, depthsSg, fixedSg = [], [], []
    for i, d in enumerate(min_dist):
        if p.contains(Point(d, depths[i])): 
            mindistg.append(d)
            depthsg.append(depths[i])
            fixedg.append(fixed[i])
        if s_picks[i] > 0:
            mindistS.append(d)
            depthsS.append(depths[i])
        if p.contains(Point(d, depths[i])) and s_picks[i] > 0:
            mindistSg.append(d)
            depthsSg.append(depths[i])
            
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_aspect('equal')
    ax.scatter(min_dist, depths, c='#B3589A', marker='x', s=8, label='bad: depth < min dist')
    ax.scatter(mindistS, depthsS, c='#F6D3E8', marker='o', s=8, label='OK: S-phase contained')
    ax.scatter(mindistg, depthsg, c='#BBD4A6', marker='o', s=12, label='good: depth > min dist')
    ax.scatter(mindistSg, depthsSg, c='#9BBF85', marker='o', s=12, label='Best: P+S constraints')
    ax.plot([0,50], [0,50], color='k', label='1:1 (P picks)')
    ax.plot([0,70], [0,50], color='k', linestyle='dashed', label='1:1.4 (S picks)')
    ax.set_xlabel('Minimum distance, km')
    ax.set_ylabel('Depth, km')
    ax.set_xlim(0,100)
    ax.set_ylim(50,0)
    plt.legend(loc='lower right')
    plt.title(str(round(len(depthsg)/len(depths)*100, 1)) + '% of earthquakes in ' 
              + region + ' have an acceptable minimum distance (less than depth)')

    divider = make_axes_locatable(ax)
    ax_histy = divider.append_axes("right", 4, pad=0.5, sharey=ax)

    bins = np.arange(0, 50, 5)
    ax_histy.hist(depths, bins=bins, orientation='horizontal', color='#B3589A', label='bad: depth < min dist')
    ax_histy.hist(depthsS, bins=bins, orientation='horizontal', color='#F6D3E8', label='OK: S-phase contained')
    ax_histy.hist(depthsg, bins=bins, orientation='horizontal', color='#BBD4A6', label='good: depth > min dist')
    ax_histy.hist(depthsSg, bins=bins, orientation='horizontal', color='#9BBF85', label='best: P+S constraints')

    ax_histy.set_xlabel('Number of earthquakes')
    plt.legend(loc='lower right')
    plt.title(str(round(fixed.count(1)/len(fixed)*100, 1)) + '% of events have fixed depth')
    
    return fig
    
def binary_counts(min_az, cat_poly, depths, min_dist, fixed):
    
    # min_azimuth requirement:
    min_az_bi=[]
    for az in min_az:
        if az<=180:
            min_az_bi.append(1)
        else:
            min_az_bi.append(0)
    #min picks and one spick requirement
    ps, min_picks =[], []
    for i, ev in enumerate(cat_poly):
        if len((ev.preferred_origin() or ev.origins[-1]).arrivals) < 8:
            min_picks.append(0)
        else:
            min_picks.append(1)
        rt=[arr.phase for arr in (ev.preferred_origin() or ev.origins[-1]).arrivals]
        if rt.count('S') < 1:
            ps.append(0)
        else:
            ps.append(1)
    #min_dist
    min_dist_bi=[]
    for i, d in enumerate(depths):
        if min_dist[i] <= d:
            min_dist_bi.append(1)
        else:
            min_dist_bi.append(0)
    #plus fixed and_spicks
    fixed_inv=[]
    for f in fixed:
        if f == 0:
            fixed_inv.append(1)
        else:
            fixed_inv.append(0)
        
    return min_az_bi, min_picks, ps, min_dist_bi, fixed_inv 
    
def plot_quality_criteria_scores(fixed_inv, cat_poly, s_picks, min_dist_bi, ps, min_picks, min_az_bi, region):

    criteria=['Not fixed', 'Min_Sdist$\leq$1.4*depth', 'Min_dist$\leq$depth', 
              'Min_Spicks$\geq1$','Min_picks$\geq$8', 'Min_az$\leq$180$^\circ$']
    counts=[fixed_inv.count(1)/len(cat_poly)*100, s_picks.count(1)/len(cat_poly)*100, min_dist_bi.count(1)/len(cat_poly)*100, 
           ps.count(1)/len(cat_poly)*100, min_picks.count(1)/len(cat_poly)*100, min_az_bi.count(1)/len(cat_poly)*100]
    countsfull=[100, 100, 100, 100, 100, 100]
    fig, ax=plt.subplots(figsize=(6,6))
    ax.barh(criteria, countsfull, color='#fee6ce', label='Not Satisfied')
    ax.barh(criteria, counts, color='#fdae6b', label='Satisfied')
    ax.legend(bbox_to_anchor=(1, 1.15))
    plt.title(region)

    ax.set_ylabel('Quality criteria')
    ax.set_xlabel('Percentage of events')
    ax.set_xlim([0, 100])
    plt.xticks(rotation=45)
    plt.show()

    return fig, counts, criteria
    
def plot_quality_score_map(quals, counts, lons, lats, region, poly, cat_poly, sums, bblons, bblats, splons, splats):
   
    fig = plt.figure(figsize=(12, 8))
    ax1=fig.add_subplot(1,1,1, projection=crs.Mercator())
    cmap = plt.get_cmap('coolwarm_r', np.max(quals) - np.min(quals) + 1)
    #plot data
    mappable = ax1.scatter(lons, lats, s=5, c=sums, cmap=cmap, vmin=np.min(quals) - 0.5, 
                          vmax=np.max(quals) + 0.5, transform=crs.Geodetic())
    ax1.scatter(bblons, bblats, s=35, color='black', marker='^', transform=crs.Geodetic(), label='Existing GeoNet Broadband')
    ax1.scatter(splons, splats, s=35, color='black', marker='v', transform=crs.Geodetic(), label='Existing GeoNet Short Period')
    ax1.legend(loc='lower right')
    ax1.set_extent([min(lons)-0.5, max(lons)+0.5, min(lats)-0.5, max(lats)+0.5])
    #add coastline, title and scale
    ax1.add_feature(cfeature.COASTLINE)
    plt.title('In ' + region + ' ' + str(round(counts[-1]/len(cat_poly)*100, 1)) + '% of events have a Quality Score = 6', loc='right')
    fig.colorbar(mappable, ticks=np.arange(np.min(quals), np.max(quals)+1), label='Quality Score')
    # plot polygon of region
    polyx, polyy = poly.exterior.xy
    ax1.plot(polyx, polyy, c='gray', transform=crs.Geodetic())
    labels=[str(n) for n in quals]
    #add in pie chart
    ax2=fig.add_axes([0.055, 0.48, 0.4, 0.4]) #left bottom width height
    colours=[cmap(n) for n in quals]
    ax2.pie(counts, labels=labels, labeldistance=.6, colors=colours, startangle=90, shadow=False)
   
    return fig 
    

def plot_quality_score_bar(region, countspc_cum, quals):

    fig, ax=plt.subplots(figsize=(10,1))
    cmap = plt.get_cmap('coolwarm_r', np.max(quals) - np.min(quals) + 1)
    colours=[cmap(n) for n in quals]
    ax.barh(region, countspc_cum[6], color=colours[6], label='QS6')
    ax.barh(region, countspc_cum[5], color=colours[5], label='QS5')
    ax.barh(region, countspc_cum[4], color=colours[4], label='QS4')
    ax.barh(region, countspc_cum[3], color=colours[3], label='QS3')
    ax.barh(region, countspc_cum[2], color=colours[2], label='QS2')
    ax.barh(region, countspc_cum[1], color=colours[1], label='QS1')
    ax.barh(region, countspc_cum[0], color=colours[0], label='QS0')
    ax.set_xlim([0, 100])
    ax.legend(bbox_to_anchor=(1, 1.15))
    
    return fig
    

