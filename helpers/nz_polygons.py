# Polygon function to retrieve New Zealand regional polygons used in NZJGG paper

from shapely.geometry import Polygon

KNOWN_REGIONS = {
    "South_Island":
        Polygon([(174.545688, -40.923890), (174.610043, -41.864054), (173.773887, -42.738016), (173.705631, -43.540129),
                 (173.200880,-44.052885), (172.362586,-44.156868), (171.669022, -44.442912), (171.352573,-45.151559), 
                 (170.966445, -45.858293), (169.889561, -46.695356), (168.931761, -46.893336), (166.773235, -46.520716), 
                 (166.089537, -45.829983), (166.339861, -45.325290), (168.052698, -44.015045), (170.673622, -42.671434),
                 (171.353355, -41.512853), (171.627393, -41.311482), (172.063509, -40.540628), (173.066875, -40.011358),
                 (174.091970, -40.481465)]),
    "North_Island":
        Polygon([(175.335698, -41.762387), (176.429361, -41.124853), (177.340701, -39.667215), (178.059350, -39.316943), 
                 (178.515786, -38.538497), (178.845089, -37.589717), (178.005820, -37.367731), (177.403344, -37.772279), 
                 (176.302243, -37.480241), (176.018331, -36.299599), (175.690780, -35.878365), (174.967883, -35.62052),
                 (174.326535, -34.864621), (173.140509, -34.223545), (172.484181, -34.285083), (172.802336, -35.292550),
                 (173.847132, -36.565495), (174.524295, -37.687032), (174.375649, -38.633822), (173.599097, -39.063196),
                 (173.589833, -39.482075), (174.785174, -40.075416), (174.889237, -40.545537), (174.545688, -40.923890),
                 (174.609166, -41.487431)]),
    "West_Coast":
        Polygon([(168.052698,-44.015045), (169.139358,-44.708077), (169.559063,-44.371633),(171.993863, -42.829153),
                 (172.962716, -41.769541), (171.627393, -41.311482), (171.353355, -41.512853), (170.673622, -42.671434)]),
    "Fiordland":
        Polygon([(168.052698, -44.015045), (169.13935,-44.708077), (166.773235, -46.520716), (166.089537, -45.829983), 
                (166.339861, -45.325290)]),
    "Southland_Otago":
        Polygon([(166.773235, -46.520716), (169.139358, -44.708077), (169.559063, -44.371633), (171.352573, -45.151559), 
                (170.966445, -45.858293), (169.889561 ,-46.695356), (168.931761, -46.893336)]),
    "Canterbuty":
        Polygon([(171.352573, -45.151559), (169.559063, -44.371633), (171.993863, -42.829153), (172.098890, -42.718091),
                 (173.773887,-42.738016), (173.705631, -43.540129), (173.200880, -44.052885), (172.362586, -44.156868),
                 (171.669022, -44.442912)]),
    "Nelson":
        Polygon([(172.962716, -41.769541), (171.627393, -41.311482), (172.063509, -40.540628), (173.066875, -40.011358),
                 (174.091970, -40.481465)]),
    "Marlborough":
        Polygon([(173.773887, -42.738016), (172.098890, -42.718091), (172.962716, -41.769541), (174.091970, -40.481465),
                 (174.545688, -40.923890), (174.610043, -41.864054)]),
    "Wellington":
        Polygon([(174.545688, -40.923890), (174.586903, -41.487268), (175.280362, -41.798212), (176.145439, -41.328410),
                 (177.02452, -40.203529), (175.817332, -39.599731), (175.205539, -39.354601), (174.805069, -40.060709),
                 (175.021339, -40.442173)]),
    "Taranaki":
        Polygon([(174.805069, -40.060709), (175.205539, -39.354601), (175.581459, -38.662912), (174.480166, -38.455052),
                 (174.284442, -38.806351), (173.604021, -39.047112), (173.639303, -39.577763)]),
    "HawkesBay":
        Polygon([(175.817332, -39.599731), (177.024528, -40.203529), (177.288841, -39.695659), (177.181510, -39.405355),
                 (177.482652,-39.245538), (177.930426, -39.439514), (178.190348, -39.088646), (176.607131, -38.517333)]),
    "Gisborne":
        Polygon([(178.190348,-39.088646), (178.536913, -38.498493), (178.837577, -37.538249), (177.810457, -37.348858), 
                 (177.167183, -37.725261), (176.607131, -38.517333)]),
    "Volcanoes":
        Polygon([(175.581459, -38.662912), (175.205539, -39.354601), (175.817332, -39.599731), (176.607131, -38.517333),
                 (177.167183,-37.725261), (176.271503, -37.429597)]),
    "Auckland":
        Polygon([(174.480166, -38.455052), (175.581459, -38.662912), (176.271503, -37.429597), (176.276068, -36.678515),
                 (175.023640, -35.442502), (173.334701, -34.209682), (172.215717, -34.426598), (174.259874, -37.211643)])

}


def extract_polygon(region: str):
    poly = KNOWN_REGIONS.get(region, None)
    if poly is None:
        raise NotImplementedError(f"{region} is not in known regions: {KNOWN_REGIONS.keys()}")
    return poly

