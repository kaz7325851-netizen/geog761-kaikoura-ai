import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def perpendicular_distance(point, start, end):
    if start == end:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    nearest = (start[0] + t * dx, start[1] + t * dy)
    return math.hypot(point[0] - nearest[0], point[1] - nearest[1])


def simplify(points, tolerance=0.0005):
    if len(points) <= 2:
        return points
    start, end = points[0], points[-1]
    distances = [perpendicular_distance(p, start, end) for p in points[1:-1]]
    if not distances:
        return [start, end]
    maximum = max(distances)
    index = distances.index(maximum) + 1
    if maximum > tolerance:
        left = simplify(points[: index + 1], tolerance)
        right = simplify(points[index:], tolerance)
        return left[:-1] + right
    return [start, end]


def clipped_parts(geometry, predicate):
    parts = []
    current = []
    for node in geometry:
        point = [round(float(node["lon"]), 7), round(float(node["lat"]), 7)]
        if predicate(point[0], point[1]):
            current.append(point)
        else:
            if len(current) >= 2:
                parts.append(simplify(current))
            current = []
    if len(current) >= 2:
        parts.append(simplify(current))
    return parts


def endpoint_key(point):
    return (round(point[0], 6), round(point[1], 6))


def merge_connected(lines):
    unique = []
    seen = set()
    for line in lines:
        forward = tuple(map(tuple, line))
        reverse = tuple(reversed(forward))
        key = min(forward, reverse)
        if key not in seen:
            seen.add(key)
            unique.append(line)

    merged = []
    while unique:
        current = unique.pop()
        changed = True
        while changed:
            changed = False
            for index, candidate in enumerate(unique):
                if endpoint_key(current[-1]) == endpoint_key(candidate[0]):
                    current.extend(candidate[1:])
                elif endpoint_key(current[-1]) == endpoint_key(candidate[-1]):
                    current.extend(reversed(candidate[:-1]))
                elif endpoint_key(current[0]) == endpoint_key(candidate[-1]):
                    current = candidate[:-1] + current
                elif endpoint_key(current[0]) == endpoint_key(candidate[0]):
                    current = list(reversed(candidate[1:])) + current
                else:
                    continue
                unique.pop(index)
                changed = True
                break
        merged.append(simplify(current))
    return merged


with (ROOT / "osm_major_roads_raw.json").open(encoding="utf-8") as stream:
    sh1_data = json.load(stream)
with (ROOT / "osm_inland_roads_raw.json").open(encoding="utf-8") as stream:
    inland_data = json.load(stream)

sh1_lines = []
inland_lines = []

for element in sh1_data["elements"]:
    for part in clipped_parts(
        element.get("geometry", []),
        lambda lon, lat: 173.35 <= lon <= 174.08 and -42.60 <= lat <= -41.95,
    ):
        sh1_lines.append(part)

for element in inland_data["elements"]:
    for part in clipped_parts(
        element.get("geometry", []),
        lambda lon, lat: 173.00 <= lon <= 173.70 and -42.70 <= lat <= -42.30,
    ):
        inland_lines.append(part)

sh1_lines = merge_connected(sh1_lines)
inland_lines = merge_connected(inland_lines)
lines = sh1_lines + inland_lines
routes = (["SH1 Oaro to Kekerengu"] * len(sh1_lines) +
          ["Route 70 Kaikoura to Waiau"] * len(inland_lines))

features = [
    {
        "type": "Feature",
        "properties": {"route": route},
        "geometry": {"type": "LineString", "coordinates": coordinates},
    }
    for route, coordinates in zip(routes, lines)
]

geojson = {"type": "FeatureCollection", "features": features}
with (ROOT / "study_route_lines.geojson").open("w", encoding="utf-8") as stream:
    json.dump(geojson, stream, ensure_ascii=False, indent=2)

with (ROOT / "study_route_geometry.js").open("w", encoding="utf-8") as stream:
    stream.write("var routeLines = ee.Geometry.MultiLineString(")
    json.dump(lines, stream, separators=(",", ":"))
    stream.write(");\n")
    stream.write("var studyArea = routeLines.buffer(2000);\n")

print(f"Selected {len(lines)} road segments with {sum(len(line) for line in lines)} simplified vertices")
