import json
import csv
import math
import re
from pathlib import Path
from bisect import bisect_left, bisect_right
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

RADIUS_LIMIT = 147.0
RING_BAND = 5.0
MIN_RING_POINTS = 45
MAX_RADIAL_SIGMA = 3.0
MIN_SECTORS = 19
TOTAL_SECTORS = 24

CONNECT_DISTANCE = 11.0
CONNECT_DISTANCE2 = CONNECT_DISTANCE * CONNECT_DISTANCE
MIN_COMPONENT_SIZE = 15

CLUSTER_MAX_ASPECT = 2.0
SCRATCH_MIN_LENGTH = 60.0
SCRATCH_MAX_WIDTH = 4.0
SCRATCH_MIN_ASPECT = 6.0

RECURRENCE_THRESHOLD = 0.60
TREND_TOLERANCE = 0.10

NOTCH_ANGLES = {
    "Right": 0.0,
    "Top": 90.0,
    "Left": 180.0,
    "Bottom": -90.0
}


def natural_key(text):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", str(text))
    ]


def load_wafer(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    defects = data.get("defectList", [])

    wafer_id = metadata.get("waferID", path.stem)
    lot_id = metadata.get("lotID", "")
    process_id = metadata.get("processID", "")
    lot_sequence = metadata.get("lotSequence", 0)
    notch = metadata.get("notchPosition", "Bottom")

    points = []

    for defect in defects:
        try:
            x = float(defect["x"])
            y = float(defect["y"])
        except (KeyError, TypeError, ValueError):
            continue

        if math.hypot(x, y) <= RADIUS_LIMIT:
            points.append((x, y))

    return wafer_id, lot_id, process_id, lot_sequence, notch, points


def align_points(points, notch):
    current_angle = NOTCH_ANGLES.get(notch, -90.0)
    rotation = math.radians(-90.0 - current_angle)
    c = math.cos(rotation)
    s = math.sin(rotation)

    return [
        (x * c - y * s, x * s + y * c)
        for x, y in points
    ]


def detect_rings(points, used):
    n = len(points)

    if n < MIN_RING_POINTS:
        return []

    radii = [math.hypot(x, y) for x, y in points]
    order = sorted(range(n), key=radii.__getitem__)
    sorted_radii = [radii[i] for i in order]

    patterns = []
    previous = None

    for pos in range(n):
        idx = order[pos]

        if used[idx]:
            continue

        candidate = sorted_radii[pos]

        if previous is not None and abs(candidate - previous) < 0.5:
            continue

        previous = candidate

        left = bisect_left(
            sorted_radii,
            candidate - RING_BAND
        )
        right = bisect_right(
            sorted_radii,
            candidate + RING_BAND
        )

        matched = [
            order[p]
            for p in range(left, right)
            if not used[order[p]]
        ]

        if len(matched) < MIN_RING_POINTS:
            continue

        mean_r = sum(radii[i] for i in matched) / len(matched)

        if any(
            abs(radii[i] - mean_r) > RING_BAND
            for i in matched
        ):
            continue

        sigma = math.sqrt(
            sum(
                (radii[i] - mean_r) ** 2
                for i in matched
            ) / len(matched)
        )

        if sigma > MAX_RADIAL_SIGMA:
            continue

        mask = 0

        for i in matched:
            x, y = points[i]
            angle = math.atan2(y, x)

            if angle < 0:
                angle += 2 * math.pi

            sector = int(
                angle / (2 * math.pi) * TOTAL_SECTORS
            )

            sector = min(
                sector,
                TOTAL_SECTORS - 1
            )

            mask |= 1 << sector

        if mask.bit_count() < MIN_SECTORS:
            continue

        ratio = mean_r / RADIUS_LIMIT

        if 0.20 <= ratio < 0.80:
            name = "Ring"

            if ratio < 0.50:
                bucket = "Inner"
            elif ratio < 0.70:
                bucket = "Mid"
            else:
                bucket = "Outer"

        elif 0.80 <= ratio <= 0.95:
            name = "EdgeRing"
            bucket = ""

        else:
            continue

        patterns.append({
            "patternName": name,
            "bucket": bucket,
            "indices": matched
        })

        for i in matched:
            used[i] = True

    return patterns


def build_components(points, used):
    active = [
        i for i in range(len(points))
        if not used[i]
    ]

    if len(active) < MIN_COMPONENT_SIZE:
        return []

    grid = {}

    for i in active:
        x, y = points[i]
        cell = (
            math.floor(x / CONNECT_DISTANCE),
            math.floor(y / CONNECT_DISTANCE)
        )
        grid.setdefault(cell, []).append(i)

    visited = set()
    components = []

    for start in active:
        if start in visited:
            continue

        visited.add(start)
        queue = [start]
        component = []
        head = 0

        while head < len(queue):
            current = queue[head]
            head += 1
            component.append(current)

            x, y = points[current]

            cx = math.floor(x / CONNECT_DISTANCE)
            cy = math.floor(y / CONNECT_DISTANCE)

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbours = grid.get(
                        (cx + dx, cy + dy)
                    )

                    if neighbours is None:
                        continue

                    for nxt in neighbours:
                        if nxt in visited:
                            continue

                        nx, ny = points[nxt]
                        ddx = x - nx
                        ddy = y - ny

                        if ddx * ddx + ddy * ddy <= CONNECT_DISTANCE2:
                            visited.add(nxt)
                            queue.append(nxt)

        if len(component) >= MIN_COMPONENT_SIZE:
            components.append(component)

    return components


def component_pca(points, component):
    arr = np.asarray(
        [points[i] for i in component],
        dtype=np.float64
    )

    center = arr.mean(axis=0)
    centered = arr - center
    covariance = centered.T @ centered

    values, vectors = np.linalg.eigh(covariance)

    major_index = int(np.argmax(values))
    minor_index = 1 - major_index

    major = vectors[:, major_index]
    minor = vectors[:, minor_index]

    major_projection = centered @ major
    minor_projection = centered @ minor

    length = float(
        major_projection.max() -
        major_projection.min()
    )

    width = float(
        minor_projection.max() -
        minor_projection.min()
    )

    aspect = (
        float("inf")
        if width <= 1e-12
        else length / width
    )

    angle = math.degrees(
        math.atan2(major[1], major[0])
    ) % 180.0

    return center, length, width, aspect, angle


def detect_clusters_and_scratches(points, used):
    components = build_components(points, used)
    patterns = []

    for component in components:
        center, length, width, aspect, angle = component_pca(
            points,
            component
        )

        cx, cy = center

        if aspect <= CLUSTER_MAX_ASPECT:
            if cx >= 0 and cy >= 0:
                bucket = "Q1"
            elif cx < 0 and cy >= 0:
                bucket = "Q2"
            elif cx < 0 and cy < 0:
                bucket = "Q3"
            else:
                bucket = "Q4"

            patterns.append({
                "patternName": "Cluster",
                "bucket": bucket,
                "indices": component
            })

        elif (
            length >= SCRATCH_MIN_LENGTH
            and width <= SCRATCH_MAX_WIDTH
            and aspect >= SCRATCH_MIN_ASPECT
        ):
            if angle < 22.5 or angle >= 157.5:
                bucket = "H"
            elif angle < 67.5:
                bucket = "D1"
            elif angle < 112.5:
                bucket = "V"
            else:
                bucket = "D2"

            patterns.append({
                "patternName": "Scratch",
                "bucket": bucket,
                "indices": component
            })

    for pattern in patterns:
        for index in pattern["indices"]:
            used[index] = True

    return patterns


def detect_wafer(points, notch):
    points = align_points(points, notch)
    used = [False] * len(points)

    patterns = detect_rings(
        points,
        used
    )

    patterns.extend(
        detect_clusters_and_scratches(
            points,
            used
        )
    )

    return patterns


def process_lot(files):
    lot_id = ""
    process_id = ""
    lot_sequence = 0
    wafer_support = {}
    intensity = {}
    wafer_count = 0

    for path in files:
        (
            wafer_id,
            current_lot,
            current_process,
            current_sequence,
            notch,
            points
        ) = load_wafer(path)

        if current_lot:
            lot_id = current_lot

        if current_process:
            process_id = current_process

        if current_sequence is not None:
            lot_sequence = current_sequence

        wafer_count += 1

        patterns = detect_wafer(
            points,
            notch
        )

        wafer_seen = set()

        for pattern in patterns:
            key = (
                pattern["patternName"],
                pattern["bucket"]
            )

            count = len(
                pattern["indices"]
            )

            intensity[key] = (
                intensity.get(key, 0)
                + count
            )

            if key not in wafer_seen:
                wafer_seen.add(key)

                wafer_support[key] = (
                    wafer_support.get(key, 0)
                    + 1
                )

    return {
        "lotID": lot_id,
        "processID": process_id,
        "lotSequence": lot_sequence,
        "waferCount": wafer_count,
        "waferSupport": wafer_support,
        "intensity": intensity
    }


def read_testcase(folder):
    lots = {}

    files = sorted(
        folder.glob("*.json"),
        key=lambda p: natural_key(p.name)
    )

    for path in files:
        try:
            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:
                data = json.load(f)

            metadata = data.get(
                "metadata",
                {}
            )

            lot_id = metadata.get(
                "lotID",
                path.stem
            )

            lots.setdefault(
                lot_id,
                []
            ).append(path)

        except Exception as e:
            print(
                f"Skipping {path.name}: {e}"
            )

    return lots


def is_lot_scope(lot, pattern):
    total = lot["waferCount"]

    if total == 0:
        return False

    support = lot[
        "waferSupport"
    ].get(pattern, 0)

    return support * 100 >= total * 60


def calculate_trend(series):
    if len(series) <= 1:
        return "Stable"

    average = sum(series) / len(series)
    tolerance = average * TREND_TOLERANCE

    increasing = False
    decreasing = False

    for i in range(1, len(series)):
        diff = series[i] - series[i - 1]

        if diff > tolerance:
            increasing = True
        elif diff < -tolerance:
            decreasing = True

    if increasing and decreasing:
        return "Fluctuating"

    if increasing:
        return "Increasing"

    if decreasing:
        return "Decreasing"

    return "Stable"


def process_process(process_id, lots):
    lots.sort(
        key=lambda lot: (
            lot["lotSequence"],
            natural_key(lot["lotID"])
        )
    )

    total_lots = len(lots)

    if total_lots == 0:
        return []

    all_patterns = set()

    for lot in lots:
        all_patterns.update(
            lot["waferSupport"].keys()
        )

    rows = []

    for pattern in sorted(
        all_patterns,
        key=lambda x: (x[0], x[1])
    ):
        supporting_lots = []

        for lot in lots:
            if is_lot_scope(
                lot,
                pattern
            ):
                supporting_lots.append(lot)

        lot_support = len(
            supporting_lots
        )

        if lot_support * 100 >= total_lots * 60:
            process_scope = "Process"
        else:
            process_scope = "NotAcrossProcess"

        series = [
            lot["intensity"].get(
                pattern,
                0
            )
            for lot in lots
        ]

        if process_scope == "Process":
            trend = calculate_trend(series)
        else:
            trend = "N/A"

        rows.append({
            "processID": process_id,
            "patternName": pattern[0],
            "bucket": pattern[1],
            "processScope": process_scope,
            "lotSupport": lot_support,
            "totalLots": total_lots,
            "supportingLots": [
                lot["lotID"]
                for lot in supporting_lots
            ],
            "lotTrend": trend
        })

    return rows


def testcase_number(name):
    match = re.search(
        r"(\d+)",
        name
    )

    return (
        int(match.group(1))
        if match
        else 999999
    )


def write_csv(path, rows):
    header = [
        "processID",
        "patternName",
        "bucket",
        "processScope",
        "lotSupport",
        "totalLots",
        "supportingLots",
        "lotTrend"
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in rows:
            writer.writerow([
                row["processID"],
                row["patternName"],
                row["bucket"],
                row["processScope"],
                row["lotSupport"],
                row["totalLots"],
                ";".join(
                    row["supportingLots"]
                ),
                row["lotTrend"]
            ])


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    testcase_folders = sorted(
        [
            folder
            for folder in BASE_DIR.iterdir()
            if (
                folder.is_dir()
                and folder.name != "output"
                and not folder.name.startswith(".")
            )
        ],
        key=lambda folder:
        testcase_number(folder.name)
    )

    for testcase_folder in testcase_folders:
        number = testcase_number(
            testcase_folder.name
        )

        lots_files = read_testcase(
            testcase_folder
        )

        lots = []

        for lot_id, files in lots_files.items():
            lot = process_lot(files)

            if not lot["lotID"]:
                lot["lotID"] = lot_id

            lots.append(lot)

        processes = {}

        for lot in lots:
            process_id = (
                lot["processID"]
                or "UNKNOWN"
            )

            processes.setdefault(
                process_id,
                []
            ).append(lot)

        rows = []

        for process_id, process_lots in processes.items():
            rows.extend(
                process_process(
                    process_id,
                    process_lots
                )
            )

        rows.sort(
            key=lambda row: (
                natural_key(row["processID"]),
                row["patternName"],
                row["bucket"]
            )
        )

        output_path = (
            OUTPUT_DIR /
            f"TestCase_4_{number}.csv"
        )

        write_csv(
            output_path,
            rows
        )

        print(
            f"{testcase_folder.name} -> "
            f"{output_path.name}"
        )


if __name__ == "__main__":
    main()