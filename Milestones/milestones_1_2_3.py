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
    notch = metadata.get("notchPosition", "Bottom")

    points = []

    for d in defects:
        try:
            x = float(d["x"])
            y = float(d["y"])
        except (KeyError, TypeError, ValueError):
            continue

        r = math.hypot(x, y)

        if r <= RADIUS_LIMIT:
            points.append([x, y])

    return wafer_id, lot_id, notch, points


def align_points(points, notch):
    notch_angles = {
        "Right": 0.0,
        "Top": 90.0,
        "Left": 180.0,
        "Bottom": -90.0
    }

    current = notch_angles.get(notch, -90.0)
    rotation = math.radians(-90.0 - current)

    c = math.cos(rotation)
    s = math.sin(rotation)

    aligned = []

    for x, y in points:
        nx = x * c - y * s
        ny = x * s + y * c
        aligned.append([nx, ny])

    return aligned


def detect_rings(points, used):
    if len(points) < MIN_RING_POINTS:
        return []

    radii = [
        math.hypot(points[i][0], points[i][1])
        for i in range(len(points))
    ]

    sorted_indices = sorted(
        range(len(points)),
        key=lambda i: radii[i]
    )

    sorted_radii = [radii[i] for i in sorted_indices]

    patterns = []
    previous_candidate = None

    for pos, original_idx in enumerate(sorted_indices):

        if used[original_idx]:
            continue

        candidate_r = sorted_radii[pos]

        if (
            previous_candidate is not None
            and abs(candidate_r - previous_candidate) < 0.5
        ):
            continue

        previous_candidate = candidate_r

        left = bisect_left(
            sorted_radii,
            candidate_r - RING_BAND
        )

        right = bisect_right(
            sorted_radii,
            candidate_r + RING_BAND
        )

        matched = []

        for p in range(left, right):
            idx = sorted_indices[p]

            if not used[idx]:
                matched.append(idx)

        if len(matched) < MIN_RING_POINTS:
            continue

        r_mean = sum(
            radii[i] for i in matched
        ) / len(matched)

        valid = True

        for i in matched:
            if abs(radii[i] - r_mean) > RING_BAND:
                valid = False
                break

        if not valid:
            continue

        variance = sum(
            (radii[i] - r_mean) ** 2
            for i in matched
        ) / len(matched)

        sigma = math.sqrt(variance)

        if sigma > MAX_RADIAL_SIGMA:
            continue

        sector_mask = 0

        for i in matched:
            x, y = points[i]

            angle = math.atan2(y, x)

            if angle < 0:
                angle += 2.0 * math.pi

            sector = int(
                angle / (2.0 * math.pi) * TOTAL_SECTORS
            )

            if sector >= TOTAL_SECTORS:
                sector = TOTAL_SECTORS - 1

            sector_mask |= (1 << sector)

        sector_count = sector_mask.bit_count()

        if sector_count < MIN_SECTORS:
            continue

        ratio = r_mean / RADIUS_LIMIT

        if 0.20 <= ratio < 0.80:
            pattern_name = "Ring"

            if ratio < 0.50:
                bucket = "Inner"
            elif ratio < 0.70:
                bucket = "Mid"
            else:
                bucket = "Outer"

        elif 0.80 <= ratio <= 0.95:
            pattern_name = "EdgeRing"
            bucket = ""

        else:
            continue

        patterns.append({
            "patternName": pattern_name,
            "bucket": bucket,
            "indices": matched,
            "r_mean": r_mean
        })

        for i in matched:
            used[i] = True

    return patterns


def build_components(points, used):
    n = len(points)

    if n < MIN_COMPONENT_SIZE:
        return []

    active = [
        i for i in range(n)
        if not used[i]
    ]

    if len(active) < MIN_COMPONENT_SIZE:
        return []

    grid = {}

    for i in active:
        x, y = points[i]

        cell_x = math.floor(x / CONNECT_DISTANCE)
        cell_y = math.floor(y / CONNECT_DISTANCE)

        key = (cell_x, cell_y)

        if key not in grid:
            grid[key] = []

        grid[key].append(i)

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

                    cell = (cx + dx, cy + dy)

                    neighbours = grid.get(cell)

                    if neighbours is None:
                        continue

                    for nxt in neighbours:

                        if nxt in visited:
                            continue

                        nx, ny = points[nxt]

                        ddx = x - nx
                        ddy = y - ny

                        if (
                            ddx * ddx +
                            ddy * ddy
                            <= CONNECT_DISTANCE2
                        ):
                            visited.add(nxt)
                            queue.append(nxt)

        if len(component) >= MIN_COMPONENT_SIZE:
            components.append(component)

    return components


def component_pca(points, component):
    arr = np.array(
        [points[i] for i in component],
        dtype=np.float64
    )

    center = arr.mean(axis=0)
    centered = arr - center

    cov = np.dot(centered.T, centered)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    major_index = int(np.argmax(eigenvalues))
    major = eigenvectors[:, major_index]

    minor_index = 1 - major_index
    minor = eigenvectors[:, minor_index]

    major_projection = centered @ major
    minor_projection = centered @ minor

    length = (
        float(np.max(major_projection))
        - float(np.min(major_projection))
    )

    width = (
        float(np.max(minor_projection))
        - float(np.min(minor_projection))
    )

    if width <= 1e-12:
        aspect = float("inf")
    else:
        aspect = length / width

    angle = math.degrees(
        math.atan2(major[1], major[0])
    )

    angle %= 180.0

    return (
        center,
        length,
        width,
        aspect,
        angle
    )


def detect_clusters_and_scratches(points, used):
    components = build_components(points, used)

    patterns = []

    for component in components:

        center, length, width, aspect, angle = component_pca(
            points,
            component
        )

        cx = float(center[0])
        cy = float(center[1])

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

            if (
                angle < 22.5
                or angle >= 157.5
            ):
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
        for i in pattern["indices"]:
            used[i] = True

    return patterns


def process_wafer(json_path):
    wafer_id, lot_id, notch, points = load_wafer(json_path)

    points = align_points(points, notch)

    n = len(points)
    used = [False] * n

    patterns = []

    ring_patterns = detect_rings(
        points,
        used
    )

    patterns.extend(ring_patterns)

    other_patterns = detect_clusters_and_scratches(
        points,
        used
    )

    patterns.extend(other_patterns)

    return wafer_id, lot_id, patterns


def process_lot(lot_folder):
    json_files = sorted(
        lot_folder.glob("*.json"),
        key=lambda p: natural_key(p.name)
    )

    wafer_results = []
    lot_id_from_metadata = None

    for json_file in json_files:

        wafer_id, lot_id, patterns = process_wafer(
            json_file
        )

        if lot_id:
            lot_id_from_metadata = lot_id

        wafer_results.append({
            "waferID": wafer_id,
            "patterns": patterns
        })

    total_wafers = len(wafer_results)

    lot_id = (
        lot_id_from_metadata
        if lot_id_from_metadata
        else lot_folder.name
    )

    support = {}

    for wafer in wafer_results:

        wafer_id = wafer["waferID"]
        seen_in_wafer = set()

        for pattern in wafer["patterns"]:

            key = (
                pattern["patternName"],
                pattern["bucket"]
            )

            if key in seen_in_wafer:
                continue

            seen_in_wafer.add(key)

            if key not in support:
                support[key] = []

            support[key].append(wafer_id)

    rows = []

    for (pattern_name, bucket), wafers in support.items():

        wafer_support = len(wafers)

        is_recurring = (
            wafer_support / total_wafers
            >= RECURRENCE_THRESHOLD
            if total_wafers > 0
            else False
        )

        if is_recurring:
            scope = "Lot"
            recurring_text = "Yes"
        else:
            scope = "Wafer"
            recurring_text = "No"

        wafers_sorted = sorted(
            wafers,
            key=natural_key
        )

        rows.append([
            lot_id,
            pattern_name,
            bucket,
            scope,
            wafer_support,
            total_wafers,
            recurring_text,
            ";".join(wafers_sorted)
        ])

    rows.sort(
        key=lambda r: (
            r[1],
            r[2]
        )
    )

    return lot_id, total_wafers, rows


def testcase_number(folder_name):
    match = re.search(
        r"(\d+)",
        folder_name
    )

    if match:
        return int(match.group(1))

    return 999999


def write_csv(
    output_path,
    lot_id,
    total_wafers,
    rows
):
    header = [
        "lotID",
        "patternName",
        "bucket",
        "scope",
        "waferSupport",
        "totalWafers",
        "isRecurring",
        "supportingWafers"
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(header)

        if rows:
            writer.writerows(rows)

        else:
            writer.writerow([
                lot_id,
                "",
                "",
                "",
                0,
                total_wafers,
                "No",
                ""
            ])


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    testcase_folders = [
        p for p in BASE_DIR.iterdir()
        if p.is_dir()
        and p.name != "output"
        and not p.name.startswith(".")
    ]

    testcase_folders.sort(
        key=lambda p: testcase_number(p.name)
    )

    if not testcase_folders:
        print("No testcase folders found.")
        return

    print("=" * 60)
    print("        WAFER DNA - MILESTONE 3")
    print("=" * 60)

    for folder in testcase_folders:

        number = testcase_number(
            folder.name
        )

        print()
        print(
            f"Processing {folder.name} ..."
        )

        try:

            lot_id, total_wafers, rows = process_lot(
                folder
            )

            output_file = (
                OUTPUT_DIR
                / f"TestCase_3_{number}.csv"
            )

            write_csv(
                output_file,
                lot_id,
                total_wafers,
                rows
            )

            print(
                f"  Lot ID       : {lot_id}"
            )

            print(
                f"  Total wafers : {total_wafers}"
            )

            print(
                f"  Patterns     : {len(rows)}"
            )

            print(
                f"  Output       : {output_file}"
            )

        except Exception as e:

            print(
                f"  ERROR in {folder.name}: {e}"
            )

    print()
    print("=" * 60)
    print("All testcases processed.")
    print("=" * 60)


if __name__ == "__main__":
    main()