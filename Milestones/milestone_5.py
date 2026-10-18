import csv, os, glob, math
from collections import defaultdict
from itertools import combinations

BASE = os.path.dirname(os.path.abspath(__file__))

M5_EXTRA = os.path.join(BASE, "m5_extra_wafers.csv")
M1_DIR = os.path.join(BASE, "..", "Milestone1", "output")
M2_DIR = os.path.join(BASE, "..", "Milestone2", "output")
OUT = os.path.join(BASE, "output")

os.makedirs(OUT, exist_ok=True)

MIN_SUPPORT = 5
MIN_LIFT = 1.5
MAX_Q = 0.05


# ---------- CSV ----------

def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_data():
    rows = []

    for p in glob.glob(os.path.join(M1_DIR, "TestCase_1_*.csv")):
        rows += read_csv(p)

    for p in glob.glob(os.path.join(M2_DIR, "TestCase_2_*.csv")):
        rows += read_csv(p)

    rows += read_csv(M5_EXTRA)

    return rows


# ---------- Wafer -> Pattern Sets ----------

def build_wafer_sets(rows):
    wafers = defaultdict(set)

    for r in rows:
        w = r.get("waferID", "").strip()
        name = r.get("patternName", "").strip()
        bucket = r.get("bucket", "").strip()

        if w and name:
            wafers[w].add((name, bucket))

    return wafers


# ---------- Fisher Exact ----------

def log_comb(n, k):
    if k < 0 or k > n:
        return float("-inf")
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
    )


def hypergeom(N, K, n, x):
    if x < 0 or x > K or x > n or n - x > N - K:
        return 0.0

    return math.exp(
        log_comb(K, x)
        + log_comb(N - K, n - x)
        - log_comb(N, n)
    )


def fisher_greater(N, a, b, observed):
    upper = min(a, b)
    p = 0.0

    for x in range(observed, upper + 1):
        p += hypergeom(N, a, b, x)

    return min(max(p, 0.0), 1.0)


# ---------- Benjamini-Hochberg ----------

def bh(pairs):
    m = len(pairs)
    result = {}

    ordered = sorted(pairs, key=lambda x: x[1])
    prev = 1.0

    for rank in range(m, 0, -1):
        idx, p = ordered[rank - 1]
        q = min(prev, p * m / rank)
        result[idx] = q
        prev = q

    return result


# ---------- M5.1 ----------

def find_edges(wafers):
    N = len(wafers)

    support = defaultdict(int)
    pair_count = defaultdict(int)

    for patterns in wafers.values():

        for p in patterns:
            support[p] += 1

        for a, b in combinations(sorted(patterns), 2):
            pair_count[(a, b)] += 1

    candidates = []

    for (a, b), observed in pair_count.items():

        if observed < MIN_SUPPORT:
            continue

        expected = support[a] * support[b] / N

        if expected <= 0:
            continue

        lift = observed / expected

        if lift < MIN_LIFT:
            continue

        p = fisher_greater(
            N,
            support[a],
            support[b],
            observed
        )

        candidates.append((a, b, p))

    qvals = bh([
        (i, c[2])
        for i, c in enumerate(candidates)
    ])

    edges = []

    for i, (a, b, _) in enumerate(candidates):
        if qvals[i] <= MAX_Q:
            edges.append((a, b))

    return edges


def write_edges(edges):
    path = os.path.join(
        OUT,
        "TestCase_5_1.csv"
    )

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        w.writerow([
            "patternA",
            "patternA_Bucket",
            "patternB",
            "patternB_Bucket"
        ])

        for a, b in sorted(edges):
            w.writerow([
                a[0], a[1],
                b[0], b[1]
            ])


# ---------- M5.2 ----------

def build_graph(edges):
    g = defaultdict(set)

    for a, b in edges:
        g[a].add(b)
        g[b].add(a)

    return g


def bron_kerbosch(R, P, X, graph, result):

    if not P and not X:
        if len(R) >= 3:
            result.add(frozenset(R))
        return

    pivot = max(
        P | X,
        key=lambda v: len(graph[v]),
        default=None
    )

    choices = P - graph[pivot] if pivot else set(P)

    for v in list(choices):
        bron_kerbosch(
            R | {v},
            P & graph[v],
            X & graph[v],
            graph,
            result
        )
        P.remove(v)
        X.add(v)


def find_cliques(edges):
    graph = build_graph(edges)
    result = set()

    bron_kerbosch(
        set(),
        set(graph),
        set(),
        graph,
        result
    )

    return sorted(
        result,
        key=lambda c: (
            len(c),
            tuple(sorted(c))
        )
    )


def write_cliques(cliques):
    path = os.path.join(
        OUT,
        "TestCase_5_2.csv"
    )

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        w.writerow([
            "cliqueID",
            "memberOrder",
            "patternName",
            "bucket"
        ])

        for i, clique in enumerate(cliques, 1):

            for order, member in enumerate(
                sorted(clique), 1
            ):

                w.writerow([
                    f"C{i}",
                    order,
                    member[0],
                    member[1]
                ])


# ---------- MAIN ----------

def main():

    rows = load_data()

    if not rows:
        print("ERROR: No input data found.")
        return

    wafers = build_wafer_sets(rows)

    edges = find_edges(wafers)
    write_edges(edges)

    cliques = find_cliques(edges)
    write_cliques(cliques)

    print("Milestone 5 completed.")
    print("Wafers :", len(wafers))
    print("Edges  :", len(edges))
    print("Cliques:", len(cliques))
    print("Output :", OUT)


if __name__ == "__main__":
    main()