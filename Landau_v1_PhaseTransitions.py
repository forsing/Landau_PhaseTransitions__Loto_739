#!/usr/bin/env python3


"""Landau-inspired ranking of complete 7/39 configurations; deterministic v1."""

import argparse
import csv
import itertools
from pathlib import Path
import numpy as np


DEFAULT = "/Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv"
# DEFAULT = "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv"
# DEFAULT = "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv"


N, K, START = 39, 7, 32


def load(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        for line, row in enumerate(csv.reader(f), 1):
            if not row or all(not v.strip() for v in row):
                continue
            try:
                values = [int(v.strip()) for v in row]
            except ValueError as e:
                raise ValueError(
                    f"Red {line}: ocekujem sedam celih brojeva."
                ) from e
            if (len(values) != K or values != sorted(values)
                    or len(set(values)) != K
                    or not all(1 <= v <= N for v in values)):
                raise ValueError(
                    f"Red {line}: neispravna sortirana 7/39 kombinacija."
                )
            rows.append(values)
    return np.asarray(rows, dtype=float).reshape(-1, K)


def gaps(x, n=N):
    return np.diff(np.r_[0., x, float(n + 1)]) - 1.


def fields(z, t):
    last = z[t - 1] - .5
    return np.r_[
        last,
        z[t - 2] - .5,
        z[t - 8:t].mean(axis=0) - .5,
        z[t - 32:t].mean(axis=0) - .5,
        last**2,
        last**3,
    ]


def design(rows):
    z = rows / N
    return np.asarray([
        fields(z, t) for t in range(START, len(rows) + 1)
    ])


def predict_field(rows, features, t, alpha):
    # Samo redovi pre ciljnog reda t ulaze u podesavanje.
    begin = max(START, t - 1024)
    x = features[begin - START:t - START]
    y = rows[begin:t]

    center = x.mean(axis=0)
    spread = np.maximum(x.std(axis=0), 1e-8)
    a = np.column_stack((
        np.ones(len(x)),
        (x - center) / spread,
    ))
    penalty = np.diag(np.r_[
        0., np.full(x.shape[1], alpha)
    ])
    coef = np.linalg.solve(a.T @ a + penalty, a.T @ y)
    fitted = a @ coef

    query = np.r_[
        1., (features[t - START] - center) / spread
    ]
    target = query @ coef

    # Robusne skale gresaka; nema pretpostavljene normalne raspodele.
    residual = y - fitted
    pos_scale = np.maximum(
        1., np.median(np.abs(residual), axis=0)
    )
    gap_residual = np.diff(
        np.column_stack((
            np.zeros(len(y)), residual, np.zeros(len(y))
        )),
        axis=1,
    )
    gap_scale = np.maximum(
        1., np.median(np.abs(gap_residual), axis=0)
    )
    return target, pos_scale, gap_scale


def phi(u, beta):
    return u*u + beta*u**4


def energy(combo, target, pos_scale, gap_scale, beta, n=N):
    position_terms = phi(
        (np.asarray(combo) - target) / pos_scale, beta
    )
    gap_terms = phi(
        (gaps(combo, n) - gaps(target, n)) / gap_scale, beta
    )
    return float(position_terms.sum() + gap_terms.sum())


def minimize(target, pos_scale, gap_scale, beta, n=N):
    # Tacan minimum nad svim rastucim kombinacijama.
    # Dinamicko programiranje obuhvata ceo prostor bez uzorkovanja.
    k = len(target)
    target_gaps = gaps(target, n)
    values = np.arange(n + 1, dtype=float)
    cost = np.full((k, n + 1), np.inf)
    parent = np.full((k, n + 1), -1, dtype=int)

    for v in range(1, n - k + 2):
        cost[0, v] = (
            phi((v - target[0]) / pos_scale[0], beta)
            + phi(
                (v - 1 - target_gaps[0]) / gap_scale[0], beta
            )
        )

    for j in range(1, k):
        for v in range(j + 1, n - k + j + 2):
            predecessors = np.arange(j, v)
            options = (
                cost[j - 1, predecessors]
                + phi(
                    (v - values[predecessors] - 1 - target_gaps[j])
                    / gap_scale[j],
                    beta,
                )
            )
            chosen = int(np.argmin(options))
            parent[j, v] = predecessors[chosen]
            cost[j, v] = (
                options[chosen]
                + phi((v - target[j]) / pos_scale[j], beta)
            )

    ends = np.arange(k, n + 1)
    total = cost[k - 1, ends] + phi(
        (n - ends - target_gaps[k]) / gap_scale[k], beta
    )
    end_index = int(np.argmin(total))
    v = int(ends[end_index])
    combo = [v]

    for j in range(k - 1, 0, -1):
        v = int(parent[j, v])
        combo.append(v)

    combo.reverse()
    return tuple(combo), float(total[end_index])


def hits(a, b):
    return len(set(map(int, a)) & set(map(int, b)))


def summarize(name, scores):
    histogram = np.bincount(scores, minlength=K + 1).tolist()
    print(
        f"{name}: prosek={np.mean(scores):.4f}; "
        f"pogodaka 0..7={histogram}"
    )


def self_test():
    target = np.array([1.8, 4.6, 7.1])
    ps = np.array([1., 1.4, 1.1])
    gs = np.array([1., 1.3, 1.2, 1.])

    for beta in (.25, 1.):
        found, score = minimize(target, ps, gs, beta, n=9)
        exhaustive = min(
            energy(c, target, ps, gs, beta, n=9)
            for c in itertools.combinations(range(1, 10), 3)
        )
        assert np.isclose(
            score, exhaustive, rtol=1e-12, atol=1e-12
        )
        assert np.isclose(
            score, energy(found, target, ps, gs, beta, n=9)
        )

    rows = np.array(list(itertools.islice(
        itertools.combinations(range(1, 40), 7), 180
    )), dtype=float)
    t = 140
    x = design(rows)

    changed = rows.copy()
    changed[t:] = np.array([2, 7, 12, 18, 24, 30, 39])

    original = predict_field(rows, x, t, 10.)
    altered = predict_field(changed, design(changed), t, 10.)

    for a, b in zip(original, altered):
        assert np.array_equal(a, b), (
            "Buduci redovi uticu na predikciju!"
        )

    print(
        "Self-test OK: globalni minimum "
        "i izolacija buducih redova."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default=DEFAULT)
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    rows = load(args.csv)
    if args.validation < 16 or args.test < 16:
        parser.error("Validation i test moraju imati bar 16 kola.")

    split = len(rows) - args.test
    validation_start = split - args.validation
    if validation_start < START + 128:
        parser.error("Premalo istorije za zadatu validaciju i test.")

    features = design(rows)
    configurations = tuple(itertools.product(
        (1., 10., 100.), (.25, 1.)
    ))
    scores = []

    # Parametri se biraju iskljucivo pre zavrsnog test-perioda.
    for alpha, beta in configurations:
        matches = []
        for t in range(validation_start, split):
            target, ps, gs = predict_field(
                rows, features, t, alpha
            )
            combo, _ = minimize(target, ps, gs, beta)
            matches.append(hits(combo, rows[t]))
        scores.append(float(np.mean(matches)))

    # Pri jednakom rezultatu ostaje prvi par iz deklarisane mreze.
    best = int(np.argmax(scores))
    alpha, beta = configurations[best]

    print(
        f"Ulaz: {Path(args.csv).resolve()}; "
        f"broj redova={len(rows)}"
    )
    print(
        f"Poslednji red: "
        f"{','.join(map(str, rows[-1].astype(int)))}"
    )
    print(
        f"Validacija: redovi {validation_start+1}..{split}; "
        f"alpha={alpha:g}; beta={beta:g}; "
        f"prosek={scores[best]:.4f}"
    )

    model_scores, previous_scores, median_scores = [], [], []

    for t in range(split, len(rows)):
        target, ps, gs = predict_field(
            rows, features, t, alpha
        )
        combo, _ = minimize(target, ps, gs, beta)

        model_scores.append(hits(combo, rows[t]))
        previous_scores.append(hits(rows[t-1], rows[t]))

        median = np.median(
            rows[max(0, t-1024):t], axis=0
        )
        median_combo, _ = minimize(
            median, np.ones(K), np.ones(K+1), .25
        )
        median_scores.append(hits(median_combo, rows[t]))

    print(
        f"Test: redovi {split+1}..{len(rows)}; "
        "svaki cilj otkriven tek posle predikcije."
    )
    summarize("Landau v1", model_scores)
    summarize("Prethodna kombinacija", previous_scores)
    summarize("Medijana rasporeda", median_scores)

    target, ps, gs = predict_field(
        rows, features, len(rows), alpha
    )
    combo, score = minimize(target, ps, gs, beta)

    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Potencijal V={score:.8f} "
        "(skor rangiranja, nije verovatnoca)."
    )


if __name__ == "__main__":
    main()



"""
Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv; broj redova=4698
Poslednji red: 4,12,18,28,35,36,37
Validacija: redovi 4587..4634; alpha=100; beta=0.25; prosek=1.2083
Test: redovi 4635..4698; svaki cilj otkriven tek posle predikcije.
Landau v1: prosek=1.2188; pogodaka 0..7=[15, 26, 17, 6, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Medijana rasporeda: prosek=1.1406; pogodaka 0..7=[16, 26, 19, 3, 0, 0, 0, 0]
SLEDECI RED 4699: 5,9,14,19,24,29,34
Potencijal V=0.12490930 (skor rangiranja, nije verovatnoca).





Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv; broj redova=2971
Poslednji red: 18,20,21,24,29,34,38
Validacija: redovi 2860..2907; alpha=100; beta=0.25; prosek=1.4375
Test: redovi 2908..2971; svaki cilj otkriven tek posle predikcije.
Landau v1: prosek=1.2188; pogodaka 0..7=[13, 27, 21, 3, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Medijana rasporeda: prosek=1.0938; pogodaka 0..7=[16, 31, 13, 3, 1, 0, 0, 0]
SLEDECI RED 2972: 5,10,15,20,24,30,35
Potencijal V=0.16613519 (skor rangiranja, nije verovatnoca).





Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv; broj redova=1727
Poslednji red: 4,12,18,28,35,36,37
Validacija: redovi 1616..1663; alpha=10; beta=0.25; prosek=1.3542
Test: redovi 1664..1727; svaki cilj otkriven tek posle predikcije.
Landau v1: prosek=1.3281; pogodaka 0..7=[11, 25, 25, 2, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Medijana rasporeda: prosek=1.3750; pogodaka 0..7=[10, 27, 21, 5, 1, 0, 0, 0]
SLEDECI RED 1728: 4,10,15,19,25,30,34
Potencijal V=0.15425578 (skor rangiranja, nije verovatnoca).
"""



"""
Teorija faznih prelaza
the theory of phase transitions


Mogući Landauov pravac bio bi opis cele konfiguracije pomoću nekoliko veličina koje čuvaju njenu strukturu. 
Za sedam poređanih brojeva, to bi mogli biti razmaci između njih, grupisanje i raspored praznina. 
Dve kombinacije sa potpuno različitim brojevima mogu imati sličnu strukturu. 
Landauova ideja parametra reda mogla bi poslužiti kao inspiracija za opis te strukture; primenu tek treba razviti.

Način da se distribucija mogućih sledećih kombinacija oblikuje tako da izdvoji najbolju predikciju celog skupa od sedam brojeva. 
Opis prošlih izvlačenja služi samo ako pomaže toj predikciji.
Za taj cilj bih iz Landauovog pristupa izdvojio efektivni potencijal: složen sistem opisati preko malog broja bitnih veličina i njihovih međusobnih veza. 
Moja analogija za loto bila bi da svakoj kombinaciji pripada neka vrednost, a najniža vrednost označava najbolju kandidaturu.
Presudno pitanje je šta određuje taj potencijal. 
Razmaci, grupisanje i odnosi među brojevima mogu opisati kombinaciju, ali sami po sebi ne daju predikciju. 
Treba pronaći vezu između prethodnih konfiguracija i naredne, koja opstaje i na kasnijim, unapred nepoznatim kolima. 
Bez toga bi samo lepše opisivali prošlost.


Model koristi raspored sedam brojeva i osam praznina između njih i granica 1-39. 
Iz prethodnih konfiguracija procenjuje naredni raspored, zatim nalazi globalni minimum potencijala sa kvadratnim i četvrtim stepenom odstupanja. 
Ovo je moja primena Landauove ideje.
Na poslednja 64 kola ostvaren je prosečno 1,2188 pogodaka, jednako poređenju sa prethodnom kombinacijom; poređenje sa medijanom rasporeda ostvarilo je 1,1406. Taj test nije pokazao prednost nad oba poređenja.
Provereni su globalni minimum. 


Zahteva NumPy; pokretanje:
"""
