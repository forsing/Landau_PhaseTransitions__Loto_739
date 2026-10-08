#!/usr/bin/env python3

"""LANDAU V1 Up: whole-configuration potential around an observed state."""

import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from scipy.spatial import cKDTree


DEFAULTS = (
    "/data/loto7_4698_k80.csv",
    "/data/loto7_4698_k80_loto_2971.csv",
    "/data/loto7_4698_k80_loto_plus_1727.csv",
)


N, K, D, WINDOW, START = 39, 7, 8, 128, 2
CONTEXT = 1 + 2*D
PAIRS = tuple(
    (i, j) for i in range(D) for j in range(i+1, D)
)
LINEAR = D*CONTEXT
WIDTH = LINEAR + 2*D + len(PAIRS)


def load(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        for line, row in enumerate(csv.reader(f), 1):
            if not row or all(not v.strip() for v in row):
                continue
            try:
                values = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem sedam celih brojeva."
                ) from error

            if (
                len(values) != K
                or values != sorted(values)
                or len(set(values)) != K
                or not all(1 <= v <= N for v in values)
            ):
                raise ValueError(
                    f"Red {line}: neispravna sortirana 7/39 kombinacija."
                )
            rows.append(values)

    return np.asarray(rows, dtype=int).reshape(-1, K)


def order_parameters(rows):
    rows = np.atleast_2d(rows)
    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), N+1),
    ))
    return np.diff(edges, axis=1)/(N+1)


def context(parameters, t):
    return np.r_[
        1.,
        parameters[t-1],
        parameters[t-2],
    ]


def features(parameters, ctx, reference):
    delta = np.atleast_2d(parameters) - reference

    field = (
        delta[:, :, None] * ctx[None, None, :]
    ).reshape(len(delta), -1)

    pairs = np.column_stack([
        delta[:, i] * delta[:, j]
        for i, j in PAIRS
    ])

    return np.column_stack((
        field,
        delta**2,
        delta**4,
        pairs,
    ))


def alternatives(row):
    empty = np.setdiff1d(np.arange(1, N+1), row)
    count = K*len(empty)

    candidates = np.repeat(
        np.asarray(row)[None, :], count, axis=0
    )
    candidates[
        np.arange(count),
        np.repeat(np.arange(K), len(empty)),
    ] = np.tile(empty, K)

    return np.sort(candidates, axis=1)


def prepare(rows, parameters, t):
    blocks = []

    for i in range(max(START, t-WINDOW), t):
        ctx = context(parameters, i)
        reference = parameters[i-1]

        positive = features(
            parameters[i], ctx, reference
        )
        negative = features(
            order_parameters(alternatives(rows[i])),
            ctx,
            reference,
        )
        blocks.append(negative-positive)

    design = np.vstack(blocks)

    # Numericko skaliranje kontrastnih osobina.
    scale = np.maximum(
        np.sqrt(
            np.einsum("ij,ij->j", design, design)
            / len(design)
        ),
        1e-8,
    )
    a = design/scale

    return (
        a.T @ a,
        a.sum(axis=0),
        scale,
        context(parameters, t),
        parameters[t-1],
    )


def coefficients(prepared, regularization):
    gram, rhs, scale, _, _ = prepared
    g = gram + regularization*np.eye(WIDTH)

    bounded = np.arange(
        LINEAR+D, LINEAR+2*D
    )
    free = np.setdiff1d(
        np.arange(WIDTH), bounded
    )

    lower = np.zeros(WIDTH)
    lower[bounded] = 1e-4*scale[bounded]
    shifted = rhs - g @ lower

    cross = g[np.ix_(free, bounded)]
    solved = np.linalg.solve(
        g[np.ix_(free, free)],
        np.column_stack((shifted[free], cross)),
    )
    u0 = solved[:, 0]
    inverse_cross = solved[:, 1:]

    schur = (
        g[np.ix_(bounded, bounded)]
        - cross.T @ inverse_cross
    )
    reduced = shifted[bounded] - cross.T @ u0

    factor = np.linalg.cholesky(
        (schur + schur.T)/2
    )
    z, _ = nnls(
        factor.T,
        np.linalg.solve(factor, reduced),
        maxiter=1000,
    )

    theta = lower.copy()
    theta[bounded] += z
    theta[free] = u0 - inverse_cross @ z

    return theta/scale


def fit(prepared, regularization):
    theta = coefficients(prepared, regularization)

    field = (
        theta[:LINEAR].reshape(D, CONTEXT)
        @ prepared[3]
    )
    quadratic = theta[LINEAR:LINEAR+D]
    quartic = theta[LINEAR+D:LINEAR+2*D]

    coupling = np.zeros((D, D))
    for value, (i, j) in zip(
        theta[LINEAR+2*D:], PAIRS
    ):
        coupling[i, j] = value
        coupling[j, i] = value

    return (
        field,
        quadratic,
        quartic,
        coupling,
        prepared[4],
    )


def potential(rows, model):
    field, quadratic, quartic, coupling, reference = model
    delta = order_parameters(rows) - reference

    return (
        delta @ field
        + (delta*delta) @ quadratic
        + (delta**4) @ quartic
        + .5*np.einsum(
            "bi,ij,bj->b",
            delta,
            coupling,
            delta,
            optimize=True,
        )
    )


def nearest_squared(parameters, prototypes):
    d = (
        (parameters*parameters).sum(axis=1)[:, None]
        + (prototypes*prototypes).sum(axis=1)[None, :]
        - 2*parameters @ prototypes.T
    )
    return np.maximum(d.min(axis=1), 0)


def support_radius(prototypes, quantile):
    if len(prototypes) < 2:
        return np.sqrt(2)/(N+1)

    d = (
        (prototypes*prototypes).sum(axis=1)[:, None]
        + (prototypes*prototypes).sum(axis=1)[None, :]
        - 2*prototypes @ prototypes.T
    )
    np.fill_diagonal(d, np.inf)

    nearest = np.sqrt(
        np.maximum(d.min(axis=1), 0)
    )
    return max(
        float(np.quantile(nearest, quantile)),
        np.sqrt(2)/(N+1),
    )


def search(history, model, quantile=.75):
    current = np.unique(history[-WINDOW:], axis=0)
    prototypes = order_parameters(current)
    radius = support_radius(prototypes, quantile)
    tree = cKDTree(prototypes)
    values = np.arange(1, N+1)

    for iteration in range(1, 101):
        available = np.all(
            values[None, :, None]
            != current[:, None, :],
            axis=2,
        )
        who, which = np.nonzero(available)

        blocks = [current]
        owners = [np.arange(len(current))]

        for j in range(K):
            neighbors = current[who].copy()
            neighbors[:, j] = values[which]
            blocks.append(np.sort(neighbors, axis=1))
            owners.append(who)

        candidates = np.vstack(blocks)
        owners = np.concatenate(owners)
        scores = potential(candidates, model)

        distance = tree.query(
            order_parameters(candidates),
            k=1,
            workers=1,
        )[0]
        unsupported = distance**2 > radius*radius + 1e-12
        scores[unsupported] = np.inf

        minimum = np.full(len(current), np.inf)
        np.minimum.at(minimum, owners, scores)

        eligible = scores <= minimum[owners] + 1e-12
        indices = np.arange(len(candidates))
        chosen = np.full(
            len(current), len(candidates), dtype=int
        )
        np.minimum.at(
            chosen,
            owners,
            np.where(eligible, indices, len(candidates)),
        )

        updated = candidates[chosen]
        if np.array_equal(updated, current):
            scores = potential(current, model)
            best = int(np.argmin(scores))
            return (
                tuple(map(int, current[best])),
                float(scores[best]),
                iteration,
            )

        current = updated

    raise RuntimeError(
        "Pretraga nije dostigla optimum zamena u 100 koraka."
    )


def hits(prediction, actual):
    return len(
        set(map(int, prediction))
        & set(map(int, actual))
    )


def summarize(label, scores):
    print(
        f"{label}: prosek={np.mean(scores):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(scores, minlength=8).tolist()}",
        flush=True,
    )


def self_test():
    rows = np.tile(np.array([
        [1, 2, 3, 4, 5, 6, 7],
        [4, 12, 18, 28, 35, 36, 37],
        [33, 34, 35, 36, 37, 38, 39],
        [2, 7, 12, 18, 24, 30, 39],
    ]), (50, 1))

    m = order_parameters(rows)
    assert np.allclose(m.sum(axis=1), 1)
    assert np.allclose(
        np.cumsum(m[:, :-1], axis=1)*(N+1),
        rows,
    )

    zero = (
        np.zeros(D),
        np.zeros(D),
        np.zeros(D),
        np.zeros((D, D)),
        m[1],
    )
    examples = [
        [5, 10, 15, 20, 25, 30, 35],
        [1, 2, 3, 4, 5, 6, 7],
        [4, 12, 18, 28, 35, 36, 37],
    ]
    assert np.array_equal(
        potential(examples, zero),
        np.zeros(len(examples)),
    )

    anchor = (
        np.zeros(D),
        np.zeros(D),
        np.ones(D),
        np.zeros((D, D)),
        m[1],
    )
    assert np.isclose(
        potential([rows[1]], anchor)[0], 0
    )
    assert potential([examples[0]], anchor)[0] > 0

    prepared = prepare(rows, m, 170)
    model = fit(prepared, 10.)

    changed = rows.copy()
    changed[170:] = [3, 8, 13, 19, 25, 31, 38]
    altered = fit(
        prepare(
            changed,
            order_parameters(changed),
            170,
        ),
        10.,
    )
    for a, b in zip(model, altered):
        assert np.array_equal(a, b)

    theta = coefficients(prepared, 10.)
    assert np.allclose(
        features(m, prepared[3], prepared[4]) @ theta,
        potential(rows, model),
    )
    assert np.all(model[2] >= 1e-4 - 1e-12)

    result = search(rows[:170], model)
    assert result == search(rows[:170], model)

    combo, score, _ = result
    assert len(set(combo)) == K
    assert list(combo) == sorted(combo)

    prototypes = order_parameters(
        np.unique(rows[:170][-WINDOW:], axis=0)
    )
    radius = support_radius(prototypes, .75)
    neighbors = alternatives(combo)
    valid = (
        nearest_squared(
            order_parameters(neighbors),
            prototypes,
        )
        <= radius*radius + 1e-12
    )
    assert (
        potential(neighbors[valid], model).min()
        >= score - 1e-12
    )

    print(
        "Self-test OK: cela konfiguracija, stvarna referenca, "
        "hronologija i podrucje."
    )


def run_file(path, validation_count=48, test_count=64):
    rows = load(path)

    if validation_count < 16 or test_count < 16:
        raise ValueError(
            "Validation i test moraju imati bar 16 kola."
        )

    split = len(rows) - test_count
    begin = split - validation_count
    if begin < START + 128:
        raise ValueError(
            f"{path}: premalo istorije za zadatu validaciju i test."
        )

    m = order_parameters(rows)
    settings = tuple(
        (reg, q)
        for reg in (1., 10., 100.)
        for q in (.5, .75, .9)
    )
    validation = [[] for _ in settings]

    print(
        f"Ulaz: {Path(path).resolve()}; "
        f"redova={len(rows)}",
        flush=True,
    )

    for t in range(begin, split):
        prepared = prepare(rows, m, t)
        models = {
            reg: fit(prepared, reg)
            for reg in (1., 10., 100.)
        }

        for j, (reg, quantile) in enumerate(settings):
            combo, _, _ = search(
                rows[:t],
                models[reg],
                quantile,
            )
            validation[j].append(hits(combo, rows[t]))

        if (t-begin+1) % 16 == 0:
            print(
                f"Validacija: {t-begin+1}/{validation_count}",
                flush=True,
            )

    means = [
        np.mean(values) for values in validation
    ]
    best = int(np.argmax(means))
    regularization, quantile = settings[best]

    print(
        f"Validacija: redovi {begin+1}..{split}; "
        f"regularizacija={regularization:g}; "
        f"kvantil_podrske={quantile:g}; "
        f"prosek={means[best]:.4f}",
        flush=True,
    )

    model_scores = []
    previous_scores = []

    for t in range(split, len(rows)):
        model = fit(
            prepare(rows, m, t),
            regularization,
        )
        combo, _, _ = search(
            rows[:t],
            model,
            quantile,
        )

        model_scores.append(hits(combo, rows[t]))
        previous_scores.append(hits(rows[t-1], rows[t]))

        if (t-split+1) % 16 == 0:
            print(
                f"Test: {t-split+1}/{test_count}",
                flush=True,
            )

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize("LANDAU V1", model_scores)
    summarize("Prethodna kombinacija", previous_scores)

    model = fit(
        prepare(rows, m, len(rows)),
        regularization,
    )
    combo, score, iterations = search(
        rows,
        model,
        quantile,
    )

    print(
        f"Poslednji red: {','.join(map(str, rows[-1]))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(f"V={score:.10f}; iteracija={iterations}")
    print(
        "Potencijal rangira cele kombinacije u empirijskom podrucju. "
        "Globalni minimum nije potvrden."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="*",
        help="Bez putanja proverava sva tri DEFAULTS fajla.",
    )
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    for path in args.csv or DEFAULTS:
        run_file(path, args.validation, args.test)


if __name__ == "__main__":
    main()



"""
Ulaz: /data/loto7_4698_k80.csv; redova=4698
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 4587..4634; regularizacija=100; kvantil_podrske=0.9; prosek=1.5833
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
LANDAU V1: prosek=1.3594; pogodaka 0..7=[14, 23, 18, 8, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 1,x,22,y,24,z,38
V=-1.8913879949; iteracija=17
Potencijal rangira cele kombinacije u empirijskom podrucju. Globalni minimum nije potvrden.



Ulaz: /data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 2860..2907; regularizacija=1; kvantil_podrske=0.5; prosek=1.4583
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
LANDAU V1: prosek=1.4375; pogodaka 0..7=[10, 28, 16, 8, 2, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 1,x,25,y,32,z,39
V=-3.2802889897; iteracija=14
Potencijal rangira cele kombinacije u empirijskom podrucju. Globalni minimum nije potvrden.



Ulaz: /data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 1616..1663; regularizacija=1; kvantil_podrske=0.5; prosek=1.4792
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
LANDAU V1: prosek=1.3438; pogodaka 0..7=[13, 24, 19, 8, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 1,x,20,y,25,z,38
V=-0.8654296590; iteracija=19
Potencijal rangira cele kombinacije u empirijskom podrucju. Globalni minimum nije potvrden.
"""




"""
U V1_Up zamenio sam predviđanje prosečnih položaja direktnim rangiranjem celih kombinacija, sa koeficijentima potencijala naučenim iz podataka. 
Nema izlaznog proseka po sortiranim pozicijama. Potencijal ocenjuje celu konfiguraciju, a koeficijenti kvadratnih, četvrtih i međusobnih članova uče se bez zadatog znaka. 
Dodao sam ograničenje lokalne važenosti: kandidat mora da bude blizu neke stvarne istorijske konfiguracije, a širina tog područja određuje se iz podataka. 

Sada direktno rangira cele kombinacije. 
Referenca je prethodna stvarna konfiguracija; uklonjeno je predviđanje prosečnih položaja brojeva. 
Potencijal se koristi u području pokrivenom stvarnim istorijskim konfiguracijama.

Fajl	    Sledeći red	   Predlog	            Prosek na 64 test-kola
Objedinjeni	4699	       1,x,22,y,24,z,38	    1,3594
Loto	    2972	       1,x,25,y,32,z,39	    1,4375
Loto Plus	1728	       1,x,20,y,25,z,38	    1,3438

Provereni su izolacija budućih redova, ponovljivost i ograničenja pretrage. 
Globalni optimum nije dokazan.
"""



"""
Up verzije sam uradio radi:
Normalnu (Gaussovu) raspodelu nisam eksplicitno ugradio ni u jedan Landau kod. 
Ali u više modela jesam ugradio usrednjavanje i regularizaciju koji mogu da povuku predikciju ka ravnomerno raspoređenim brojevima. 

Kod     Šta je ugrađeno u osnovne verzije
V1	    Regresija položaja i potencijal oko jednog predviđenog rasporeda — sklonost centralnom rasporedu.
V2a     Quantum	Regresija matrice gustine i vremenski proseci — mogu zagladiti različite strukture.
V2b     Levels	Regresija težina nivoa — može usredniti odvojena strukturna stanja.
V2c     Diamagnetism	Regresija promena rasporeda — može proizvesti vraćanje prema sredini.
V3      Superfluidity	Izričita prosečna pozadina prethodnih 32 kola, uz predviđene pobude.
V4      Damping	Konstantna pozadina uz regularizovane talase — slabi talasni doprinos ostavlja centralni raspored.
V5      Fermi-liquid	Direktno rangiranje celih kombinacija energetskim funkcionalom; nema izlaznog proseka sedam položaja.
V6      Ginzburg-Landau	Direktno rangiranje celih kombinacija; pozitivan četvrti stepen može favorizovati ujednačene razmake.

Dakle, odsustvo Gaussove formule nije značilo da nema normalne raspodele. 
Posebno V1, V2c, V3 i V4 imaju konstrukciju koja može dati upravo centralno zaglađivanje.
"""
