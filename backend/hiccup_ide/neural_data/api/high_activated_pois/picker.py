import random
from neural_data.api.high_activated_pois.common import FlattenedContrib


def get_highest_contribs(
    contribs: list[FlattenedContrib], k: int
) -> list[FlattenedContrib]:
    "sorts contribs in descending order and gives the highest ones"
    contribs = sorted(contribs, key=lambda x: x[2], reverse=True)
    return contribs[:k]


def get_stratified_contribs(
    contribs: list[FlattenedContrib], k: int, num_strata: int = 5
) -> list[FlattenedContrib]:
    "returns k/2 highest contribs + stratified sampling for remaining"
    if k <= 0:
        return []

    sorted_contribs: list[FlattenedContrib] = sorted(
        contribs, key=lambda x: x[2], reverse=True
    )

    k_half = k // 2
    highest_contribs = sorted_contribs[:k_half]

    remaining_contribs = sorted_contribs[k_half:]
    remaining_k = k - k_half

    if remaining_k <= 0 or not remaining_contribs:
        return highest_contribs

    stratified_contribs = _perform_stratified_sampling(
        remaining_contribs, remaining_k, num_strata
    )
    return highest_contribs + stratified_contribs


def _perform_stratified_sampling(
    remaining_contribs: list[FlattenedContrib], remaining_k: int, num_strata: int = 5
) -> list[FlattenedContrib]:
    """Perform stratified sampling on remaining contributions"""
    if len(remaining_contribs) <= remaining_k:
        return remaining_contribs

    min_val = remaining_contribs[-1][2]
    max_val = remaining_contribs[0][2]

    if max_val == min_val:
        return random.sample(remaining_contribs, remaining_k)

    num_strata = min(num_strata, remaining_k)
    strata = _create_strata(remaining_contribs, num_strata)
    return _sample_from_strata(strata, remaining_k)


def _create_strata(
    remaining_contribs: list[FlattenedContrib], num_strata: int
) -> list[list[FlattenedContrib]]:
    """Group contributions into strata based on value ranges"""
    min_val = remaining_contribs[-1][2]
    max_val = remaining_contribs[0][2]
    strata_size = (max_val - min_val) / num_strata

    strata = [[] for _ in range(num_strata)]
    for contrib in remaining_contribs:
        value = contrib[2]
        stratum_idx = min(int((value - min_val) / strata_size), num_strata - 1)
        strata[stratum_idx].append(contrib)

    return strata


def _sample_from_strata(
    strata: list[list[FlattenedContrib]], remaining_k: int
) -> list[FlattenedContrib]:
    """Sample proportionally from each stratum"""
    num_strata = len(strata)
    samples_per_stratum = remaining_k // num_strata
    extra_samples = remaining_k % num_strata

    stratified_contribs = []
    for i, stratum in enumerate(strata):
        if not stratum:
            continue

        samples_needed = samples_per_stratum
        if i < extra_samples:
            samples_needed += 1

        samples_to_take = min(samples_needed, len(stratum))
        if samples_to_take > 0:
            stratified_contribs.extend(random.sample(stratum, samples_to_take))

    return stratified_contribs
