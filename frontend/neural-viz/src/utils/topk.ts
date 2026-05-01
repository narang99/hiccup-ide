/**
 * Given a flat array of values and a threshold ratio (0–1), returns a boolean
 * mask (same length as `values`) where `true` means the element is one of the
 * top-K positive values whose cumulative sum first exceeds
 * `threshold_ratio * total_positive_sum`.
 *
 * Elements that are not positive, or that fall outside the top-K, are `false`.
 */
export function getTopKThreshold(values: number[], thresholdRatio: number): number {
  // 1. Collect positive values with their original indices
  const positives: { value: number; index: number }[] = [];
  for (let i = 0; i < values.length; i++) {
    if (values[i] > 0) {
      positives.push({ value: values[i], index: i });
    }
  }

  if (positives.length === 0) {
    return 0;
  }

  // 2. Total positive sum and threshold
  const totalPosSum = positives.reduce((acc, p) => acc + p.value, 0);
  const threshold = thresholdRatio * totalPosSum;

  // 3. Sort descending by value
  positives.sort((a, b) => b.value - a.value);

  // 4. Walk down until cumulative sum exceeds threshold
  let cumSum = 0;
  let valueThreshold = 0;
  for (const p of positives) {
    cumSum += p.value;
    if (cumSum >= threshold)  {
      // stuff needs to be above this value
      valueThreshold = p.value;
      break;
    }
  }

  return valueThreshold;
}

/**
 * Stats returned alongside the mask, useful for display.
 */
// export interface TopKStats {
//   k: number;           // number of selected pixels
//   totalPositive: number; // total count of positive pixels
//   coveredRatio: number;  // actual ratio covered (may be slightly > thresholdRatio)
// }

// export function getTopKMaskWithStats(
//   values: number[],
//   thresholdRatio: number
// ): { mask: boolean[]; stats: TopKStats } {
//   const mask = new Array<boolean>(values.length).fill(false);

//   const positives: { value: number; index: number }[] = [];
//   for (let i = 0; i < values.length; i++) {
//     if (values[i] > 0) positives.push({ value: values[i], index: i });
//   }

//   if (positives.length === 0) {
//     return { mask, stats: { k: 0, totalPositive: 0, coveredRatio: 0 } };
//   }

//   const totalPosSum = positives.reduce((acc, p) => acc + p.value, 0);
//   const threshold = thresholdRatio * totalPosSum;

//   positives.sort((a, b) => b.value - a.value);

//   let cumSum = 0;
//   let k = 0;
//   for (const p of positives) {
//     cumSum += p.value;
//     mask[p.index] = true;
//     k++;
//     if (cumSum >= threshold) break;
//   }

//   return {
//     mask,
//     stats: {
//       k,
//       totalPositive: positives.length,
//       coveredRatio: totalPosSum > 0 ? cumSum / totalPosSum : 0,
//     },
//   };
// }
