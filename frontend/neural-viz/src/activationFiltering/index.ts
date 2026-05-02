import type { ActivationFilterAlgorithm } from "./types"

export const filterActivation = (data: number[][], algorithm: ActivationFilterAlgorithm): number[][] => {
    switch (algorithm.type) {
        case "Id":
            return data;
        case "ThresholdAlgorithm":
            return data.map(row => row.map(val => Math.abs(val) < algorithm.threshold ? 0 : val));
        default:
            return data;
    }
}