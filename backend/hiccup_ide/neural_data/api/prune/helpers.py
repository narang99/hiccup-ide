def apply_algorithm(data, algorithm_dict):
    alg_type = algorithm_dict.get("type")
    if alg_type == "Id":
        return data
    elif alg_type == "ThresholdAlgorithm":
        threshold = algorithm_dict.get("threshold", 0)
        # Handle list of lists (2D array)
        if isinstance(data, list):
            return [
                [val if val >= threshold else 0 for val in row] for row in data
            ]
        # Handle single value
        elif isinstance(data, (int, float)):
            return data if abs(data) >= threshold else 0
    return data
