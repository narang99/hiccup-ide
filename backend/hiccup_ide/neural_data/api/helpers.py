def get_min_max(data_list):
    """Helper to find min and max in a list of nested data (JSON)."""
    # keep going down until we find a list of elements
    # then simply run min/max on them and return them

    def _find_min_max(items):
        if isinstance(items, list):
            if not items:
                return 0.0, 0.0
            elif isinstance(items[0], (int, float)):
                return min(items), max(items)
            else:
                main_min, main_max = None, None
                for item in items:
                    cur_min, cur_max = _find_min_max(item)
                    if main_min is None or cur_min < main_min:
                        main_min = cur_min
                    if main_max is None or cur_max > main_max:
                        main_max = cur_max
                return main_min, main_max
        elif isinstance(items, (int, float)):
            return items, items
        else:
            raise Exception(
                f"only lists or floats are allowed for finding min-max, got={type(items)} items={items}"
            )

    return _find_min_max(data_list)


def apply_algorithm(data, algorithm_dict):
    alg_type = algorithm_dict.get("type")
    if alg_type == "Id":
        return data
    elif alg_type == "ThresholdAlgorithm":
        threshold = algorithm_dict.get("threshold", 0)
        # Handle list of lists (2D array)
        if isinstance(data, list):
            return [
                [val if abs(val) >= threshold else 0 for val in row] for row in data
            ]
        # Handle single value
        elif isinstance(data, (int, float)):
            return data if abs(data) >= threshold else 0
    return data
