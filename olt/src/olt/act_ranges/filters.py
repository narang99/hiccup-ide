class NoiseRatioRangeFilter:
    """
    Filters clusters by their noise-above ratio, and caps how many
    get plotted. Once `max_clusters` have been accepted, all further
    calls return False (no more plotting).
    """

    def __init__(self, min_d, max_d, max_clusters, kind="ratio"):
        self.min_d = min_d
        self.max_d = max_d
        self.max_clusters = max_clusters
        self.count = 0
        self.kind = kind

    def __call__(self, ratio, point_dist):
        d = ratio if self.kind == "ratio" else point_dist
        if self.count >= self.max_clusters:
            return False
        if not (self.min_d <= d <= self.max_d):
            return False
        self.count += 1
        return True

    @property
    def is_full(self):
        return self.count >= self.max_clusters
