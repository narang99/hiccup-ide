def merge_pw_samples_into(target, update):
    """
    Merges one call's pw_samples (NeuronParentAnalyser.collect_cluster_stats_df's
    second return value) into an accumulator built up across many images:
    extends each cid's sample list rather than overwriting it, the same way
    layer_by_channel_by_label_by_points/"ACTS_DICT" accumulates each cluster's
    activation population across the whole dataset rather than replacing it
    per image. Mutates and returns target, so callers can fold each image's
    pw_samples in as they go:

        pw_samples = {}
        for ...:
            _, pw_by_dep = analyser.collect_cluster_stats_df(..., image_key=image_key)
            merge_pw_samples_into(pw_samples, pw_by_dep)
    """
    for dep_layer_name, channels in update.items():
        target_channels = target.setdefault(dep_layer_name, {})
        for dep_channel, cids in channels.items():
            target_cids = target_channels.setdefault(dep_channel, {})
            for cid, samples in cids.items():
                target_cids.setdefault(cid, []).extend(samples)
    return target
