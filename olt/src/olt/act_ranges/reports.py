from PIL import Image


def crop_top(image_path, max_height):
    """
    Crops an image to at most `max_height` pixels from the top.
    If the image is already shorter than max_height, it's left unchanged.
    """
    img = Image.open(image_path)
    width, height = img.size

    if height <= max_height:
        # Image is already smaller (or equal) — no cropping needed
        cropped = img
    else:
        # box = (left, upper, right, lower)
        cropped = img.crop((0, 0, width, max_height))

    return cropped


def get_cluster_photo(
    base_report_dir,
    layer_name,
    channel,
    cluster_label,
    kind="combined",
    crop_max_height=None,
):
    report_dir = base_report_dir / layer_name / str(channel)
    jpeg = report_dir / f"cluster_{cluster_label}_{kind}.jpeg"
    if not jpeg.exists():
        if not report_dir.exists():
            raise Exception(f"report dir {report_dir} does not exist")
        print(f"file {jpeg} does not exist, it might be a singleton cluster")
        return
    if crop_max_height is not None:
        image = crop_top(jpeg, crop_max_height)
    else:
        image = Image.open(jpeg)
    return image
