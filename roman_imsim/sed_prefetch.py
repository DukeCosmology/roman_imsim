"""SkyCatalog SED prefetch orchestration for Roman image batches."""


def prefetch_seds(skycat, start, stop):
    """Prefetch runtime SEDs for a half-open range of image objects."""
    batch_key = (int(start), int(stop))
    if batch_key == skycat._prefetched_sed_batch:
        return
    if not 0 <= start <= stop <= len(skycat.objects):
        raise IndexError(
            f"Invalid SkyCatalog SED prefetch range [{start}, {stop}) " f"for {len(skycat.objects)} objects"
        )

    objects = skycat.objects[start:stop]
    diffsky_objects = [obj for obj in objects if obj.object_type == "diffsky26_galaxy"]
    if diffsky_objects and hasattr(diffsky_objects[0], "prefetch_seds"):
        if hasattr(diffsky_objects[0], "clear_prefetched_seds"):
            diffsky_objects[0].clear_prefetched_seds()
        skycat.logger.info(
            "Prefetching Diffsky SEDs for %d objects [%d, %d)",
            len(diffsky_objects),
            start,
            stop,
        )
        diffsky_objects[0].prefetch_seds(diffsky_objects)

    skycat._prefetched_sed_batch = batch_key
