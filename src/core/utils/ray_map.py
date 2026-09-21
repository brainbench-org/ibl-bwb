import ray

_EXHAUSTED = object()


def imap_unordered(submit, items, *, max_in_flight: int | None = None):
    """Yield each ray task's result as it completes, one value per item.

    Loading stays sequential on the driver while computation runs in parallel, with at
    most ``max_in_flight`` tasks queued in the object store (``None`` submits everything
    up front). Yields ``None`` for an item ``submit`` declined and for a task that
    returned ``None``, so a caller can advance its progress bar either way.

    Args:
        submit: ``submit(item)``, returns an ``ObjectRef`` or None to skip the item.
        items: iterable of work items, consumed lazily.
        max_in_flight: cap on queued tasks, or None for no cap.
    """
    pending = iter(items)
    futures = []
    exhausted = False

    while True:
        while not exhausted and (max_in_flight is None or len(futures) < max_in_flight):
            item = next(pending, _EXHAUSTED)
            if item is _EXHAUSTED:
                exhausted = True
                break
            future = submit(item)
            if future is None:
                yield None
            else:
                futures.append(future)

        if not futures:
            if exhausted:
                return
            continue

        done, futures = ray.wait(futures, num_returns=1)
        yield ray.get(done[0])
