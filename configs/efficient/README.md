# Efficient AE-TFPE configs

Deliberately in a SUBDIRECTORY, not in `configs/`.

`scripts/check_shapes.py` and `scripts/print_run_matrix.py` glob `configs/*.yaml`
non-recursively. Keeping the Efficient arms here preserves the frozen-v1
invariant those tools report (**16 configs, 16 unique runs**) exactly as it was
before the Major Revision, so v1 provenance stays verifiable.

These arms belong to **Efficient AE-TFPE**, the improvement introduced during the
Major Revision. They are NOT part of the original submitted method and must never
be presented as if they were.
