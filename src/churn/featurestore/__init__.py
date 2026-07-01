"""Feature store: offline (DuckDB/Parquet + ASOF PIT joins) + online (Redis).

Phase 1 builds the offline PIT half; Phase 2 adds the online half and the
online/offline parity test. Empty at Phase 0.
"""
