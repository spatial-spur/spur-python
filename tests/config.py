from pathlib import Path

ATOL = 1e-8
RTOL = 0.0
PARITY_ATOL = 1e-5
PARITY_RTOL = 0.0

# Frozen synthetic verification data, also used for the SCPC parity fixtures.
BENCH_200 = Path(__file__).parent / "fixtures" / "rust_core" / "bench_200.csv"
