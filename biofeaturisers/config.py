"""Configuration dataclasses for HDX and SAXS pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class HDXConfig:
    """Configuration values for HDX featurisation/forward behavior."""

    beta_c: float = 0.35
    beta_h: float = 2.0
    beta_0: float = 0.0
    cutoff_c: float = 6.5
    cutoff_h: float = 2.4
    steepness_c: float = 5.0
    steepness_h: float = 10.0
    seq_sep_min: int = 2
    intrachain_only: bool = False

    include_hetatm: bool = False
    disulfide_exchange: bool = False
    exchange_mask: list[str] = field(default_factory=list)

    use_hdxrate: bool = False
    hdxrate_pH: float = 7.0
    hdxrate_temp: float = 298.15
    timepoints: list[float] = field(default_factory=list)

    chunk_size: int = 0
    batch_size: int = 8

    def __post_init__(self) -> None:
        if self.cutoff_c <= 0 or self.cutoff_h <= 0:
            raise ValueError("cutoff_c and cutoff_h must be > 0")
        if self.steepness_c <= 0 or self.steepness_h <= 0:
            raise ValueError("steepness_c and steepness_h must be > 0")
        if self.seq_sep_min < 0:
            raise ValueError("seq_sep_min must be >= 0")
        if self.chunk_size < 0:
            raise ValueError("chunk_size must be >= 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.hdxrate_temp <= 0:
            raise ValueError("hdxrate_temp must be > 0")
        if any(tp < 0 for tp in self.timepoints):
            raise ValueError("timepoints must all be >= 0")


@dataclass(slots=True)
class SAXSConfig:
    """Configuration values for SAXS featurisation/forward behavior."""

    q_min: float = 0.01
    q_max: float = 0.50
    n_q: int = 300

    c1: float = 1.0
    c2: float = 0.0
    fit_c1_c2: bool = True
    c1_range: tuple[float, float] = (0.95, 1.12)
    c2_range: tuple[float, float] = (0.0, 4.0)
    c1_steps: int = 18
    c2_steps: int = 17
    rho0: float = 0.334

    ff_table: str = "waasmaier_kirfel"

    chunk_size: int = 512
    batch_size: int = 4

    include_chains: list[str] | None = None
    exclude_chains: list[str] | None = None
    include_hetatm: bool = False

    def __post_init__(self) -> None:
        if self.q_min <= 0:
            raise ValueError("q_min must be > 0")
        if self.q_max <= self.q_min:
            raise ValueError("q_max must be greater than q_min")
        if self.n_q <= 0:
            raise ValueError("n_q must be > 0")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.c1_steps <= 0 or self.c2_steps <= 0:
            raise ValueError("c1_steps and c2_steps must be > 0")
        if self.c1_range[0] >= self.c1_range[1]:
            raise ValueError("c1_range lower bound must be < upper bound")
        if self.c2_range[0] >= self.c2_range[1]:
            raise ValueError("c2_range lower bound must be < upper bound")
        if self.ff_table not in {"waasmaier_kirfel", "cromer_mann"}:
            raise ValueError("ff_table must be 'waasmaier_kirfel' or 'cromer_mann'")

