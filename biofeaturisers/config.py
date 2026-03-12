from dataclasses import dataclass, field

@dataclass
class HDXConfig:
    temperature: float = 293.15  # Kelvin
    ph: float = 7.0
    isotope_abundance: float = 1.0  # D2O percentage
    sequence: str = ""

    # Best-Vendruscolo / Wan model parameters
    beta_c: float = 0.35
    beta_h: float = 2.0
    beta_0: float = 0.0
    cutoff_c: float = 6.5
    cutoff_h: float = 2.4
    steepness: float = 10.0
    seq_sep_min: int = 2
    intrachain_only: bool = False

    # Topology/feature options
    include_hetatm: bool = False
    disulfide_exchange: bool = False

    # HDXrate integration
    use_hdxrate: bool = False
    hdxrate_pH: float = 7.0
    hdxrate_temp: float = 298.15
    timepoints: tuple[float, ...] = field(default_factory=tuple)

    # Compute
    chunk_size: int = 0
    batch_size: int = 8

@dataclass
class SAXSConfig:
    q_min: float = 0.0
    q_max: float = 0.5
    num_q_points: int = 101
    c1_init: float = 1.0
    c2_init: float = 0.0
    vacuum_form_factor_table: str = "wk" # options: "wk", "cromer_mann"
