from dataclasses import dataclass

@dataclass
class HDXConfig:
    temperature: float = 293.15  # Kelvin
    ph: float = 7.0
    isotope_abundance: float = 1.0  # D2O percentage
    sequence: str = ""

@dataclass
class SAXSConfig:
    q_min: float = 0.0
    q_max: float = 0.5
    num_q_points: int = 101
    c1_init: float = 1.0
    c2_init: float = 0.0
    vacuum_form_factor_table: str = "wk" # options: "wk", "cromer_mann"
