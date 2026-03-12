from biofeaturisers.config import HDXConfig, SAXSConfig

def test_hdx_config_defaults():
    config = HDXConfig()
    assert config.temperature == 293.15
    assert config.ph == 7.0

def test_saxs_config_defaults():
    config = SAXSConfig()
    assert config.q_min == 0.0
    assert config.q_max == 0.5
    assert config.num_q_points == 101
    assert config.vacuum_form_factor_table == "wk"
