import numpy as np
import tempfile
import os
from biofeaturisers.core.topology import MinimalTopology

def test_minimal_topology_dummy():
    topo = MinimalTopology.from_biotite_dummy(num_atoms=5)
    assert len(topo.atom_names) == 5
    assert topo.atom_names[0] == "CA"
    
def test_minimal_topology_json():
    topo = MinimalTopology.from_biotite_dummy(num_atoms=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "topo.json")
        topo.to_json(filepath)
        
        loaded_topo = MinimalTopology.from_json(filepath)
        np.testing.assert_array_equal(topo.atom_names, loaded_topo.atom_names)
        np.testing.assert_array_equal(topo.res_ids, loaded_topo.res_ids)
        np.testing.assert_array_equal(topo.chain_ids, loaded_topo.chain_ids)
