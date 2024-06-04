from fullwave_simulation.constants import MaterialProperties
from fullwave_simulation.constants.constant import Constant


def build_instance():
    material_properties = MaterialProperties()
    return material_properties


def test_instance():
    material_properties = build_instance()
    assert isinstance(material_properties, MaterialProperties)
    assert isinstance(material_properties, Constant)


def test_propeties():
    material_properties = build_instance()
    for material_name in [
        "fat",
        "liver",
        "muscle",
        "water",
        "skin",
        "tissue",
        "connective",
        "blood",
    ]:
        test_data = material_properties.__getattribute__(material_name)
        assert isinstance(test_data, dict)
        assert "bovera" in test_data
        assert "alpha" in test_data
        assert "ppower" in test_data
        assert "c0" in test_data
        assert "rho0" in test_data
        assert "beta" in test_data

    assert hasattr(material_properties, "c0")
    assert hasattr(material_properties, "rho0")
    assert hasattr(material_properties, "a0")
    assert hasattr(material_properties, "beta0")
