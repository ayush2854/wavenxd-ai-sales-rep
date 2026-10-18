"""Run from the backend folder:  python -m pytest -v"""
from app.recommender import recommend


def test_nothing_provided_asks_for_more_info():
    result = recommend()
    assert result["status"] == "needs_more_info"
    assert set(result["missing_info"]) == {"application", "viscosity_cp", "flow_ml_min", "droplet_size_um"}


def test_stent_coating_gets_180khz_torch():
    result = recommend("stent coating", viscosity_cp=10, flow_ml_min=3, droplet_size_um=10)
    assert result["status"] == "match"
    assert result["lead"] == "nozzle"
    assert result["best"]["id"] == "std_torch_180khz"
    # the micro dispensing system also lists stents, so it is offered as an option
    assert "micro_dispensing" in [s["id"] for s in result["systems_to_consider"]]


def test_fuel_cell_small_droplets_gets_60khz_high_viscosity():
    result = recommend("fuel cell coating", viscosity_cp=500, flow_ml_min=2, droplet_size_um=40)
    assert result["best"]["id"] == "hv_conical_60khz"


def test_fuel_cell_larger_droplets_gets_24khz_high_viscosity():
    result = recommend("fuel cell coating", viscosity_cp=500, flow_ml_min=2, droplet_size_um=60)
    assert result["best"]["id"] == "hv_conical_24khz"


def test_industrial_spray_dryer_gets_high_flow_nozzle():
    result = recommend("industrial spray dryer", viscosity_cp=20, flow_ml_min=150)
    assert result["best"]["id"] == "high_flow_conical_60khz"


def test_lab_spray_dryer_follows_droplet_size():
    small = recommend("lab spray dryer", viscosity_cp=10, flow_ml_min=2, droplet_size_um=25)
    large = recommend("lab spray dryer", viscosity_cp=10, flow_ml_min=2, droplet_size_um=40)
    assert small["best"]["id"] == "std_120khz"
    assert large["best"]["id"] == "std_60khz"


def test_pipac_chemotherapy_gets_pipac_nozzle():
    result = recommend("PIPAC chemotherapy", viscosity_cp=40, flow_ml_min=15)
    assert result["best"]["id"] == "pipac_conical_60khz"


def test_syringe_coating_high_viscosity_gets_extended_horn_torch():
    result = recommend("syringe coating", viscosity_cp=800, flow_ml_min=3)
    assert result["best"]["id"] == "ext_horn_25mm_torch_60khz"
    assert "micro_dispensing" in [s["id"] for s in result["systems_to_consider"]]


def test_blood_collection_tube_gets_extended_horn_conical():
    result = recommend("blood collection tubes", viscosity_cp=5, flow_ml_min=2)
    assert result["best"]["id"] == "ext_horn_15mm_conical_60khz"


def test_transparent_conductive_coating_leads_with_xy_system():
    result = recommend("transparent conductive coating", viscosity_cp=20, flow_ml_min=2)
    assert result["status"] == "match"
    assert result["lead"] == "system"
    assert result["systems_to_consider"][0]["id"] == "xy_spray_pyrolysis"


def test_viscosity_above_catalog_range_is_no_match():
    result = recommend("coating", viscosity_cp=5000)
    assert result["status"] == "no_match"
    assert any("1000" in r for r in result["reasons"])


def test_flow_above_catalog_range_is_no_match():
    result = recommend("spray drying", flow_ml_min=500)
    assert result["status"] == "no_match"
    assert any("200" in r for r in result["reasons"])


def test_high_viscosity_and_high_flow_together_is_no_match():
    result = recommend("coating", viscosity_cp=500, flow_ml_min=10)
    assert result["status"] == "no_match"
    assert "follow-up" in result["next_step"]


def test_unknown_application_warns_and_defers_to_team():
    result = recommend("food processing", viscosity_cp=10, flow_ml_min=2)
    assert result["status"] == "match"
    assert any("WaveNxD team" in w for w in result["warnings"])


def test_viscosity_exactly_at_limit_is_flagged_borderline():
    result = recommend("lab spray dryer", viscosity_cp=50, flow_ml_min=2)
    assert result["status"] == "match"
    assert any("exactly at" in w for w in result["best"]["warnings"])


def test_partial_information_still_recommends_and_lists_what_is_missing():
    result = recommend(viscosity_cp=10)
    assert result["status"] == "match"
    assert "application" in result["missing_info"]
    assert "flow_ml_min" in result["missing_info"]


def test_same_input_gives_same_output():
    a = recommend("fuel cell coating", viscosity_cp=300, flow_ml_min=1)
    b = recommend("fuel cell coating", viscosity_cp=300, flow_ml_min=1)
    assert a == b