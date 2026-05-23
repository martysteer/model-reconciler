"""Smoke tests: W3C manifest endpoint."""


def test_manifest_shape(app_client):
    """GET /reconcile/{slug} returns valid W3C manifest."""
    services = app_client.get("/").json()
    slug = services[0]["slug"]

    r = app_client.get(f"/reconcile/{slug}")
    assert r.status_code == 200

    m = r.json()
    assert "0.2" in m["versions"]
    assert m["name"]
    assert m["identifierSpace"]
    assert m["schemaSpace"]
    assert isinstance(m["defaultTypes"], list)
    assert len(m["defaultTypes"]) >= 1
    assert "id" in m["defaultTypes"][0]
    assert "name" in m["defaultTypes"][0]


def test_manifest_404_for_unknown_slug(app_client):
    r = app_client.get("/reconcile/nonexistent")
    assert r.status_code == 404
