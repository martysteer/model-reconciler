"""Smoke tests: health and service listing."""


def test_health_returns_200(app_client):
    r = app_client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["profiles_loaded"] >= 1
    assert isinstance(data["profiles"], list)


def test_root_lists_services(app_client):
    r = app_client.get("/")
    assert r.status_code == 200
    services = r.json()
    assert len(services) >= 1
    for s in services:
        assert "slug" in s
        assert "name" in s
        assert "url" in s
        assert s["url"].startswith("/reconcile/")
