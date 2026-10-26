from __future__ import annotations

import pytest


class TestAuthAPI:
    def test_register_new_user(self, client):
        resp = client.post("/api/v1/auth/register", json={
            "username": "newuser",
            "email": "new@example.com",
            "password": "SecurePass123!",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["username"] == "newuser"
        assert "hashed_password" not in data

    def test_register_duplicate_username_returns_409(self, client):
        payload = {"username": "dupuser", "email": "dup@example.com", "password": "Pass123!"}
        client.post("/api/v1/auth/register", json=payload)
        resp = client.post("/api/v1/auth/register", json=payload)
        assert resp.status_code == 409

    def test_register_with_form_data(self, client):
        resp = client.post("/api/v1/auth/register", data={
            "username": "formuser",
            "email": "formuser@example.com",
            "password": "SecurePass123!",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["username"] == "formuser"

    def test_login_returns_token(self, client, test_user):
        resp = client.post("/api/v1/auth/token", data={
            "username": "testuser",
            "password": "TestPass123!",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_with_json_returns_token(self, client, test_user):
        resp = client.post("/api/v1/auth/token", json={
            "username": "testuser",
            "password": "TestPass123!",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password_returns_401(self, client, test_user):
        resp = client.post("/api/v1/auth/token", data={
            "username": "testuser",
            "password": "WrongPassword!",
        })
        assert resp.status_code == 401

    def test_get_me_with_valid_token(self, client, test_user_token):
        resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "testuser"

    def test_get_me_without_token_returns_401(self, client):
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 401

    def test_get_me_with_invalid_token_returns_401(self, client):
        resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalidtoken123"},
        )
        assert resp.status_code == 401


class TestHealthAPI:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_error_response_format(self, client):
        """Non-existent endpoint returns 404 with no stack trace."""
        resp = client.get("/api/v1/nonexistent")
        assert resp.status_code == 404
        # Ensure no Python stack trace in response
        assert "Traceback" not in resp.text
