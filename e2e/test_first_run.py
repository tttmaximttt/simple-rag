"""The first-run flow in the browser: temporary password, forced change, setup wizard.

Needs a stack that nobody has logged in to yet, so it runs first (see conftest)."""
import httpx
import pytest
from playwright.sync_api import expect

from conftest import BASE_URL, PASSWORD, TEMP_PASSWORD, login, wizard_settings


def _flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flatten(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def test_first_run(page):
    with httpx.Client(base_url=BASE_URL) as c:
        if login(c, TEMP_PASSWORD).status_code != 200:
            pytest.skip("the temporary password no longer works; run e2e/run.sh for a fresh stack")
        c.post("/api/auth/logout")

    page.goto("/")
    expect(page.locator("#view-login")).to_be_visible()
    expect(page.locator("#nav")).to_be_hidden()

    page.fill("#login-form [name=password]", TEMP_PASSWORD)
    page.click("#login-form button")
    expect(page.locator("#view-password")).to_be_visible()
    expect(page.locator("#password-reason")).to_contain_text("temporary password")
    expect(page.locator("#nav")).to_be_hidden()

    # Mismatched repeat is caught in the browser.
    page.fill("[name=current_password]", TEMP_PASSWORD)
    page.fill("[name=new_password]", PASSWORD)
    page.fill("[name=repeat]", PASSWORD + "x")
    page.click("#password-form button")
    expect(page.locator("#password-result")).to_have_text("The new passwords don't match")

    page.fill("[name=repeat]", PASSWORD)
    page.click("#password-form button")
    # Not configured yet, so the app lands on Setup.
    expect(page.locator("#view-setup")).to_be_visible()
    expect(page).to_have_url(f"{BASE_URL}/#setup")
    expect(page.locator("#nav")).to_be_visible()

    fields = dict(_flatten(wizard_settings()))
    for name, value in fields.items():  # selects first: they decide which inputs are shown
        if name.endswith((".type", ".provider")):
            page.select_option(f"#setup-form [name='{name}']", value)
    for name, value in fields.items():
        if not name.endswith((".type", ".provider")):
            page.fill(f"#setup-form [name='{name}']", str(value))
    expect(page.locator("#token-hint")).to_contain_text("Tokenizer:")

    for component in ("vector_store", "embeddings", "llm"):
        page.click(f"[data-test={component}]")
        expect(page.locator(f"#test-{component}")).to_contain_text("✓", timeout=120_000)

    page.click("#setup-form button[type=submit]")
    expect(page.locator("#save-result")).to_have_text("Saved. You can upload documents now.")

    # The old password is gone and the new one works.
    with httpx.Client(base_url=BASE_URL) as c:
        assert login(c, TEMP_PASSWORD).status_code == 401
        assert login(c, PASSWORD).status_code == 200
        assert c.get("/api/health").json()["configured"] is True
        cfg = c.get("/api/config").json()
    assert cfg["vector_store"]["type"] == fields["vector_store.type"]
    assert cfg["embeddings"]["provider"] == fields["embeddings.provider"]
    # Secrets never come back to the browser.
    for name, value in fields.items():
        if name.endswith("api_key") and value:
            section, provider, _ = name.split(".")
            assert cfg[section][provider]["api_key"] == "********"
