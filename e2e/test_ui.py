"""The web UI in a real browser, after first-run setup."""
import re

from playwright.sync_api import expect

from conftest import ANSWER, PASSWORD, QUESTION, RUN
from pdfs import make_pdf


def test_ui_needs_login(page, admin):
    page.goto("/#admin")
    expect(page.locator("#view-login")).to_be_visible()
    expect(page.locator("#view-admin")).to_be_hidden()
    page.fill("#login-form [name=password]", "wrong-password")
    page.click("#login-form button")
    expect(page.locator("#login-error")).to_have_text("Wrong username or password")
    page.fill("#login-form [name=password]", PASSWORD)
    page.click("#login-form button")
    expect(page.locator("#view-admin")).to_be_visible()


def test_upload_files_and_folder(ui, tmp_path):
    expect(ui.locator("#view-documents")).to_be_visible()
    (tmp_path / "puffin.pdf").write_bytes(make_pdf([f"Puffins {RUN} nest on the Skellig cliffs every May."]))
    (tmp_path / "notes.txt").write_text("not a pdf")
    with ui.expect_file_chooser() as fc:
        ui.click("#pick-files")
    fc.value.set_files([tmp_path / "puffin.pdf"])
    row = ui.locator("#doc-rows tr", has_text="puffin.pdf")
    expect(row).to_contain_text("ready", timeout=120_000)
    expect(row.locator("td").nth(2)).to_have_text("1")  # pages
    expect(ui.locator("#upload-summary")).to_contain_text("1 queued for ingestion")

    # The same file again is reported as a duplicate.
    with ui.expect_file_chooser() as fc:
        ui.click("#pick-files")
    fc.value.set_files([tmp_path / "puffin.pdf"])
    expect(ui.locator("#upload-list")).to_contain_text("duplicate of puffin.pdf")


def test_chat_answers_with_sources(ui, lighthouse):
    ui.click("nav a[href='#chat']")
    ui.fill("#question", QUESTION)
    ui.click("#chat-form button")
    reply = ui.locator(".msg.bot").last
    expect(reply).to_contain_text(ANSWER, timeout=120_000)
    reply.locator("summary").click()
    expect(reply.locator(".hit").first).to_contain_text("lighthouse.pdf, page 1")
    expect(ui.locator(".msg.user").last).to_have_text(QUESTION)


def test_admin_page(ui, lighthouse):
    ui.click("nav a[href='#admin']")
    expect(ui.locator("#cards")).to_contain_text("vectors stored")
    expect(ui.locator("#component-rows")).to_contain_text("vector store")
    expect(ui.locator("#component-rows td.ok")).to_have_count(2, timeout=30_000)

    ui.click("#check-llm")
    expect(ui.locator("#llm-result")).to_contain_text("✓", timeout=120_000)

    ui.fill("#debug-q", QUESTION)
    ui.fill("#debug-k", "1")
    ui.click("#debug-form button")
    expect(ui.locator("#debug-hits")).to_contain_text("1 hits")
    expect(ui.locator("#debug-hits .hit")).to_contain_text("lighthouse.pdf, page 1")

    expect(ui.locator("#events")).to_contain_text("[ingest]")


def test_connect_page(ui):
    ui.click("nav a[href='#connect']")
    expect(ui.locator("#connect-urls tr", has_text="MCP server")).to_contain_text(re.compile(r"https?://.+/mcp"))
    expect(ui.locator("#connect-clients summary")).to_have_count(9)


def test_create_and_revoke_api_key(ui):
    ui.click("nav a[href='#admin']")
    name = f"ui-key-{RUN}"
    ui.fill("#key-name", name)
    ui.click("#key-form button")
    expect(ui.locator("#new-key code")).to_have_text(re.compile(r"^rag_"))
    key = ui.locator("#new-key code").inner_text()
    assert ui.request.get("/v1/documents", headers={"Authorization": f"Bearer {key}"}).status == 200

    row = ui.locator("#key-rows tr", has_text=name)
    row.locator("[data-revoke]").click()
    expect(row).to_have_count(0)
    assert ui.request.get("/v1/documents", headers={"Authorization": f"Bearer {key}"}).status == 401


def test_setup_page_masks_secrets(ui):
    ui.click("nav a[href='#setup']")
    expect(ui.locator("#setup-form [name='vector_store.type']")).not_to_have_value("")
    for el in ui.locator("#setup-form input[type=password]").all():
        assert el.input_value() in ("", "********")


def test_logout(ui):
    ui.click("#logout")
    expect(ui.locator("#view-login")).to_be_visible()
    expect(ui.locator("#nav")).to_be_hidden()
    assert ui.request.get("/api/documents").status == 401
