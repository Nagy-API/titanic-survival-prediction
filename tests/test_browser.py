import os
import re
import socket
from threading import Thread
from time import monotonic, sleep

import pytest
import uvicorn
from playwright.sync_api import expect, sync_playwright

from app.main import create_app


@pytest.fixture(scope="module")
def live_site():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(create_app(), log_level="warning"))
        thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = monotonic() + 15
            while not server.started and thread.is_alive() and monotonic() < deadline:
                sleep(0.02)
            assert server.started, "Uvicorn failed to start"
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=10)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        channel = os.environ.get("PLAYWRIGHT_CHANNEL")
        browser = playwright.chromium.launch(channel=channel)
        yield browser
        browser.close()


@pytest.mark.parametrize("width", [390, 768, 1440])
def test_keyboard_prediction_and_responsive_layout(browser, live_site, width):
    page = browser.new_page(viewport={"width": width, "height": 900}, reduced_motion="reduce")
    try:
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        response = page.goto(live_site)
        assert response.status == 200
        assert page.evaluate("document.documentElement.scrollWidth === document.documentElement.clientWidth")
        for label in ("Passenger class", "Sex in dataset", "Title"):
            expect(page.get_by_label(label, exact=True)).to_be_visible()
        assert page.get_by_label("Passenger name", exact=True).count() == 0
        assert page.evaluate("document.querySelector('#predictionForm').checkValidity()") is False
        predict = page.get_by_role("button", name="Predict survival")
        assert predict.is_enabled()
        predict.click()
        expect(page.get_by_label("Passenger class", exact=True)).to_be_focused()

        page.keyboard.press("Tab")
        expect(page.get_by_label("Sex in dataset", exact=True)).to_be_focused()
        sample = page.get_by_role("button", name="Try a sample")
        sample.focus()
        page.keyboard.press("Enter")
        expect(page.get_by_label("Passenger class", exact=True)).not_to_have_value("")
        expect(page.get_by_label("Sex in dataset", exact=True)).not_to_have_value("")
        expect(page.get_by_label("Title", exact=True)).not_to_have_value("")

        page.get_by_label(re.compile(r"^Age \(years\)")).fill("31")
        predict.focus()
        with page.expect_response("**/api/predict") as result:
            page.keyboard.press("Enter")
        response = result.value
        assert response.ok
        assert response.request.post_data_json["Age"] == 31
        reply = response.json()
        expected_label = "Survived" if reply["prediction_code"] == 1 else "Did not survive"
        expect(page.locator("#prediction-label")).to_have_text(expected_label)
        expect(page.locator("#prediction-percentage")).to_contain_text("%")
        displayed = page.locator("#prediction-percentage").inner_text()
        assert float(displayed.strip().removesuffix("%")) == pytest.approx(
            reply["survival_percentage"], abs=0.05
        )
        expect(page.locator("#prediction-result")).to_be_visible()
        expect(page.locator("#prediction-result")).to_be_focused()
        page.locator("#prediction-result summary").click()
        expect(page.locator("#used-inputs")).to_be_visible()
        expect(page.locator("#used-inputs")).to_contain_text("31")
        assert page.evaluate("document.documentElement.scrollWidth === document.documentElement.clientWidth")
        assert not errors

        page.get_by_role("button", name="Reset").click()
        expect(page.locator("#prediction-result")).to_be_hidden()
        expect(page.get_by_label("Passenger class", exact=True)).to_have_value("")
    finally:
        page.close()


def test_loading_and_safe_error_state(browser, live_site):
    page = browser.new_page()
    try:
        pending = []
        page.route("**/api/predict", lambda request: pending.append(request))
        page.goto(live_site)
        page.get_by_role("button", name="Try a sample").click()
        page.get_by_role("button", name="Predict survival").click()
        expect(page.get_by_role("button", name="Estimating…")).to_be_disabled()
        expect(page.locator("#prediction-status")).not_to_have_text("")
        assert len(pending) == 1
        marker = "<em>Unexpected private value</em>"
        pending[0].fulfill(status=503, json={"detail": marker})
        error = page.locator("#prediction-error")
        expect(error).to_be_visible()
        expect(error).not_to_contain_text(marker)
        expect(error.locator("em")).to_have_count(0)
        expect(page.get_by_role("button", name="Predict survival")).to_be_enabled()
    finally:
        page.close()
