"""Capture local app screenshots through a temporary, headless Chrome session."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


async def main():
    import websockets

    chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    if not chrome.exists():
        raise FileNotFoundError("This screenshot helper expects Chrome at its Windows installation path.")
    profile = ROOT / ".cache/chrome-preview"
    profile.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([str(chrome), "--headless=new", "--disable-gpu", "--no-first-run",
                                "--no-default-browser-check", "--remote-debugging-port=9223",
                                f"--user-data-dir={profile}", "about:blank"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        pages = None
        for _ in range(20):
            try:
                with urllib.request.urlopen("http://127.0.0.1:9223/json", timeout=2) as response:
                    pages = json.load(response)
                break
            except OSError:
                await asyncio.sleep(.5)
        if not pages:
            raise RuntimeError("Headless Chrome did not start.")
        target = next(page for page in pages if page["type"] == "page")
        async with websockets.connect(target["webSocketDebuggerUrl"], max_size=20 * 1024 * 1024) as socket:
            next_id = 0

            async def command(method, params=None):
                nonlocal next_id
                next_id += 1
                request_id = next_id
                await socket.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
                while True:
                    message = json.loads(await socket.recv())
                    if message.get("id") == request_id:
                        if "error" in message:
                            raise RuntimeError(message["error"])
                        return message.get("result", {})

            await command("Page.enable")
            await command("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1080,
                          "deviceScaleFactor": 1, "mobile": False})
            output = ROOT / "presentation/screenshots"
            output.mkdir(parents=True, exist_ok=True)
            for language in ["en", "ar"]:
                await command("Page.navigate", {"url": f"http://127.0.0.1:8501/?lang={language}"})
                await asyncio.sleep(8)
                captured = await command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
                (output / f"app-{language}.png").write_bytes(base64.b64decode(captured["data"]))
                print(output / f"app-{language}.png")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    asyncio.run(main())
