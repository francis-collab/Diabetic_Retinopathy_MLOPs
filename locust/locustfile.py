from locust import HttpUser, task, between
from PIL import Image
import io


class DRUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        img = Image.new("RGB", (128, 128), (40, 10, 10))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.img_bytes = buf.getvalue()

    @task(5)
    def predict(self):
        self.client.post(
            "/predict",
            files={"file": ("fundus.png", self.img_bytes, "image/png")},
            timeout=10,
        )

    @task(1)
    def health(self):
        self.client.get("/health", timeout=5)
