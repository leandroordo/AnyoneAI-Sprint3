from locust import HttpUser, between, task


class APIUser(HttpUser):
    wait_time = between(1, 5)

    # Put your stress tests here.
    # See https://docs.locust.io/en/stable/writing-a-locustfile.html for help.

    @task
    def test_index(self):
        self.client.get("http://127.0.0.1/")

    @task
    def test_predict(self):
        files = [("file", ("dog.jpeg", open("dog.jpeg", "rb"), "image/jpeg"))]
        headers = {}
        payload = {}
        self.client.post("http://127.0.0.1/predict",
            headers=headers,
            data=payload,
            files=files,
        )
