from metaflow import FlowSpec, step, retry
import random

def trigger_random_error():
    # List of potential exceptions to raise
    errors = [ValueError, TypeError, KeyError]
    # Randomly pick one and raise it
    raise random.choice(errors)("A random error occurred!")

class RetryFlow(FlowSpec):

    @step
    def start(self):
        self.next(self.retry_step)

    @retry(times=4)
    @step
    def retry_step(self):
        trigger_random_error()
        self.next(self.end)

    @step
    def end(self):
        print("Pipeline finished successfully.")

if __name__ == "__main__":
    RetryFlow()