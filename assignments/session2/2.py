from metaflow import FlowSpec, step

class SimpleFlow(FlowSpec):
    """A simple flow that tracks a sequence of numerical operations."""

    @step
    def start(self):
        """Initialize the start value artifact."""
        self.start_value = 1
        self.values = [self.start_value]
        self.next(self.add_one)

    @step
    def add_one(self):
        """Add one to the start value."""
        self.added_one = self.start_value + 1
        self.values.append(self.added_one)
        self.next(self.power_of_two)

    @step
    def power_of_two(self):
        """Raise the added_one value to the power of two."""
        self.powered = self.added_one ** 2
        self.values.append(self.powered)
        self.next(self.end)

    @step
    def end(self):
        """Print the final artifact values and compute average and sum."""
        print("Start value:", self.start_value)
        print("Added one:", self.added_one)
        print("Powered:", self.powered)
        print("All values:", self.values)
        print("Average:", sum(self.values) / len(self.values))
        print("Sum:", sum(self.values))

if __name__ == "__main__":
    SimpleFlow()
