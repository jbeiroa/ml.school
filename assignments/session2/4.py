from metaflow import FlowSpec, step, Parameter

class ForEachSquare(FlowSpec):
    """A class that uses foreach to square a list of numbers in a parameter."""

    numbers = Parameter("numbers",
                        help="A list of numbers",
                        default=[1,2,3,4])

    @step
    def start(self):
        self.next(self.square, foreach="numbers")

    @step
    def square(self):
        """Square each parameter."""
        self.squared = int(self.input) ** 2
        print(self.squared)
        self.next(self.join)

    @step
    def join(self, inputs):
        self.squared_numbers = [i.squared for i in inputs]
        self.next(self.end)

    @step
    def end(self):
        print(f"Orignal list of numbers: {self.numbers}")
        print(f"Squared numbers: {self.squared_numbers}")
        print(f"Sum of squared numbers: {sum(self.squared_numbers)}")


if __name__ == "__main__":
    ForEachSquare()