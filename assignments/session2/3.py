from metaflow import FlowSpec, step

class Branches(FlowSpec):
    """A flow with some parallel branchess"""

    @step
    def start(self):
        self.number = 25
        self.next(self.add_constant, self.multiply_constant)

    @step
    def add_constant(self):
        self.addition = self.number + 25
        self.next(self.join)
    
    @step
    def multiply_constant(self):
        self.multiplication = self.number * 3
        self.next(self.join)

    @step
    def join(self, inputs):
        self.merge_artifacts(inputs, exclude="common")
        print(self.addition + self.multiplication)
        self.next(self.end)

    @step
    def end(self):
        print("This is the end, buddy.")

if __name__ == "__main__":
    Branches()