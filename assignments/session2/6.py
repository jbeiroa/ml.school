from metaflow import FlowSpec, card, step, current
from metaflow.cards import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TableViewFlow(FlowSpec):

    @step
    def start(self):
        self.x = np.linspace(0, 100, 1000)
        self.y = np.power(self.x, 2)
        self.df = pd.DataFrame({"x": self.x, "x^2": self.y})
        self.df.loc[:, "x^2"] += np.random.normal(0, 2, len(self.x))
        self.next(self.visualization)

    @card(type="blank")
    @step
    def visualization(self):
        fig, (ax1, ax2)= plt.subplots(1, 2)
        ax1.scatter(self.x, self.y)
        ax2.scatter(self.df.x, self.df["x^2"])
        current.card.append(Image.from_matplotlib(fig))
        self.next(self.end)
    
    @step
    def end(self):
        print("All steps completed.")

if __name__ == "__main__":
    TableViewFlow()