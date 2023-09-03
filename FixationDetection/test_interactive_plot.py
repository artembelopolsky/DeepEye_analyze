# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 17:49:40 2023

@author: artem
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


# Create or load your DataFrame
data = {'X': [1, 2, 3, 4, 5],
        'Y': [10, 11, 14, 13, 12]}
df = pd.DataFrame(data)

# Create a custom class to handle button clicks
class PlotUpdater:
    def __init__(self, ax, df):
        self.ax = ax
        self.df = df
        self.current_index = 0
        self.button = Button(ax, 'Next', color='lightgoldenrodyellow')
        self.button.on_clicked(self.next_click)

    def next_click(self, event):
        if self.current_index < len(self.df) - 1:
            self.current_index += 1
            self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.ax.scatter(self.df['X'][:self.current_index + 1], self.df['Y'][:self.current_index + 1], marker='o', label='Data')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.legend()
        self.ax.set_title('Interactive Plot')
        self.ax.grid(True)
        plt.draw()

# Create a figure and initialize the plot
fig, ax = plt.subplots(figsize=(8, 6))
plot_updater = PlotUpdater(ax, df)
plot_updater.update_plot()

# Position the 'Next' button at the bottom of the plot
button_ax = plt.axes([0.8, 0.01, 0.1, 0.05])
plot_updater.button.ax = button_ax
plot_updater.button.label.set_fontsize(10)

plt.show()

