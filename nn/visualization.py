from nn.array_backend import xp
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from nn.tensor import Tensor
from nn.modules import Module


class PredictionViewer:
    def __init__(self, model, x_test, y_test, class_names=None):
        # 1. Generate and move data to CPU safely
        inputs = Tensor(x_test, requires_grad=False)
        logits = model(inputs).data
        predictions = xp.argmax(logits, axis=1)
        true_classes = xp.argmax(y_test, axis=1)
        
        # Optional human-readable labels (e.g. Fashion MNIST class names).
        # Defaults to the numeric class index when not provided.
        self.class_names = class_names
        
        if hasattr(x_test, "get"):
            # Handle CuPy arrays by moving them to CPU for visualization
            self.images = x_test.get()
            self.preds  = predictions.get()
            self.trues  = true_classes.get()
        else:
            # Handle NumPy arrays directly
            self.images = x_test
            self.preds  = predictions
            self.trues  = true_classes

        self.total     = len(x_test)
        self.index     = 0
        self.page_size = 12 # 3 rows x 4 cols

        # 2. Setup Figure and UI Elements
        self.fig, self.axes = plt.subplots(3, 4, figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("Model Predictions")
        self.fig.subplots_adjust(bottom=0.15)
        self.flat_axes = self.axes.flatten()

        # Add buttons
        ax_prev = self.fig.add_axes([0.3, 0.04, 0.15, 0.07])
        ax_next = self.fig.add_axes([0.55, 0.04, 0.15, 0.07])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_prev.on_clicked(self.prev_page)
        
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.next_page)

        self.update_display()

    def update_display(self):
        for ax in self.flat_axes:
            ax.clear()
            ax.axis('off')

        start = self.index
        end = min(start + self.page_size, self.total)
        
        for i in range(end - start):
            idx = start + i
            ax = self.flat_axes[i]
            
            img = self.images[idx].reshape(28, 28)
            pred = self.preds[idx]
            true = self.trues[idx]
            
            ax.imshow(img, cmap='gray')
            color = 'green' if pred == true else 'red'
            if self.class_names is not None:
                pred_label = self.class_names[pred]
                true_label = self.class_names[true]
                ax.set_title(f"P: {pred_label} | T: {true_label}", color=color, fontsize=8, fontweight='bold')
            else:
                ax.set_title(f"P: {pred} | T: {true}", color=color, fontsize=10, fontweight='bold')

        self.fig.suptitle(f"Showing samples {start} to {end} of {self.total}", fontsize=12)
        self.fig.canvas.draw_idle()

    def next_page(self, event):
        if self.index + self.page_size < self.total:
            self.index += self.page_size
            self.update_display()

    def prev_page(self, event):
        if self.index - self.page_size >= 0:
            self.index -= self.page_size
            self.update_display()

def visualize_predictions(model: Module, x_test: xp.ndarray, y_test: xp.ndarray, class_names=None):
    """Instantiates and blocks on the interactive viewer."""
    # Keep reference in the local scope of this function so it isn't garbage collected
    global _active_viewer
    _active_viewer = PredictionViewer(model, x_test, y_test, class_names)
    plt.show()

