"""
Train a simple two-layer neural network on MNIST or Fashion MNIST.

Example usage:
    uv run train.py                          # Train on MNIST (default, CPU)
    uv run train.py --dataset fashion_mnist  # Train on Fashion MNIST

    # GPU (Linux / macOS)
    USE_GPU="1" uv run train.py

    # GPU (Windows PowerShell)
    $env:USE_GPU="1"; uv run train.py
"""

from nn.array_backend import xp
import nn
import time
import argparse

class MNISTNetwork(nn.Module):
    def __init__(self):
        # 784 inputs (28x28 image) -> 128 hidden -> 10 output classes
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x: nn.Tensor) -> nn.Tensor:
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        return x
        
    def parameters(self) -> list[nn.Tensor]:
        return self.fc1.parameters() + self.fc2.parameters()

def get_accuracy(model: MNISTNetwork, x: xp.ndarray, y_one_hot: xp.ndarray) -> float:
    # We do not need gradients for inference only.
    inputs = nn.Tensor(x, requires_grad=False)

    # Grab the raw backend array to keep math on the GPU (if using CuPy)
    logits = model(inputs).data

    # Small note: I am cheating a bit here...
    # 1. I am using the raw backend array (xp) simply because I did not want to implement the argmax and sum operations in the Tensor class.
    # 2. I am running the entire test set through the model in a single forward pass.
    #    In a real-world scenario, you would run the test set in mini-batches to avoid running out of memory.

    # Convert from one-hot to class indices for comparison
    predictions = xp.argmax(logits, axis=1)
    true_labels = xp.argmax(y_one_hot, axis=1)
    
    num_correct = xp.sum(predictions == true_labels)
    return float(num_correct) / len(true_labels)

def train(dataset_name: str = "mnist"):
    # Load the requested dataset. This will download the dataset from the internet
    # to your local machine's temp directory if not already present.
    if dataset_name == "fashion_mnist":
        (x_train, y_train, x_test, y_test), class_names = nn.datasets.fashion_mnist()
    else:
        (x_train, y_train, x_test, y_test), class_names = nn.datasets.mnist()
        # Shapes: x_train: (60000, 784), y_train: (60000, 10), x_test: (10000, 784), y_test: (10000, 10)

    print(f"Training samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")
    
    # Initialize Model and Optimizer
    model     = MNISTNetwork()
    optimizer = nn.SGD(model.parameters(), learning_rate=0.1)
    
    # Training Hyperparameters
    epochs     = 10
    batch_size = 128
    
    print("\nStarting Training...")
    start_time = time.time()

    # An epoch is one full pass through the training dataset.
    for epoch in range(epochs):
        # Shuffle training data at the start of each epoch
        indices    = xp.random.permutation(len(x_train))
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        # Used to track the average loss for this epoch
        epoch_losses = []
        
        # Mini-batch iteration. Step by batch_size to get each mini-batch.
        for i in range(0, len(x_train), batch_size):
            # Grab the current mini-batch from the shuffled data
            batch_x = x_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]

            # Convert the current mini-batch from NumPy arrays into Tensors for our model to use.
            inputs  = nn.Tensor(batch_x)
            targets = nn.Tensor(batch_y)
            
            # Forward pass
            logits = model(inputs)
            # Turn logits into probabilities using softmax.
            probs = nn.softmax(logits)
            loss  = nn.cross_entropy_loss(probs, targets)
            
            # Zero gradients to prevent gradient accumulation from previous mini-batches
            optimizer.zero_grad()

            # Backward pass
            loss.backward()
            
            # Update weights slightly by using the optimizer
            optimizer.step()

            # Track the loss for this mini-batch
            epoch_losses.append(loss.data)
            
        # Average the loss over all mini-batches for this epoch and compute test accuracy.
        avg_loss = xp.mean(xp.array(epoch_losses))
        test_acc = get_accuracy(model, x_test, y_test)
        print(f"Epoch {epoch + 1:2d}/{epochs} | Avg Loss: {avg_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")

    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds.")

    print("\nVisualizing final predictions...")
    nn.visualize_predictions(model, x_test, y_test, class_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple neural network on MNIST or Fashion MNIST.")
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashion_mnist"],
        default="mnist",
        help="Which dataset to train on (default: mnist).",
    )
    args = parser.parse_args()
    train(args.dataset)

