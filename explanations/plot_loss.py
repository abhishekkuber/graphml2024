import matplotlib.pyplot as plt


def plot_loss_curve(filename):
    # Read loss values from file
    with open(filename, 'r') as file:
        losses = [float(line.strip()) for line in file.readlines()]

    # Create a plot of the losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show()


# Assuming your file is named 'losses.txt' and located in the same directory as your script
plot_loss_curve('losses.txt')
