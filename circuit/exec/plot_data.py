import matplotlib.pyplot as plt
import numpy as np

def main():
    labels = []
    accuracy = []

    with open("plot_data.txt") as file:
        for line in file:
            l, a = line.strip().split()
            labels.append(l)
            accuracy.append(float(a))

    labels = np.array(labels)
    accuracy = np.array(accuracy)

    # Use index positions for plotting
    x = np.arange(len(labels))

    plt.bar(x, accuracy)
    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Network type")
    plt.ylabel("Accuracy [%]")
    plt.title("Network Accuracy Comparison")
    plt.tight_layout()

    plt.savefig("plot.png")  # always works
    # plt.show()  # only if you have GUI

if __name__ == "__main__":
    main()
