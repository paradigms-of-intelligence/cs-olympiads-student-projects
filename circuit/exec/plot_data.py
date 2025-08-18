import matplotlib.pyplot as plt

def plot_results():
    # Load the scores
    labels, scores = [], []
    with open("plot_data.txt", "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                labels.append(parts[0])
                scores.append(float(parts[1]))

    # Plot
    plt.figure(figsize=(8,5))
    plt.bar(labels, scores, color="skyblue", edgecolor="black")

    # Fix y-axis from 0 to 100
    plt.ylim(0, 100)

    # Horizontal lines
    if scores:
        best = max(scores)
        plt.axhline(best, color="green", linestyle="--", linewidth=1.5, label=f"Best: {best:.2f}%")

    
    plt.xlabel("Experiment")
    plt.ylabel("Accuracy (%)")
    plt.title("Network Evaluation Results")
    plt.legend()
    plt.grid(axis="y", linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.savefig("plot.png")


if __name__ == "__main__":
    plot_results()