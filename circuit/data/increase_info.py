import sys

def extract_features(img_flat, TOL):
    size = 28  # MNIST immagini 28x28
    img = [img_flat[i*size:(i+1)*size] for i in range(size)]  # reshape in matrice

    features = []

    for block_size in [2, 4, 7]:
        for r in range(0, size - block_size + 1):
            for c in range(0, size - block_size + 1):
                block = [img[rr][c:c+block_size] for rr in range(r, r+block_size)]
                count = sum(sum(row) for row in block)
                features.append(1 if count > TOL * (block_size*block_size) else 0)

    return img_flat + features


def main():
    TOL = 0.3
    path = sys.argv[1]
    
    data = []
    labels = []
    with open(path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

        for i in range(0, len(lines), 2):
            img = [int(ch) for ch in lines[i]]
            img_features = extract_features(img, TOL)
            data.append(img_features)

            if i + 1 < len(lines):
                labels.append(int(lines[i+1]))

    # stampa solo lunghezze (saranno tutte 2622)
    for d in data:
        print(len(d))

    # scrivi su new_file.txt
    with open("new_file.txt", "w") as f:
        for img, lbl in zip(data, labels):
            f.write("".join(map(str, img)) + "\n")
            f.write(str(lbl) + "\n")


if __name__ == "__main__":
    main()
