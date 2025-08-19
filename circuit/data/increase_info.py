import sys

def extract_features(img_flat, TOL):
    size = 28  # MNIST immagini 28x28
    img = [img_flat[i*size:(i+1)*size] for i in range(size)]  # reshape in matrice

    features = []

    # --- Sliding windows blocks (2x2, 4x4, 7x7) ---
    for block_size in [2, 4, 7]:
        for r in range(size - block_size + 1):
            for c in range(size - block_size + 1):
                block = [img[rr][c:c+block_size] for rr in range(r, r+block_size)]
                count = sum(sum(row) for row in block)
                features.append(1 if count > TOL * (block_size * block_size) else 0)

    # --- Statistiche globali ---
    stats = []

    # 1. Conteggi per riga
    for r in range(size):
        stats.append(sum(img[r]))

    # 2. Conteggi per colonna
    for c in range(size):
        stats.append(sum(img[r][c] for r in range(size)))

    # 3. Centro di massa (convertito in intero)
    coords = [(r, c) for r in range(size) for c in range(size) if img[r][c] == 1]
    if coords:
        mean_r = int(round(sum(r for r, _ in coords) / len(coords)))
        mean_c = int(round(sum(c for _, c in coords) / len(coords)))
    else:
        mean_r = mean_c = 0
    stats.extend([mean_r, mean_c])

    # 4. Bounding box
    if coords:
        min_r = min(r for r, _ in coords)
        max_r = max(r for r, _ in coords)
        min_c = min(c for _, c in coords)
        max_c = max(c for _, c in coords)
    else:
        min_r = max_r = min_c = max_c = 0
    stats.extend([min_r, max_r, min_c, max_c])

    # 5. Totale pixel accesi
    total_on = sum(sum(row) for row in img)
    stats.append(total_on)

    # 6. Simmetria orizzontale e verticale (convertita in 0/1)
    left = sum(img[r][c] for r in range(size) for c in range(size//2))
    right = sum(img[r][c] for r in range(size) for c in range(size//2, size))
    sym_h = 1 if abs(left - right) < 1e-5 else 0

    top = sum(img[r][c] for r in range(size//2) for c in range(size))
    bottom = sum(img[r][c] for r in range(size//2, size) for c in range(size))
    sym_v = 1 if abs(top - bottom) < 1e-5 else 0

    stats.extend([sym_h, sym_v])

    # Replico le statistiche 4 volte
    stats = stats * 4  # dimensione 260

    # --- Risultato finale: pixel flatten + blocchi + stats replicate ---
    return img_flat + features + stats


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

    # stampa solo lunghezze (dovrebbe essere 2882)
    print(len(data[0]))

    # scrivi su new_file.txt
    with open("new_file.txt", "w") as f:
        for img, lbl in zip(data, labels):
            f.write("".join(map(str, img)) + "\n")
            f.write(str(lbl) + "\n")


if __name__ == "__main__":
    main()
