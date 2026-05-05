# Praktikum Kecerdasan Buatan — Pertemuan 6

> **Topik:** Jaringan Syaraf Tiruan — Perceptron & Backpropagation  
> **Nama:** Izaz Falih  
> **NIM:** H1D024034  
> **Program Studi:** Informatika  
> **Mata Kuliah:** Praktikum Kecerdasan Buatan  

---

## 📋 Deskripsi

Repositori ini berisi implementasi dua algoritma dasar Jaringan Syaraf Tiruan (JST) menggunakan Python:

1. **Perceptron** — untuk menyelesaikan masalah klasifikasi linear (gerbang logika OR)
2. **Backpropagation** — untuk menyelesaikan masalah non-linear (gerbang logika XOR)

Kedua model dibangun dari awal (*from scratch*) menggunakan NumPy tanpa framework deep learning eksternal.

---

## 📁 Struktur File

```
Pertemuan6/
├── Perceptron.py            # Implementasi kelas Perceptron
├── Perceptron_or.py         # Runner: melatih Perceptron pada masalah OR
├── Backpropagation.py       # Implementasi kelas Backpropagation
├── Backpropagation_xor.py   # Runner: melatih Backpropagation pada masalah XOR
├── HasilPerceptron.txt      # Output hasil pelatihan Perceptron
├── hasilBackpropagation.txt # Output hasil pelatihan Backpropagation
└── Pert6.pdf                # Modul praktikum pertemuan 6
```

---

## 🧠 Algoritma

### 1. Perceptron (`Perceptron.py`)

Model Perceptron sederhana dengan **fungsi aktivasi bipolar** (`+1` atau `-1`).

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `alpha`   | `0.1`   | Learning rate |
| `epoch`   | `10`    | Maksimum epoch pelatihan |

**Alur kerja:**
1. Inisialisasi bobot dan bias = **0**
2. Hitung net input: `y_in = Σ(xi * wi) + b`
3. Terapkan fungsi aktivasi bipolar
4. Perbarui bobot menggunakan **Delta Rule**: `Δw = α × error × xi`
5. Ulangi hingga SSE = 0 atau max epoch tercapai

**Dataset (OR Bipolar):**

| x1 | x2 | Target |
|----|----|--------|
| 1  | 1  | 1      |
| 1  | -1 | 1      |
| -1 | 1  | 1      |
| -1 | -1 | -1     |

---

### 2. Backpropagation (`Backpropagation.py`)

Model Backpropagation dengan **arsitektur 2-2-1** (2 input, 2 hidden, 1 output) menggunakan **fungsi aktivasi sigmoid bipolar (tanh)**.

| Parameter      | Nilai   | Keterangan |
|----------------|---------|------------|
| `alpha`        | `0.3`   | Learning rate |
| `epoch`        | `1000`  | Maksimum epoch pelatihan |
| `target_error` | `0.001` | Target SSE minimum |

**Arsitektur Jaringan:**
```
Input Layer   Hidden Layer   Output Layer
   x1  ──┐
          ├──► h1 ──┐
   x2  ──┘          ├──► y
          ┌──► h2 ──┘
```

**Alur kerja:**

*Forward Propagation:*
1. Hitung input hidden layer: `h_in = X · W_hidden + b_hidden`
2. Aktivasi hidden: `h = tanh(h_in)`
3. Hitung input output layer: `y_in = h · W_output + b_output`
4. Aktivasi output: `y = tanh(y_in)`

*Backward Propagation:*
1. Hitung error: `error = target - y`
2. Hitung delta output: `δy = error × (1 - y²)`
3. Propagasi error ke hidden: `error_h = δy · W_output^T`
4. Hitung delta hidden: `δh = error_h × (1 - h²)`
5. Perbarui bobot dan bias semua layer

**Dataset (XOR Bipolar):**

| x1 | x2 | Target |
|----|----|--------|
| 1  | 1  | -1     |
| 1  | -1 | 1      |
| -1 | 1  | 1      |
| -1 | -1 | -1     |

---

## ▶️ Cara Menjalankan

### Prasyarat
Pastikan Python dan library berikut telah terinstal:
```bash
pip install numpy matplotlib
```

### Menjalankan Perceptron (masalah OR)
```bash
python Perceptron_or.py
```
- Hasil pelatihan disimpan di `HasilPerceptron.txt`
- Grafik *decision boundary* ditampilkan setiap epoch

### Menjalankan Backpropagation (masalah XOR)
```bash
python Backpropagation_xor.py
```
- Hasil pelatihan disimpan di `hasilBackpropagation.txt`
- Grafik penurunan error (SSE) per epoch ditampilkan

---

## 📊 Output

### Perceptron
- **`HasilPerceptron.txt`**: Log detail setiap epoch berisi input, target, prediksi, error, bobot, dan bias; diakhiri dengan bobot dan bias akhir.
- **Grafik Decision Boundary**: Visualisasi garis pemisah data di setiap epoch.

### Backpropagation
- **`hasilBackpropagation.txt`**: Log lengkap proses forward & backward propagation setiap epoch, termasuk nilai bobot dan bias yang diperbarui.
- **Grafik SSE**: Visualisasi penurunan *Sum Square Error* per epoch hingga mencapai konvergensi.

---

## 🔑 Konsep Kunci

| Konsep | Keterangan |
|--------|-----------|
| **Perceptron** | Model JST paling sederhana; hanya bisa memisahkan data yang **linearly separable** |
| **Backpropagation** | Algoritma pelatihan JST multi-layer dengan propagasi error ke belakang |
| **Sigmoid Bipolar (tanh)** | Fungsi aktivasi dengan output di rentang `[-1, 1]` |
| **Delta Rule** | Aturan pembaruan bobot berdasarkan selisih output dan target |
| **SSE (Sum Square Error)** | Metrik error = `Σ(target - output)²` |
| **Learning Rate (α)** | Menentukan seberapa besar langkah pembaruan bobot |

---

## 📚 Referensi

- Modul Praktikum Kecerdasan Buatan Pertemuan 6 (`Pert6.pdf`)
- Fausett, L. (1994). *Fundamentals of Neural Networks*. Prentice Hall.
- NumPy Documentation: https://numpy.org/doc/
- Matplotlib Documentation: https://matplotlib.org/stable/

---

<div align="center">
  <sub>Universitas Jenderal Soedirman &nbsp;•&nbsp; Fakultas Teknik &nbsp;•&nbsp; Informatika &nbsp;•&nbsp; 2025/2026</sub>
</div>
