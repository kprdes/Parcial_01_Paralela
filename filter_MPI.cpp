#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

using namespace std;
using namespace std::chrono;

inline int clampValue(int val, int minVal, int maxVal) {
    return max(minVal, min(val, maxVal));
}

class Image {
public:
    string magic;
    int width, height, maxColor;
    vector<int> pixels;

    bool load(const string& filename) {
        ifstream in(filename);
        if (!in.is_open()) return false;
        in >> magic >> width >> height >> maxColor;
        int channels = (magic == "P3") ? 3 : 1;
        pixels.resize(width * height * channels);
        for (int &p : pixels) in >> p;
        return true;
    }

    bool save(const string& filename) {
        ofstream out(filename);
        if (!out.is_open()) return false;
        out << magic << "\n" << width << " " << height << "\n" << maxColor << "\n";
        for (int p : pixels) out << p << "\n";
        return true;
    }
};

// Aplica un kernel en un rango de filas
void applyKernel(const Image& input, vector<int>& output,
                 int startRow, int endRow, int channels,
                 const vector<vector<float>>& kernel) {
    int w = input.width, h = input.height;
    int half = kernel.size() / 2;
    output.resize((endRow - startRow) * w * channels);

    for (int y = startRow; y < endRow; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0;
                for (int ky = -half; ky <= half; ky++) {
                    for (int kx = -half; kx <= half; kx++) {
                        int nx = x + kx, ny = y + ky;
                        if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                            int idx = (ny * w + nx) * channels + c;
                            sum += input.pixels[idx] * kernel[ky + half][kx + half];
                        }
                    }
                }
                int localY = y - startRow;
                int idxOut = (localY * w + x) * channels + c;
                output[idxOut] = clampValue((int)sum, 0, input.maxColor);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) cerr << "Uso: mpirun -np N ./mpi_filterer input.ppm output.ppm [blur|laplace|sharpen]\n";
        MPI_Finalize();
        return 1;
    }

    // Definir kernel
    vector<vector<float>> kernel;
    string filter = argv[3];
    if (filter == "blur") {
        kernel = {{1/9.f,1/9.f,1/9.f},{1/9.f,1/9.f,1/9.f},{1/9.f,1/9.f,1/9.f}};
    } else if (filter == "laplace") {
        kernel = {{0,-1,0},{-1,4,-1},{0,-1,0}};
    } else if (filter == "sharpen") {
        kernel = {{0,-1,0},{-1,5,-1},{0,-1,0}};
    } else {
        if (rank == 0) cerr << "Filtro no reconocido\n";
        MPI_Finalize();
        return 1;
    }

    Image img;
    int w,h,maxColor,channels;
    string magic;

    if (rank == 0) {
        if (!img.load(argv[1])) {
            cerr << "Error cargando imagen\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        w = img.width; h = img.height; maxColor = img.maxColor;
        channels = (img.magic == "P3") ? 3 : 1;
        magic = img.magic;
    }

    // Compartir metadatos
    MPI_Bcast(&w,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&h,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&maxColor,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&channels,1,MPI_INT,0,MPI_COMM_WORLD);

    if (rank != 0) {
        img.width=w; img.height=h; img.maxColor=maxColor;
        img.magic=(channels==3)?"P3":"P2";
        img.pixels.resize(w*h*channels);
    }

    // Compartir imagen completa
    MPI_Bcast(img.pixels.data(), w*h*channels, MPI_INT, 0, MPI_COMM_WORLD);

    // División de trabajo
    int rowsPerProc = h / size;
    int startRow = rank * rowsPerProc;
    int endRow = (rank == size-1) ? h : startRow + rowsPerProc;

    // Cronómetro
    auto start = high_resolution_clock::now();

    vector<int> localBlock;
    applyKernel(img, localBlock, startRow, endRow, channels, kernel);

    auto end = high_resolution_clock::now();
    double elapsed = duration<double>(end - start).count();

    // Recolectar resultados
    vector<int> recvCounts(size), displs(size);
    for (int i=0; i<size; i++) {
        int s=i*rowsPerProc, e=(i==size-1)?h:s+rowsPerProc;
        recvCounts[i]=(e-s)*w*channels;
    }
    displs[0]=0;
    for (int i=1; i<size; i++) displs[i]=displs[i-1]+recvCounts[i-1];

    vector<int> finalPixels;
    if (rank==0) finalPixels.resize(w*h*channels);

    MPI_Gatherv(localBlock.data(), localBlock.size(), MPI_INT,
                rank==0?finalPixels.data():nullptr, recvCounts.data(), displs.data(),
                MPI_INT,0,MPI_COMM_WORLD);

    // Mostrar tiempo total y guardar resultado
    if (rank==0) {
        Image result{magic,w,h,maxColor,finalPixels};
        result.save(argv[2]);
        cout << "Filtro aplicado: " << filter << "\n";
        cout << "Imagen guardada en " << argv[2] << "\n";
        cout << "Tiempo total: " << elapsed << " s\n";
    }

    MPI_Finalize();
    return 0;
}
