#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>


using namespace std;

inline int clampValue(int val, int minVal, int maxVal) {
    if (val < minVal) return minVal;
    if (val > maxVal) return maxVal;
    return val;
}


class Image {
public:
    string magic;
    int width, height, maxColor;
    vector<int> pixels;

    bool load(const string& filename) {
        ifstream in(filename.c_str());
        if (!in.is_open()) return false;

        in >> magic >> width >> height >> maxColor;
        int channels = (magic == "P3") ? 3 : 1;
        pixels.resize(width * height * channels);
        for (int i = 0; i < (int)pixels.size(); i++) in >> pixels[i];
        return true;
    }

    bool save(const string& filename) {
        ofstream out(filename.c_str());
        if (!out.is_open()) return false;

        out << magic << "\n" << width << " " << height << "\n" << maxColor << "\n";
        for (int i = 0; i < (int)pixels.size(); i++) out << pixels[i] << "\n";
        return true;
    }
};


void applyBlur(const Image& input, Image& output, int startRow, int endRow) {
    output = input; // copiar metadatos
    int channels = (input.magic == "P3") ? 3 : 1;
    float kernel[3][3] = {
        {1/9.f, 1/9.f, 1/9.f},
        {1/9.f, 1/9.f, 1/9.f},
        {1/9.f, 1/9.f, 1/9.f}
    };

    for (int y = startRow; y < endRow; y++) {
        for (int x = 0; x < input.width; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int nx = x + kx;
                        int ny = y + ky;
                        if (nx >= 0 && nx < input.width && ny >= 0 && ny < input.height) {
                            int idx = (ny * input.width + nx) * channels + c;
                            sum += input.pixels[idx] * kernel[ky+1][kx+1];
                        }
                    }
                }
                int idx = (y * input.width + x) * channels + c;
                output.pixels[idx] = clampValue((int)sum, 0, input.maxColor);
            }
        }
    }
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Image img, localResult;
    int width, height, maxColor, channels;

    if (rank == 0) {
        if (!img.load(argv[1])) {
            cerr << "Error cargando imagen\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        width = img.width;
        height = img.height;
        maxColor = img.maxColor;
        channels = (img.magic == "P3") ? 3 : 1;
    }

    // Compartir metadatos con todos los procesos
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxColor, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) img.magic = "P3"; 
    channels = 3;

    // Dividir filas entre procesos
    int rowsPerProc = height / size;
    int startRow = rank * rowsPerProc;
    int endRow = (rank == size - 1) ? height : startRow + rowsPerProc;

    // Crear salida parcial
    localResult.magic = img.magic;
    localResult.width = width;
    localResult.height = height;
    localResult.maxColor = maxColor;
    localResult.pixels.resize(width * height * channels);

    // Todos reciben toda la imagen (simplificación, no scatter)
    if (rank != 0) img.pixels.resize(width * height * channels);
    MPI_Bcast(img.pixels.data(), width * height * channels, MPI_INT, 0, MPI_COMM_WORLD);

    // Cada proceso aplica blur a su bloque
    applyBlur(img, localResult, startRow, endRow);

    // Recolectar resultados en el maestro
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : localResult.pixels.data(),
               localResult.pixels.data(),
               width * height * channels, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Maestro guarda imagen final
    if (rank == 0) {
        localResult.save(argv[2]);
        cout << "Imagen guardada en " << argv[2] << "\n";
    }

    MPI_Finalize();
    return 0;
}
