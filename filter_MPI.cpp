#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

// ==== Helper clamp ====
inline int clampValue(int val, int minVal, int maxVal) {
    if (val < minVal) return minVal;
    if (val > maxVal) return maxVal;
    return val;
}

class Image {
public:
    std::string magic;
    int width, height, maxColor;
    std::vector<int> pixels; // P3 = RGB consecutivos, P2 = escala de grises

    bool load(const std::string& filename) {
        std::ifstream in(filename.c_str());
        if (!in.is_open()) {
            std::cerr << "Error abriendo archivo: " << filename << "\n";
            return false;
        }
        in >> magic >> width >> height >> maxColor;
        int pixelCount = (magic == "P3") ? width * height * 3 : width * height;
        pixels.resize(pixelCount);
        for (int i = 0; i < pixelCount; i++) in >> pixels[i];
        return true;
    }

    bool save(const std::string& filename) {
        std::ofstream out(filename.c_str());
        if (!out.is_open()) {
            std::cerr << "Error guardando archivo: " << filename << "\n";
            return false;
        }
        out << magic << "\n" << width << " " << height << "\n" << maxColor << "\n";
        for (size_t i = 0; i < pixels.size(); i++) {
            out << pixels[i] << "\n";
        }
        return true;
    }
};

// ==== Filtros ====
class Filter {
public:
    virtual void applyRegion(const Image& input, Image& output, int startY, int endY) = 0;
    virtual ~Filter() {}
};

class ConvolutionFilter : public Filter {
protected:
    std::vector<std::vector<float> > kernel;
public:
    ConvolutionFilter(const std::vector<std::vector<float> >& k) : kernel(k) {}

    void applyRegion(const Image& input, Image& output, int startY, int endY) {
        int channels = (input.magic == "P3") ? 3 : 1;
        int kw = kernel[0].size();
        int half = kw / 2;

        for (int y = startY; y < endY; y++) {
            for (int x = 0; x < input.width; x++) {
                for (int c = 0; c < channels; c++) {
                    float sum = 0.0f;
                    for (int ky = -half; ky <= half; ky++) {
                        for (int kx = -half; kx <= half; kx++) {
                            int nx = x + kx;
                            int ny = y + ky;
                            if (nx >= 0 && nx < input.width && ny >= 0 && ny < input.height) {
                                int idx = (ny * input.width + nx) * channels + c;
                                sum += input.pixels[idx] * kernel[ky+half][kx+half];
                            }
                        }
                    }
                    int idx = (y * input.width + x) * channels + c;
                    output.pixels[idx] = clampValue((int)sum, 0, input.maxColor);
                }
            }
        }
    }
};

class BlurFilter : public ConvolutionFilter {
public:
    BlurFilter() : ConvolutionFilter({
        {1/9.f, 1/9.f, 1/9.f},
        {1/9.f, 1/9.f, 1/9.f},
        {1/9.f, 1/9.f, 1/9.f}
    }) {}
};

class LaplaceFilter : public ConvolutionFilter {
public:
    LaplaceFilter() : ConvolutionFilter({
        {0, -1, 0},
        {-1, 4, -1},
        {0, -1, 0}
    }) {}
};

class SharpenFilter : public ConvolutionFilter {
public:
    SharpenFilter() : ConvolutionFilter({
        {0, -1, 0},
        {-1, 5, -1},
        {0, -1, 0}
    }) {}
};

// ==== MAIN con MPI ====
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0)
            std::cout << "Uso: mpirun -np N " << argv[0] << " input.ppm output.ppm [blur|laplace|sharpen]\n";
        MPI_Finalize();
        return 1;
    }

    Image img, localInput, localOutput;
    Filter* filter = NULL;
    std::string filterArg = argv[3];

    if (filterArg == "blur") filter = new BlurFilter();
    else if (filterArg == "laplace") filter = new LaplaceFilter();
    else if (filterArg == "sharpen") filter = new SharpenFilter();
    else {
        if (rank == 0) std::cerr << "Filtro no reconocido\n";
        MPI_Finalize();
        return 1;
    }

    // ==== Maestro carga imagen ====
    if (rank == 0) {
        img.load(argv[1]);
    }

    // Broadcast metadata
    int width, height, maxColor;
    if (rank == 0) {
        width = img.width;
        height = img.height;
        maxColor = img.maxColor;
    }
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxColor, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::string magic;
    if (rank == 0) magic = img.magic;
    int magicLen = (rank == 0) ? magic.size() : 0;
    MPI_Bcast(&magicLen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) magic.resize(magicLen);
    MPI_Bcast(&magic[0], magicLen, MPI_CHAR, 0, MPI_COMM_WORLD);

    int channels = (magic == "P3") ? 3 : 1;
    int totalPixels = width * height * channels;

    // ==== Scatter rows ====
    int rowsPerProc = height / size;
    int remainder = height % size;
    int localRows = rowsPerProc + (rank < remainder ? 1 : 0);

    std::vector<int> sendcounts(size), displs(size);
    if (rank == 0) {
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = rowsPerProc + (i < remainder ? 1 : 0);
            sendcounts[i] = rows * width * channels;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    int localSize;
    MPI_Scatter(sendcounts.data(), 1, MPI_INT, &localSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    localInput.magic = magic;
    localInput.width = width;
    localInput.height = localRows;
    localInput.maxColor = maxColor;
    localInput.pixels.resize(localSize);
    MPI_Scatterv(img.pixels.data(), sendcounts.data(), displs.data(), MPI_INT,
                 localInput.pixels.data(), localSize, MPI_INT, 0, MPI_COMM_WORLD);

    // ==== Aplicar filtro en bloque ====
    localOutput = localInput;
    filter->applyRegion(localInput, localOutput, 0, localRows);

    // ==== Recolectar ====
    std::vector<int> resultPixels;
    if (rank == 0) resultPixels.resize(totalPixels);

    MPI_Gatherv(localOutput.pixels.data(), localSize, MPI_INT,
                resultPixels.data(), sendcounts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        Image result;
        result.magic = magic;
        result.width = width;
        result.height = height;
        result.maxColor = maxColor;
        result.pixels = resultPixels;
        result.save(argv[2]);
    }

    delete filter;
    MPI_Finalize();
    return 0;
}
