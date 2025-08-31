#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <omp.h>

// clampValue: asegura que un valor entero se encuentre dentro de un rango [minVal, maxVal].
// Esto se usa para evitar que los valores de los píxeles se salgan del rango válido.
inline int clampValue(int val, int minVal, int maxVal) {
    if (val < minVal) return minVal;
    if (val > maxVal) return maxVal;
    return val;
}

// La clase Image representa una imagen en memoria.
// Contiene sus metadatos (tipo P2/P3, ancho, alto, valor máximo de color) y los píxeles.
class Image {
public:
    std::string magic;
    int width, height, maxColor;
    std::vector<int> pixels;

    // load: carga una imagen desde un archivo .pgm o .ppm en memoria
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

    // save: guarda una imagen desde memoria a un archivo .pgm o .ppm
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

// Clase padre Filter
class Filter {
public:
    // aplicar: función virtual pura que cada filtro debe implementar
    virtual void aplicar(const Image& input, Image& output) = 0;
    virtual ~Filter() {}
};

// Clase Padre ConvolutionFilter: implementa filtros de convolución
class ConvolutionFilter : public Filter {
protected:
    std::vector<std::vector<float> > kernel;
public:
    ConvolutionFilter(const std::vector<std::vector<float> >& k) : kernel(k) {}

    // aplicar: aplica el kernel sobre toda la imagen
    void aplicar(const Image& input, Image& output) {
        output = input;
        int channels = (input.magic == "P3") ? 3 : 1;
        int kw = kernel[0].size();
        int kh = kernel.size();
        int half = kw / 2;

        // Directiva OpenMP para paralelizar los bucles anidados
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < input.height; y++) {
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

// Filtro Blur
class BlurFilter : public ConvolutionFilter {
public:
    BlurFilter() : ConvolutionFilter({
        {1/9.f, 1/9.f, 1/9.f},
        {1/9.f, 1/9.f, 1/9.f},
        {1/9.f, 1/9.f, 1/9.f}
    }) {}
};

// Filtro Laplaciano
class LaplaceFilter : public ConvolutionFilter {
public:
    LaplaceFilter() : ConvolutionFilter({
        {0, -1, 0},
        {-1, 4, -1},
        {0, -1, 0}
    }) {}
};

// Filtro Sharpen
class SharpenFilter : public ConvolutionFilter {
public:
    SharpenFilter() : ConvolutionFilter({
        {0, -1, 0},
        {-1, 5, -1},
        {0, -1, 0}
    }) {}
};


int main(int argc, char* argv[]) {


    Image img, result;
    // load: carga una imagen desde un archivo .pgm o .ppm en memoria
    if (!img.load(argv[1])) return 1;
    result = img;

    std::string filterArg = argv[3];
    Filter* filter = NULL;

    // Seleccionar filtro según el argumento
    if (filterArg == "blur") filter = new BlurFilter();
    else if (filterArg == "laplace") filter = new LaplaceFilter();
    else if (filterArg == "sharpen") filter = new SharpenFilter();
    else {
        std::cerr << "Filtro no creado: " << filterArg << "\n";
        return 1;
    }

    filter->aplicar(img, result);
    
    result.save(argv[2]);

    delete filter;
    return 0;
}