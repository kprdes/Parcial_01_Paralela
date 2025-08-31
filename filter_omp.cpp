#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <omp.h>
#include <chrono>

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
        if (!in.is_open()) {
            cerr << "Error abriendo archivo: " << filename << "\n";
            return false;
        }
        in >> magic >> width >> height >> maxColor;
        int pixelCount = (magic == "P3") ? width * height * 3 : width * height;
        pixels.resize(pixelCount);
        for (int i = 0; i < pixelCount; i++) in >> pixels[i];
        return true;
    }

    bool save(const string& filename) {
        ofstream out(filename.c_str());
        if (!out.is_open()) {
            cerr << "Error guardando archivo: " << filename << "\n";
            return false;
        }
        out << magic << "\n" << width << " " << height << "\n" << maxColor << "\n";
        for (size_t i = 0; i < pixels.size(); i++) {
            out << pixels[i] << "\n";
        }
        return true;
    }
};

class Filter {
public:
    virtual void aplicar(const Image& input, Image& output) = 0;
    virtual ~Filter() {}
};

class ConvolutionFilter : public Filter {
protected:
    vector<vector<float>> kernel;
public:
    ConvolutionFilter(const vector<vector<float>>& k) : kernel(k) {}
    void aplicar(const Image& input, Image& output) {
        output = input;
        int channels = (input.magic == "P3") ? 3 : 1;
        int kw = kernel[0].size();
        int half = kw / 2;

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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Uso: " << argv[0] << " input.ppm\n";
        return 1;
    }

    Image img;
    if (!img.load(argv[1])) return 1;

    Image resultBlur, resultLaplace, resultSharpen;

    auto totalStart = chrono::high_resolution_clock::now();

    // Se ejecutan los 3 filtros en paralelo con OpenMP sections
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            auto start = chrono::high_resolution_clock::now();
            BlurFilter blur;
            blur.aplicar(img, resultBlur);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = end - start;
            cout << "Tiempo Blur: " << elapsed.count() << " s\n";
            resultBlur.save("out_blur.ppm");
        }
        #pragma omp section
        {
            auto start = chrono::high_resolution_clock::now();
            LaplaceFilter laplace;
            laplace.aplicar(img, resultLaplace);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = end - start;
            cout << "Tiempo Laplace: " << elapsed.count() << " s\n";
            resultLaplace.save("out_laplace.ppm");
        }
        #pragma omp section
        {
            auto start = chrono::high_resolution_clock::now();
            SharpenFilter sharp;
            sharp.aplicar(img, resultSharpen);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = end - start;
            cout << "Tiempo Sharpen: " << elapsed.count() << " s\n";
            resultSharpen.save("out_sharpen.ppm");
        }
    }

    auto totalEnd = chrono::high_resolution_clock::now();
    chrono::duration<double> totalElapsed = totalEnd - totalStart;
    cout << "Tiempo total de ejecución: " << totalElapsed.count() << " s\n";

    return 0;
}
