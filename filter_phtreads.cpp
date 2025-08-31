#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <pthread.h>
#include <algorithm>
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
        if (!in.is_open()) return false;
        in >> magic >> width >> height >> maxColor;
        int pixelCount = (magic == "P3") ? width * height * 3 : width * height;
        pixels.resize(pixelCount);
        for (int i = 0; i < pixelCount; i++) in >> pixels[i];
        return true;
    }
    bool save(const string& filename) {
        ofstream out(filename.c_str());
        if (!out.is_open()) return false;
        out << magic << "\n" << width << " " << height << "\n" << maxColor << "\n";
        for (size_t i = 0; i < pixels.size(); i++) out << pixels[i] << "\n";
        return true;
    }
};

class Filter {
public:
    virtual void ApliRegion(const Image& input, Image& output,
                            int startX, int startY, int endX, int endY) = 0;
    virtual ~Filter() {}
};

class ConvolutionFilter : public Filter {
protected:
    vector<vector<float> > kernel;
public:
    ConvolutionFilter(const vector<vector<float> >& k) : kernel(k) {}
    void ApliRegion(const Image& input, Image& output,
                    int startX, int startY, int endX, int endY) {
        int channels = (input.magic == "P3") ? 3 : 1;
        int kw = kernel[0].size();
        int kh = kernel.size();
        int half = kw / 2;
        for (int y = startY; y < endY; y++) {
            for (int x = startX; x < endX; x++) {
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

struct ThreadInfo {
    const Image* input;
    Image* output;
    Filter* filter;
    int startX, startY, endX, endY;
};

void* Func(void* arg) {
    ThreadInfo* data = (ThreadInfo*)arg;
    data->filter->ApliRegion(*data->input, *data->output,
                             data->startX, data->startY, data->endX, data->endY);
    return NULL;
}

int main(int argc, char* argv[]) {
    auto start = chrono::high_resolution_clock::now(); // Inicia el cronómetro
    Image img, result;
    if (!img.load(argv[1])) return 1;
    result = img;
    string filterArg = argv[3];
    Filter* filter = NULL;
    if (filterArg == "blur") filter = new BlurFilter();
    else if (filterArg == "laplace") filter = new LaplaceFilter();
    else if (filterArg == "sharpen") filter = new SharpenFilter();
    else {
        cerr << "Filtro no creado: " << filterArg << "\n";
        return 1;
    }
    int midX = img.width / 2;
    int midY = img.height / 2;
    pthread_t threads[4];
    ThreadInfo data[4] = {
        {&img, &result, filter, 0,    0,    midX, midY},
        {&img, &result, filter, midX, 0,    img.width, midY},
        {&img, &result, filter, 0,    midY, midX, img.height},
        {&img, &result, filter, midX, midY, img.width, img.height}
    };
    for (int i = 0; i < 4; i++) pthread_create(&threads[i], NULL, Func, &data[i]);
    for (int i = 0; i < 4; i++) pthread_join(threads[i], NULL);
    result.save(argv[2]);
    auto end = chrono::high_resolution_clock::now(); // Detiene el cronómetro
    chrono::duration<double> elapsed = end - start;
    cout << "Tiempo de ejecución: " << elapsed.count() << " segundos" << endl;
    delete filter;
    return 0;
}
