#include <iostream>
#include <cstdlib>
#include <chrono>
#include <pthread.h>

using namespace std;


const int n = 5000000;
int A[n];
double sum = 0.0;
int thread_count = 10;
pthread_mutex_t paralel_sum; 

int calcularArreglo(){
    for (int i = 0; i < n; i++){
        int randomNumber = (rand() % 1000);
        A[i] = randomNumber;
    }
}

void* CalcularSuma(void* rank){
    long my_rank = (long) rank;
    long long i;
    long long my_n = n/thread_count;
    long long my_first_i = my_n*my_rank;
    long long my_last_i = my_first_i + my_n;
    double local_sum = 0.0;

    for (i = my_first_i; i < my_last_i; i++){
        local_sum += A[i];
    }
    pthread_mutex_lock(&paralel_sum);
        sum += local_sum;
    pthread_mutex_unlock(&paralel_sum);
    return NULL;
}


int main(){
    auto start = chrono::high_resolution_clock::now(); // Inicia el cronómetro
    calcularArreglo();
    pthread_t threads[thread_count];
    pthread_mutex_init(&paralel_sum, NULL);

    for (long t=0; t < thread_count; t++){
        pthread_create(&threads[t], NULL, CalcularSuma, (void*) t);
    }

    for (int t = 0; t < thread_count; t++){
        pthread_join(threads[t], NULL);
    }
    pthread_mutex_destroy(&paralel_sum);
    cout << "Suma total con n " << n << " pthreads " << sum << endl;
    auto end = chrono::high_resolution_clock::now(); // Detiene el cronómetro
    chrono::duration<double> elapsed = end - start;
    cout << "Tiempo de ejecución: " << elapsed.count() << " segundos" << endl;
    return 0;
}