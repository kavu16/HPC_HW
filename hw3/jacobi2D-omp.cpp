#include <vector>
#include <cmath>
#include <cstdio>
#include <iostream>
#if defined(_OPENMP)
#include <omp.h>
#endif

void jacobi_omp(std::vector<double>& u, const std::vector<double> f, int n) {

    // omp_set_num_threads(omp_get_max_threads());
    
    double h = 1/((double) (n+1));
    printf("h = %f\n", h);

    double res = 0;
    #pragma omp parallel for reduction(+:res) collapse(2)
    for (int i = 1; i < n + 1; i++) {
        for (int j = 1; j < n + 1; j++) {
            int index = i*(n+1) + j + 1;
            double curr = (1/(h*h))*(2*u[index] - u[index - 1] - u[index + 1]) - f[index];
            res += curr*curr;
        }
    }
    res = sqrt(res);
    printf("Initial res = %f\n", res);

    int iter = 0;
    while(iter < 5000) {
        std::vector<double> unew (u);
        #pragma omp parallel for collapse(2)
        for (int i=1; i < n + 1; i++) {
            for (int j=1; j < n + 1; j++) {
                int index = i*(n+1) + j + 1;
                unew[index] = 0.25*(h*h*f[index] + u[index - 1] + u[index+1] + u[index - (n+1)] + u[index + (n+1)]);
            }
        }
        
        u.swap(unew);

        res = 0;
        #pragma omp parallel for reduction(+:res) collapse(2)
        for (int i = 1; i < n + 1; i++) {
            for (int j = 1; j < n + 1; j++) {
                int index = i*(n+1) + j + 1;
                res += pow((1/(h*h))*(2*u[index] - u[index - 1] - u[index + 1]) - f[index],2);
            }
        }
        res = sqrt(res);
        if (iter%500 == 0) printf("Residual = %f\n", res);
        ++iter;
    }
}

void jacobi(std::vector<double>& u, const std::vector<double> f, const double n) {

    // omp_set_num_threads(omp_get_max_threads());
    double h = 1/(n+1);

    double res = 0;
    for (int i = 1; i < n + 1; i++) {
        for (int j = 1; j < n + 1; j++) {
            int index = i*(n+1) + j + 1;
            double curr = (1/(h*h))*(2*u[index] - u[index - 1] - u[index + 1]) - f[index];
            res += curr*curr;
        }
    }
    res = sqrt(res);
    printf("Initial res = %f\n", res);

    int iter = 0;
    while(iter < 5000) {
        std::vector<double> unew (u);
        for (int i=1; i < n + 1; i++) {
            for (int j=1; j < n + 1; j++) {
                int index = i*(n+1) + j + 1;
                unew[index] = 0.25*(h*h*f[index] + u[index - 1] + u[index+1] + u[index - (n+1)] + u[index + (n+1)]);
            }
        }
        
        u.swap(unew);

        res = 0;
        for (int i = 1; i < n + 1; i++) {
            for (int j = 1; j < n + 1; j++) {
                int index = i*(n+1) + j + 1;
                res += pow((1/(h*h))*(2*u[index] - u[index - 1] - u[index + 1]) - f[index],2);
            }
        }
        res = sqrt(res);
        if (iter%500 == 0) printf("Residual = %f\n", res);
        ++iter;
    }
}

int main() {
    #if defined(_OPENMP)
        for (int N = 10; N <= 1000; N *= 10) {
            std::vector<double> u ((N+2)*(N+2));
            std::vector<double> f ((N+2)*(N+2));
            std::fill(u.begin(),u.end(),0);
            std::fill(f.begin(),f.end(),1);
            
            double tt = omp_get_wtime();
            jacobi_omp(u,f,N);
            printf("jacobi timing, 5000 iterations, %d points  = %fs\n", N, omp_get_wtime() - tt);
        }
    #else 
        for (int N = 10; N <= 1000; N*=10) {
            std::vector<double> u ((N+2)*(N+2));
            std::vector<double> f ((N+2)*(N+2));
            std::fill(u.begin(),u.end(),0);
            std::fill(f.begin(),f.end(),1);
            jacobi(u, f, N);
        }
    #endif
    

    return 0;
}