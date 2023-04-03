#include <vector>
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <iostream>

void jacobi_omp(std::vector<double>& u, const std::vector<double> f, const double n) {

    omp_set_num_threads(omp_get_max_threads());
    double h = 1/(n+1);

    double res = 0;
    #pragma parallel for reduction(+:res) collapse(2)
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
        #pragma parallel for collapse(2)
        for (int i=1; i < n + 1; i++) {
            for (int j=1; j < n + 1; j++) {
                int index = i*(n+1) + j + 1;
                unew[index] = 0.25*(h*h*f[index] + u[index - 1] + u[index+1] + u[index - (n+1)] + u[index + (n+1)]);
            }
        }
        
        u.swap(unew);

        res = 0;
        #pragma parallel for reduction(+:res) collapse(2)
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
    int N = 1000;

    std::vector<double> u ((N+2)*(N+2));
    std::vector<double> f ((N+2)*(N+2));
    std::fill(u.begin(),u.end(),0);
    std::fill(f.begin(),f.end(),1);

    jacobi_omp(u, f, N);

    return 0;
}