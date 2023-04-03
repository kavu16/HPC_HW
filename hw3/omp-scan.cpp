#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <climits>
#include <vector>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = A[0];
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i];
  }
  printf("Final sum = %d\n", prefix_sum[n-1]);
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  int p = omp_get_max_threads();
  omp_set_num_threads(p);
  // Fill out parallel scan: One way to do this is array into p chunks
  // Do a scan in parallel on each chunk, then share/compute the offset
  // through a shared vector and update each chunk by adding the offset
  // in parallel
  if (n==0) return;
  prefix_sum[0] = A[0];
  
  #pragma omp parallel 
  {
  int t = omp_get_thread_num();
  #pragma omp for schedule(monotonic: static, n/p)
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i];
  }
  }

  std::vector<int> offsets;
  for (int i = 0; i < p; i++) {
    offsets.push_back(prefix_sum[(i+1)*(n/p)]);
  }

  #pragma omp parallel for schedule(monotonic: static) collapse(2)
  for (int i = 1; i < p; i++) {
    for (int j = 0; j < (n/p); j++) {
      for (int o = 0; o < i; o++) {
        prefix_sum[i*(n/p) + j] += offsets[o];
      }
    }
  }
  
  printf("Final sum = %d\n", prefix_sum[n-1]);
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N; i++) B1[i] = 0;

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  long err_sum = 0;
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
