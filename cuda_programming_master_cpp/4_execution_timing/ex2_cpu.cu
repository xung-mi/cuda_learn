#include <iostream>
#include <cstdlib>  
#include <ctime>    

int main() {
    const int SIZE = 4194304;
    int* arr1 = new int[SIZE];
    int* arr2 = new int[SIZE];
    int* arr3 = new int[SIZE];
    int* sum_arr = new int[SIZE];

    srand(time(0));  

    for (int i = 0; i < SIZE; i++) {
        arr1[i] = rand() % 100;
        arr2[i] = rand() % 1000;
        arr3[i] = rand() % 10000;
    }

    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    for (int i = 0; i < SIZE; i++){
        sum_arr[i] = arr1[i] + arr2[i] + arr3[i];
    }
    cpu_end = clock();
    int duration = cpu_end-cpu_start;
    printf("Execution time: %4.6f \n", (double)((double)duration/CLOCKS_PER_SEC));
    return 0;
}

