#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t read_cycles(void) {
  uint64_t cycles;
  asm volatile ("rdcycle %0" : "=r" (cycles));
  return cycles;
}

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} memref2d_i32;

extern memref2d_i32 forward(int8_t *a_allocated, int8_t *a_aligned, int64_t a_offset,
                            int64_t a_size0, int64_t a_size1,
                            int64_t a_stride0, int64_t a_stride1,
                            int8_t *b_allocated, int8_t *b_aligned, int64_t b_offset,
                            int64_t b_size0, int64_t b_size1,
                            int64_t b_stride0, int64_t b_stride1);

int main(void) {
  int8_t a[8 * 8] = {
      21, 22, 23, 24, 25, 26, 27, 28,
      29, 30, 31, 32, 33, 34, 35, 36,
      37, 38, 39, 40, 41, 42, 43, 44,
      45, 46, 47, 48, 49, 50, 51, 52,
      53, 54, 55, 56, 57, 58, 59, 60,
      61, 62, 63, 64, 65, 66, 67, 68,
      69, 70, 71, 72, 73, 74, 75, 76,
      77, 78, 79, 80, 81, 82, 83, 84,
  };

  int8_t b[8 * 8] = {
      84, 83, 82, 81, 80, 79, 78, 77,
      76, 75, 74, 73, 72, 71, 70, 69,
      68, 67, 66, 65, 64, 63, 62, 61,
      60, 59, 58, 57, 56, 55, 54, 53,
      52, 51, 50, 49, 48, 47, 46, 45,
      44, 43, 42, 41, 40, 39, 38, 37,
      36, 35, 34, 33, 32, 31, 30, 29,
      28, 27, 26, 25, 24, 23, 22, 21,
  };

  uint64_t start_cycles = read_cycles();
  memref2d_i32 result = forward(a, a, 0, 8, 8, 8, 1, b, b, 0, 8, 8, 8, 1);
  uint64_t end_cycles = read_cycles();

  for (int row = 0; row < 8; ++row) {
    for (int col = 0; col < 8; ++col) {
      int64_t index = result.offset + row * result.strides[0] + col * result.strides[1];
      printf("%d ", result.aligned[index]);
    }
    printf("\n");
  }
  printf("cycles: %llu\n", (unsigned long long)(end_cycles - start_cycles));

  free(result.allocated);
  return 0;
}
