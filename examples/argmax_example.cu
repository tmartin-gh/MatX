#include <stdio.h>
#include "matx.h"

void test_argmax(void)
{
  using cub_index_t = int;

  int num_segments = 4;
  auto t_in = matx::make_tensor<int>({num_segments, 2, 5});
  auto t_value_out = matx::make_tensor<int>({num_segments});
  auto t_index_out = matx::make_tensor<int>({num_segments});

  t_in.SetVals(
  {
    {{1, 2, 3, 4, 5},  {6, 7, 8, 9, 10}},
    {{1, 3, 5, 7, 10}, {2, 4, 6, 8, 9}},
    {{2, 4, 5, 1, 1},  {1, 1, 1, 1, 1}},
    {{INT_MIN, INT_MIN, INT_MIN, INT_MIN, INT_MIN},  {INT_MIN, INT_MIN, INT_MIN, INT_MIN, INT_MIN}},
  });
  cub_index_t golden_index[] {9, 4, 2, 0};
  int golden_value[] {10, 10, 5, INT_MIN};


  (matx::mtie(t_value_out, t_index_out) = argmax(t_in, {1, 2})).run();
  cudaDeviceSynchronize();
  printf("MatX\n");
  for (int k=0; k<num_segments; k++)
  {
    printf("  [%d] %d => %d",k, t_index_out(k), t_value_out(k));
    if (golden_index[k] != t_index_out(k))
    {
      printf(" Index Mismatch (%d)",golden_index[k]);
    }
    if (golden_value[k] != t_value_out(k))
    {
      printf(" Value Mismatch (%d)",golden_value[k]);
    }
    printf("\n");
  }

  //matx::tensor_t<int, 0> t_max_value_out{{}};
  //matx::tensor_t<int, 0> t_max_index_out{{}};
  //(matx::mtie(t_max_value_out, t_max_index_out) = argmax(t_in)).run();
  //cudaDeviceSynchronize();
  //matx::print(t_max_value_out);
  //matx::print(t_max_index_out);
}

int main(void)
{
  MATX_ENTER_HANDLER();

  test_argmax();

  CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
  return 0;
}
