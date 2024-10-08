////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#include "assert.h"
#include "matx.h"
#include "utilities.h"
#include "gtest/gtest.h"

using namespace matx;
using complex = cuda::std::complex<float>;

class RadarAmbiguityFunction : public ::testing::Test {
protected:
  void SetUp() override
  {

    pb = std::make_unique<detail::MatXPybind>();
    pb->InitAndRunTVGenerator<complex>("01_radar", "ambgfun", "run",
                                       {sig_size});

    pb->NumpyToTensorView(xv, "x");
  }

  void TearDown() override { pb.reset(); }

  index_t sig_size = 16;
  tensor_t<complex, 1> xv{{sig_size}};
  std::unique_ptr<detail::MatXPybind> pb;
};

//#ifdef CUPY_INSTALLED
//TEST_F(RadarAmbiguityFunction, Cut2D)
//#else
TEST_F(RadarAmbiguityFunction, DISABLED_Cut2D)
//#endif
{
  MATX_ENTER_HANDLER();

  tensor_t<float, 2> amf2dv(
      {2 * sig_size - 1,
       (index_t)pow(2, std::ceil(std::log2(2 * sig_size - 1)))});

  (amf2dv = ambgfun(xv, 1e3, AMBGFUN_CUT_TYPE_2D, 1.0)).run();
  MATX_TEST_ASSERT_COMPARE(pb, amf2dv, "amf_2d", 0.01);

  MATX_EXIT_HANDLER();
}

//#ifdef CUPY_INSTALLED
//TEST_F(RadarAmbiguityFunction, CutDelay)
//#else
TEST_F(RadarAmbiguityFunction, DISABLED_CutDelay)
//#endif
{
  MATX_ENTER_HANDLER();

  tensor_t<float, 2> amf_delay_v(
      {1, (index_t)pow(2, std::ceil(std::log2(2 * sig_size - 1)))});

  // example-begin ambgfun-test-1
  (amf_delay_v = ambgfun(xv, 1e3, AMBGFUN_CUT_TYPE_DELAY, 1.0)).run();
  // example-end ambgfun-test-1

  auto delay1d = slice<1>(amf_delay_v, {0, 0}, {matxDropDim, matxEnd});
  MATX_TEST_ASSERT_COMPARE(pb, delay1d, "amf_delay", 0.01);

  MATX_EXIT_HANDLER();
}

//#ifdef CUPY_INSTALLED
//TEST_F(RadarAmbiguityFunction, CutDoppler)
//#else
TEST_F(RadarAmbiguityFunction, DISABLED_CutDoppler)
//#endif
{
  MATX_ENTER_HANDLER();

  tensor_t<float, 2> amf_doppler_v({1, xv.Size(0) * 2 - 1});

  // example-begin ambgfun-test-2
  (amf_doppler_v = ambgfun(xv, 1e3, AMBGFUN_CUT_TYPE_DOPPLER, 1.0)).run();
  // example-end ambgfun-test-2

  auto doppler1d = slice<1>(amf_doppler_v, {0, 0}, {matxDropDim, matxEnd});
  MATX_TEST_ASSERT_COMPARE(pb, doppler1d, "amf_doppler", 0.01);

  MATX_EXIT_HANDLER();
}
