# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for rccl ops. See also the cc test for rccl_communicator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib import rccl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class AllReduceTest(test.TestCase):

  def testAllReduce(self):
    if not test.is_gpu_available():
      return  # Test requires access to a GPU

    for dtype in [np.float32, np.int32, np.int64, np.float64]:
      # Create session inside outer loop to test use of
      # same communicator across multiple sessions.
      with self.test_session(use_gpu=True) as sess:
        self._testSingleAllReduce(sess, dtype, rccl.all_sum, lambda x, y: x + y)
#        self._testSingleAllReduce(sess, dtype, rccl.all_prod,
#                                  lambda x, y: x * y)
#        self._testSingleAllReduce(sess, dtype, rccl.all_min, np.minimum)
#        self._testSingleAllReduce(sess, dtype, rccl.all_max, np.maximum)

  def _testSingleAllReduce(self, sess, np_type, rccl_fn, numpy_accumulation_fn):
    for devices in [['/gpu:0', '/gpu:1']]:
      shape = (3, 4)
      np_ans = None
      tensors = []
      for d in devices:
        with ops.device(d):
          t = ((np.random.random_sample(shape) - .5) * 1024).astype(np_type)
          if np_ans is None:
            np_ans = t
          else:
            np_ans = numpy_accumulation_fn(np_ans, t)
          tensors.append(array_ops.identity(t))

      all_reduce_tensors = rccl_fn(tensors)

      # Test shape inference.
      for r in all_reduce_tensors:
        self.assertEqual(shape, r.get_shape())

      # Test execution and results.
      rccl_results = sess.run(all_reduce_tensors)
      for r in rccl_results:
        self.assertAllClose(r, np_ans)

  def testErrors(self):
    with self.assertRaisesRegexp(ValueError, 'Device assignment required'):
      rccl.all_sum([array_ops.identity(np.random.random_sample((3, 4)))])
    with self.assertRaisesRegexp(ValueError, 'Must pass >0 tensors'):
      rccl.all_sum([])


if __name__ == '__main__':
  test.main()
