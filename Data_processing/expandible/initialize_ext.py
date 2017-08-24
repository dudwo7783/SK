# Copyright 2016 Google Inc. All Rights Reserved.
#
# Made by Kim Youngjae
# 17-08-16
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to train Inception using multiple GPUs with synchronous updates.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from os.path import join, exists
from os import mkdir,remove
import time as tm

import data_processing as dp

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default=r'.\raw_data',
      help='Raw data path.\ndefault : ./raw_data'
  )
  parser.add_argument(
      '--save_dir',
      type=str,
      default=r'.\RTDB_dir',
      help='RTDB path to save RTDB.csv\ndefault : ./RTDB_dir'
  )
  parser.add_argument(
      '--time_unit',
      type=str,
      default='1min',
      help='Unit to expand time series(1min, S, H, D)\ndefault : 1min'
  )
  parser.add_argument(
      '--start_date',
      type=str,
      default='2017-01-01',
      help='Start date (2017-01-01)'
  )
  parser.add_argument(
      '--end_date',
      type=str,
      default='2017-01-10',
      help='End date (2017-01-10)'
  )
  parser.add_argument(
      '--fill_method',
      type=str,
      default='pad_to_bfill',
      help='How to fill NA rows.'
  )
  parser.add_argument(
      '--split_data',
      type=str,
      default=['CRAC003.Y', 'ARPC014A.Y'],
      nargs='+',
      help='Divided data by month or year\nAdd multiple file using white space.'
  )
  FLAGS, unparsed = parser.parse_known_args()

  split_files = FLAGS.split_data
  data_dir = FLAGS.data_dir
  save_dir = FLAGS.save_dir
  time_unit = FLAGS.time_unit
  start_date = FLAGS.start_date
  end_date = FLAGS.end_date
  fill_method = FLAGS.fill_method
  split_data = FLAGS.split_data

  if not exists(save_dir):
    mkdir(save_dir)

  for file in split_files:
      if exists(join(data_dir, file + ".csv")):
          remove(join(data_dir, file + ".csv"))

  start_time = tm.time()

  L_files, L_tags, Y_files, Y_tags = dp.getFileList(data_dir, split_files)
  Ldf, Ydf, RTDB = dp.getDataFrame(L_files, L_tags, Y_files, Y_tags, time_unit, start_date, end_date, fill_method)
  RTDB_file_name = join(save_dir, "RTDB.csv")
  RTDB.to_csv(RTDB_file_name, index=False)

  print("--- %s seconds ---" % (tm.time() - start_time))