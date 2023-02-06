"""
 Copyright (c) 2023, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import tqdm
import sys
import struct
from os.path import getsize
from os.path import exists


def usage():
    print("Usage: python analyzer.py <path to directory>")
    print("Directory must contain 'key' and 'slot_id' files created using LocalizedSlotEmbedding.")
    quit()


if len(sys.argv) != 2:
    usage()

dir = sys.argv[1]
if not exists(dir + "/key") or not exists(dir + "/slot_id"):
    usage()

print("Running analysis on model files in directory: " + dir)

keysize = getsize(dir + "/key")
slotsize = getsize(dir + "/slot_id")

keyfile = open(dir + "/key", "rb")
slotfile = open(dir + "/slot_id", "rb")

slot_set = [set()]

readsize = 0
while readsize < slotsize:
    slotval = struct.unpack("q", slotfile.read(8))[0]
    keyval = struct.unpack("q", keyfile.read(8))[0]

    while len(slot_set) < slotval + 1:
        slot_set.append(set())

    slot_set[slotval].add(keyval)
    readsize += 8

print("Analysis complete. Total keys: " + str(int(keysize / 8)))
print("Number of slots: " + str(len(slot_set)))
print("Vocabulary size (unique keys) per slot:")
for x in range(len(slot_set)):
    print("Slot " + str(x) + ": " + str(len(slot_set[x])))
