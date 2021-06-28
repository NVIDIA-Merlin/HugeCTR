"""
 Copyright (c) 2021, NVIDIA CORPORATION.
 
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sparse_operation_kit.core.embedding_variable import EmbeddingVariable
from tensorflow.python.distribute.values import MirroredVariable
from tensorflow.python.distribute.values import DistributedVariable

def split_embedding_variable_from_others(variables):
    """Used to split embedding variables from other variables.
    
    Args:
        variables: a list or tuple of Variables.

    Returns:
        A tuple of ('embedding_variables', 'other_variables') 
        where 'embedding_variables' is also a tuple, and 'other_variables'
        is tuple too.
    """
    if not (isinstance(variables, list) or isinstance(variables, tuple)):
        raise ValueError("Variables must be list or tuple. But got ", type(variables))

    embedding_variables = list()
    other_variables = list()

    for variable in variables:
        if (isinstance(variable, DistributedVariable) and 
            not isinstance(variable, MirroredVariable)):
            if isinstance(variable.values[0], EmbeddingVariable):
                embedding_variables.append(variable)
            else:
                other_variables.append(variable)
        else:
            other_variables.append(variable)

    return embedding_variables, other_variables
        
