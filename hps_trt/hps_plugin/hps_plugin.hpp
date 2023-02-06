/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstdlib>
#include <cstring>
#include <hps/plugin/facade.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace HierarchicalParameterServer;

namespace nvinfer1 {

namespace plugin {

class HpsPlugin : public IPluginV2DynamicExt {
 public:
  HpsPlugin(std::string plugin_layer_name, std::string ps_config_file, std::string model_name,
            int32_t table_id, int32_t emb_vec_size);

  HpsPlugin(std::string plugin_layer_name, const void* data, size_t length);

  ~HpsPlugin() override = default;

  HpsPlugin() = delete;

  int initialize() noexcept override;

  void terminate() noexcept override;

  void destroy() noexcept override;

  int getNbOutputs() const noexcept override;

  DimsExprs getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs,
                                IExprBuilder& exprBuilder) noexcept override;

  DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes,
                             int32_t nbInputs) const noexcept override;

  size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs,
                          PluginTensorDesc const* outputs,
                          int32_t nbOutputs) const noexcept override;

  size_t getSerializationSize() const noexcept override;

  const char* getPluginType() const noexcept override;

  const char* getPluginVersion() const noexcept override;

  bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override;

  void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInput,
                       const DynamicPluginTensorDesc* out, int32_t nbOutput) noexcept override;

  int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                  void const* const* inputs, void* const* outputs, void* workspace,
                  cudaStream_t stream) noexcept override;

  void serialize(void* buffer) const noexcept override;

  IPluginV2DynamicExt* clone() const noexcept override;

 protected:
  void setPluginNamespace(const char* libNamespace) noexcept override { mNamespace = libNamespace; }

  const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

  std::string mNamespace;

 private:
  std::string mLayerName;
  std::string ps_config_file;
  std::string model_name;
  int32_t table_id;
  int32_t emb_vec_size;
  DataType mKeyType{DataType::kINT32};
  DataType mVecType{DataType::kFLOAT};
  size_t mInputVolume{0};
  size_t mOutputVolume{0};
};

class HpsPluginCreator : public IPluginCreator {
 public:
  HpsPluginCreator();

  ~HpsPluginCreator() override = default;

  const char* getPluginName() const noexcept override;

  const char* getPluginVersion() const noexcept override;

  const PluginFieldCollection* getFieldNames() noexcept override;

  IPluginV2DynamicExt* createPlugin(const char* name,
                                    const PluginFieldCollection* fc) noexcept override;

  IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData,
                                         size_t serialLength) noexcept override;

  void setPluginNamespace(const char* libNamespace) noexcept override { mNamespace = libNamespace; }

  const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

 protected:
  std::string mNamespace;

 private:
  static PluginFieldCollection mFC;
  static std::vector<PluginField> mPluginAttributes;
};

}  // namespace plugin

}  // namespace nvinfer1
