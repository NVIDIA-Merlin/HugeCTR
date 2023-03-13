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

#include <NvInfer.h>

#include <hps_trt/hps_plugin/hps_plugin.hpp>
#include <hps_trt/hps_plugin/trt_plugin_utils.hpp>
#include <utility>

using namespace nvinfer1;
using nvinfer1::plugin::HpsPlugin;
using nvinfer1::plugin::HpsPluginCreator;

static const char* HPS_PLUGIN_VERSION{"1"};
static const char* HPS_PLUGIN_NAME{"HPS_TRT"};
PluginFieldCollection HpsPluginCreator::mFC{};
std::vector<PluginField> HpsPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(HpsPluginCreator);

HpsPlugin::HpsPlugin(std::string name, std::string ps_config_file, std::string model_name,
                     int32_t table_id, int32_t emb_vec_size)
    : mLayerName(std::move(name)),
      ps_config_file(std::move(ps_config_file)),
      model_name(std::move(model_name)),
      table_id(table_id),
      emb_vec_size(emb_vec_size) {}

HpsPlugin::HpsPlugin(std::string name, const void* data, size_t length)
    : mLayerName(std::move(name)) {
  // Deserialize in the same order as serialization
  const char *d = static_cast<const char*>(data), *a = d;

  int32_t ps_config_file_str_size;
  int32_t model_name_str_size;
  ps_config_file_str_size = read<int32_t>(d);
  ps_config_file = read_string(d, ps_config_file_str_size);
  model_name_str_size = read<int32_t>(d);
  model_name = read_string(d, model_name_str_size);
  table_id = read<int32_t>(d);
  emb_vec_size = read<int32_t>(d);
  mKeyType = read<DataType>(d);
  mVecType = read<DataType>(d);
  mInputVolume = read<size_t>(d);
  mOutputVolume = read<size_t>(d);
  HCTR_CHECK_HINT(d == (a + length), "The size for reading serialized data is not correct");
}

int HpsPlugin::initialize() noexcept { return 0; }

void HpsPlugin::terminate() noexcept {}

void HpsPlugin::destroy() noexcept { delete this; }

int HpsPlugin::getNbOutputs() const noexcept { return 1; }

DimsExprs HpsPlugin::getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs,
                                         int32_t nbInputs, IExprBuilder& exprBuilder) noexcept {
  try {
    HCTR_CHECK_HINT(nbInputs == 1, "The number of inputs should be 1");
    HCTR_CHECK_HINT(inputs[0].nbDims == 2, "The dimensions of inputs[0] should be 2");
    DimsExprs ret;
    ret.nbDims = 3;
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = inputs[0].d[1];
    ret.d[2] = exprBuilder.constant(emb_vec_size);
    return ret;
  } catch (const std::exception& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
  }
  return DimsExprs{};
}

DataType HpsPlugin::getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes,
                                      int32_t nbInputs) const noexcept {
  return DataType::kFLOAT;
}

size_t HpsPlugin::getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs,
                                   PluginTensorDesc const* outputs,
                                   int32_t nbOutputs) const noexcept {
  return 0;
}

size_t HpsPlugin::getSerializationSize() const noexcept {
  return 4 * sizeof(int32_t) + 2 * sizeof(DataType) + 2 * sizeof(size_t) + ps_config_file.size() +
         model_name.size();
}

const char* HpsPlugin::getPluginType() const noexcept { return HPS_PLUGIN_NAME; }

const char* HpsPlugin::getPluginVersion() const noexcept { return HPS_PLUGIN_VERSION; }

bool HpsPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut,
                                          int32_t nbInputs, int32_t nbOutputs) noexcept {
  if (pos == 0) {
    PluginTensorDesc const& input = inOut[0];
    return (input.type == mKeyType) && (input.format == TensorFormat::kLINEAR);
  }
  if (pos == 1) {
    const PluginTensorDesc& output = inOut[1];
    return (output.type == mVecType) && (output.format == TensorFormat::kLINEAR);
  }
  return false;
}

void HpsPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInput,
                                const DynamicPluginTensorDesc* out, int32_t nbOutput) noexcept {}

int32_t HpsPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                           void const* const* inputs, void* const* outputs, void* workspace,
                           cudaStream_t stream) noexcept {
  try {
    auto num_elements = inputDesc->dims.d[0];
    for (size_t i{1}; i < inputDesc->dims.nbDims; i++) {
      num_elements *= inputDesc->dims.d[i];
    }
    int32_t device_id;
    HCTR_LIB_THROW(cudaGetDevice(&device_id));
    bool i64_input_key = !(inputDesc->type == DataType::kINT32);
    Facade::instance()->forward(model_name.c_str(), table_id, device_id, num_elements, emb_vec_size,
                                inputs[0], outputs[0], i64_input_key, stream);
    return 0;
  } catch (const std::exception& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
  }
  return -1;
}

void HpsPlugin::serialize(void* buffer) const noexcept {
  int32_t ps_config_file_str_size = ps_config_file.size();
  int32_t model_name_str_size = model_name.size();

  char *d = static_cast<char*>(buffer), *a = d;

  // Serialize plugin data
  write(d, ps_config_file_str_size);
  write_string(d, ps_config_file);
  write(d, model_name_str_size);
  write_string(d, model_name);
  write(d, table_id);
  write(d, emb_vec_size);
  write(d, mKeyType);
  write(d, mVecType);
  write(d, mInputVolume);
  write(d, mOutputVolume);

  HCTR_CHECK_HINT(d == a + getSerializationSize(), "The serialization size does not match");
}

IPluginV2DynamicExt* HpsPlugin::clone() const noexcept {
  try {
    HpsPlugin* ret = new HpsPlugin(mLayerName, ps_config_file, model_name, table_id, emb_vec_size);
    ret->mInputVolume = mInputVolume;
    ret->mOutputVolume = mOutputVolume;
    ret->setPluginNamespace(mNamespace.c_str());
    return ret;
  } catch (const std::exception& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
  }
  return nullptr;
}

HpsPluginCreator::HpsPluginCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(PluginField("ps_config_file", nullptr, PluginFieldType::kCHAR, 1));
  mPluginAttributes.emplace_back(PluginField("model_name", nullptr, PluginFieldType::kCHAR, 1));
  mPluginAttributes.emplace_back(PluginField("table_id", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("emb_vec_size", nullptr, PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* HpsPluginCreator::getPluginName() const noexcept { return HPS_PLUGIN_NAME; }

const char* HpsPluginCreator::getPluginVersion() const noexcept { return HPS_PLUGIN_VERSION; }

const PluginFieldCollection* HpsPluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2DynamicExt* HpsPluginCreator::createPlugin(const char* name,
                                                    const PluginFieldCollection* fc) noexcept {
  try {
    int32_t table_id{0}, emb_vec_size{0};
    std::string model_name, ps_config_file;
    const PluginField* fields = fc->fields;

    validateRequiredAttributesExist({"ps_config_file", "model_name", "table_id", "emb_vec_size"},
                                    fc);
    HCTR_CHECK_HINT(fc->nbFields == 4, "The number of fields for HPS plugin should 4");

    for (int32_t i = 0; i < fc->nbFields; i++) {
      if (strcmp(fields[i].name, "ps_config_file") == 0) {
        HCTR_CHECK_HINT(fields[i].type == PluginFieldType::kCHAR, "ps_config_file should be CHAR");
        ps_config_file = static_cast<const char*>(fields[i].data);
      } else if (strcmp(fields[i].name, "model_name") == 0) {
        HCTR_CHECK_HINT(fields[i].type == PluginFieldType::kCHAR, "model_name should be CHAR");
        model_name = static_cast<const char*>(fields[i].data);
      }
      if (strcmp(fields[i].name, "table_id") == 0) {
        HCTR_CHECK_HINT(fields[i].type == PluginFieldType::kINT32, "emb_vec_size should be INT32");
        table_id = *(static_cast<const int32_t*>(fields[i].data));
      } else if (strcmp(fields[i].name, "emb_vec_size") == 0) {
        HCTR_CHECK_HINT(fields[i].type == PluginFieldType::kINT32, "emb_vec_size should be INT32");
        emb_vec_size = *(static_cast<const int32_t*>(fields[i].data));
      }
    }
    Facade::instance()->init(ps_config_file.c_str(), pluginType_t::TENSORRT);
    return new HpsPlugin(name, ps_config_file, model_name, table_id, emb_vec_size);
  } catch (const std::exception& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
  }
  return nullptr;
}

IPluginV2DynamicExt* HpsPluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                         size_t serialLength) noexcept {
  try {
    // This object will be deleted when the network is destroyed, which will
    // call HpsPlugin::destroy()
    const char* d = static_cast<const char*>(serialData);
    int32_t ps_config_file_str_size;
    std::string ps_config_file;
    ps_config_file_str_size = read<int32_t>(d);
    ps_config_file = read_string(d, ps_config_file_str_size);
    Facade::instance()->init(ps_config_file.c_str(), pluginType_t::TENSORRT);
    return new HpsPlugin(name, serialData, serialLength);
  } catch (const std::exception& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
  }
  return nullptr;
}
