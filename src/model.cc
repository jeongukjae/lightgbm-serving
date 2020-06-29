#include "lightgbm-serving/model.hh"

#include "LightGBM/c_api.h"

namespace lgbm_serving {

Model::Model() : handle(nullptr) {}

Model::~Model() {
  clear();
}

void Model::clear() {
  if (handle != nullptr) {
    LGBM_BoosterFree(handle);
  }
}

void Model::setConfig(const ModelConfig* config) {
  this->config = config;
}

const ModelConfig* Model::getConfig() const {
  return config;
}

void Model::load(const std::string filename) {
  int numIterations;
  if (LGBM_BoosterCreateFromModelfile(filename.c_str(), &numIterations, &handle) != 0) {
    throw std::runtime_error("cannot load file from " + filename);
  }
}

BoosterHandle& Model::getHandle() {
  return handle;
}

}  // namespace lgbm_serving
