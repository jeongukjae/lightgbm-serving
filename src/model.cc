#include "model.hh"

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

  LGBM_BoosterGetNumFeature(handle, &numFeatures);
}

BoosterHandle& Model::getHandle() {
  return handle;
}

int Model::getNumFeatures() const {
  return numFeatures;
}

}  // namespace lgbm_serving
