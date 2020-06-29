#ifndef __LGBM_SERVING_MODEL_HH__
#define __LGBM_SERVING_MODEL_HH__

#include <string>

#include "LightGBM/c_api.h"
#include "lightgbm-serving/config_parser.hh"

namespace lgbm_serving {

class Model {
 public:
  Model();
  ~Model();

  void clear();

  void setConfig(const ModelConfig* config);
  const ModelConfig* getConfig() const;
  void load(const std::string filename);
  BoosterHandle& getHandle();

 private:
  BoosterHandle handle;
  const ModelConfig* config;
};

}  // namespace lgbm_serving

#endif  // __LGBM_SERVING_MODEL_HH__
