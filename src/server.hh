#ifndef __LGBM_SERVING_SERVER_HH__
#define __LGBM_SERVING_SERVER_HH__

#include <map>
#include <string>
#include <vector>
#include "model.hh"
#include "rapidjson/document.h"

namespace lgbm_serving {

std::string getServerStat(std::map<std::string, lgbm_serving::Model*> models);
std::string serializeModelOutput(int nrows, int nClasses, double* outResult);
std::pair<size_t, std::vector<float*>> parse2DFloatArray(const std::string& payload);

}  // namespace lgbm_serving

#endif  // __LGBM_SERVING_SERVER_HH__
