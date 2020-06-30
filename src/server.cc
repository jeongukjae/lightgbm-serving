#include "server.hh"

#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace lgbm_serving {

std::string getServerStat(std::map<std::string, lgbm_serving::Model*> models) {
  rapidjson::Document document;
  document.AddMember("num_models", (int)models.size(), document.GetAllocator());

  rapidjson::StringBuffer buffer;
  buffer.Clear();

  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  document.Accept(writer);

  return std::string(buffer.GetString());
}

std::string serializeModelOutput(int nrows, int nClasses, double* outResult) {
  rapidjson::Document document;
  document.SetArray();

  for (size_t i = 0; i < nrows; i++) {
    if (nClasses == 1) {
      document.PushBack(outResult[i], document.GetAllocator());
    } else {
      rapidjson::Value value(rapidjson::kArrayType);
      for (size_t j = 0; j < nClasses; j++)
        value.PushBack(outResult[i * nClasses + j], document.GetAllocator());
      document.PushBack(value, document.GetAllocator());
    }
  }

  rapidjson::StringBuffer buffer;
  buffer.Clear();
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  document.Accept(writer);
  return std::string(buffer.GetString());
}

std::pair<size_t, std::vector<float*>> parse2DFloatArray(const std::string& payload) {
  std::vector<float*> features;
  size_t ncol;

  rapidjson::Document document;
  document.Parse(payload.c_str());

  if (!document.IsArray())
    throw std::runtime_error("Cannot parse");

  auto array = document.GetArray();
  features.reserve(array.Size());

  for (const auto& value : array) {
    if (!value.IsArray() || (features.size() != 0 && value.GetArray().Size() != ncol)) {
      for (const auto* feat : features)
        delete[] feat;
      features.clear();

      throw std::runtime_error("Cannot parse");
    }

    if (features.size() == 0)
      ncol = value.GetArray().Size();

    auto array = value.GetArray();
    float* feature = new float[ncol];
    for (size_t i = 0; i < ncol; i++)
      feature[i] = array[i].GetFloat();

    features.push_back(feature);
  }

  return std::make_pair(ncol, features);
}

}  // namespace lgbm_serving
