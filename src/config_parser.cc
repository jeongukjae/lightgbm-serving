#include "config_parser.hh"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "rapidjson/document.h"

namespace lgbm_serving {

ConfigParser::ConfigParser() {}
ConfigParser::~ConfigParser() {
  clear();
}

void ConfigParser::clear() {
  configs.clear();
}

void ConfigParser::parseModelConfig(const std::string path) {
  std::ifstream ifs(path);

  if (!ifs.good())
    throw std::runtime_error("Cannot open file or directory: " + path);

  std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  rapidjson::Document document;
  document.Parse(str.c_str());

  if (!document.IsObject())
    throw std::runtime_error("Root object is not an object");

  if (!document.HasMember("config") || !document["config"].IsArray())
    throw std::runtime_error("There is no config");

  for (auto& value : document["config"].GetArray()) {
    if (!value.HasMember("name") || !value.HasMember("path"))
      throw std::runtime_error("Cannot get attribute `name` or `path`");

    if (!value["name"].IsString() || !value["path"].IsString())
      throw std::runtime_error("Name or path is not a string.");

    ModelConfig config{value["name"].GetString(), value["path"].GetString()};
    configs.push_back(config);
  }
}

size_t ConfigParser::getLength() const {
  return configs.size();
}

ModelConfig& ConfigParser::get(size_t index) {
  return configs[index];
}

std::vector<ModelConfig>::const_iterator ConfigParser::begin() const {
  return configs.begin();
}

std::vector<ModelConfig>::const_iterator ConfigParser::end() const {
  return configs.end();
}

}  // namespace lgbm_serving
