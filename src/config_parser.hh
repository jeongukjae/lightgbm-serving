#ifndef __LGBM_SERVINGP_CONFIG_PARSER_HH__
#define __LGBM_SERVINGP_CONFIG_PARSER_HH__

#include <string>
#include <vector>

namespace lgbm_serving {

struct ModelConfig {
  std::string name;
  std::string path;
  size_t nClasses;
};

class ConfigParser {
 public:
  ConfigParser();
  ~ConfigParser();

  void clear();

  void parseModelConfig(const std::string path);
  void dumpModelConfig(const std::string path) const;

  size_t getLength() const;
  ModelConfig& get(size_t index);

  std::vector<ModelConfig>::const_iterator begin() const;
  std::vector<ModelConfig>::const_iterator end() const;

 private:
  std::vector<ModelConfig> configs;
};

}  // namespace lgbm_serving

#endif  // __LGBM_SERVINGP_CONFIG_PARSER_HH__
