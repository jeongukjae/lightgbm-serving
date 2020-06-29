#include "config_parser.hh"

#include "gtest/gtest.h"

using namespace lgbm_serving;

TEST(config_parser, parse_invalid_path_should_raise_error) {
  ConfigParser parser;

  ASSERT_THROW(parser.parseModelConfig("SOME_INVALID_PATH"), std::runtime_error);
}

TEST(config_parser, parse_invalid_structure) {
  ConfigParser parser;

  ASSERT_THROW(parser.parseModelConfig("../src/test_data/test_config_no_config.json"), std::runtime_error);
}

TEST(config_parser, parse_config) {
  ConfigParser parser;

  ASSERT_NO_THROW(parser.parseModelConfig("../src/test_data/test_config.json"));

  ASSERT_EQ(parser.getLength(), 2);
  ASSERT_EQ(parser.get(0).name, "TestModel1");
  ASSERT_EQ(parser.get(0).path, "/path/to/model1");
  ASSERT_EQ(parser.get(1).name, "TestModel2");
  ASSERT_EQ(parser.get(1).path, "/path/to/model2");
}
