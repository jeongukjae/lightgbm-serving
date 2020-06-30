#include "server.hh"

#include <tuple>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lgbm_serving;

class server_serialize_test : public ::testing::TestWithParam<std::tuple<std::vector<double>, int, int, std::string>> {};

TEST_P(server_serialize_test, test_serialize_double_array) {
  auto param = GetParam();

  auto result = serializeModelOutput(std::get<1>(param), std::get<2>(param), std::get<0>(param).data());
  ASSERT_EQ(result, std::get<3>(param));
}

INSTANTIATE_TEST_SUITE_P(
    SerializeTest,
    server_serialize_test,
    ::testing::Values(
        std::make_tuple<std::vector<double>, int, int, std::string>({1, 2, 3, 4}, 4, 1, "[1.0,2.0,3.0,4.0]"),
        std::make_tuple<std::vector<double>, int, int, std::string>({1, 2, 3, 4}, 1, 4, "[[1.0,2.0,3.0,4.0]]"),
        std::make_tuple<std::vector<double>, int, int, std::string>({1, 2, 3, 4, 5, 6, 7, 8}, 2, 4, "[[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0]]"),
        std::make_tuple<std::vector<double>, int, int, std::string>({1, 2, 3, 4, 5, 6, 7, 8}, 4, 2, "[[1.0,2.0],[3.0,4.0],[5.0,6.0],[7.0,8.0]]")));

TEST(server, test_deserialize_float_array) {
  auto k = parse2DFloatArray("[[1,2,3], [1.2, 3,4]]");
}

class server_deserialize_test : public ::testing::TestWithParam<std::tuple<std::string, size_t, std::vector<std::vector<float>>>> {};

TEST_P(server_deserialize_test, test_deserialize_double_array) {
  auto param = GetParam();

  auto result = parse2DFloatArray(std::get<0>(param));
  ASSERT_EQ(result.first, std::get<1>(param));
  for (size_t i = 0; i < result.second.size(); i++)
    ASSERT_THAT(std::vector<float>(result.second[i], result.second[i] + result.first), ::testing::ElementsAreArray(std::get<2>(param)[i]));

  for (const auto item : result.second)
    delete[] item;
}

INSTANTIATE_TEST_SUITE_P(
    DeserializeTest,
    server_deserialize_test,
    ::testing::Values(
        std::make_tuple<std::string, size_t, std::vector<std::vector<float>>>("[[1.0,2,3,4], [4,5 ,6, 7]]", 4, {{1, 2, 3, 4}, {4, 5, 6, 7}}),
        std::make_tuple<std::string, size_t, std::vector<std::vector<float>>>("[[1,2,3,4,5,6,7]]", 7, {{1, 2, 3, 4, 5, 6, 7}}),
        std::make_tuple<std::string, size_t, std::vector<std::vector<float>>>("[[1,2,3],[4,5,6]]", 3, {{1, 2, 3}, {4, 5, 6}})));
