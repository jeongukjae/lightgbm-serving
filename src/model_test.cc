#include "model.hh"

#include "gtest/gtest.h"

TEST(model, test_loading_model) {
  lgbm_serving::Model model;
  model.load("../src/test_data/test-model");

  ASSERT_EQ(model.getNumFeatures(), 40);
  ASSERT_EQ(model.getNumClasses(), 5);
}
