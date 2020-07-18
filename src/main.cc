#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include "config_parser.hh"
#include "cxxopts.hpp"
#include "httplib.h"
#include "model.hh"
#include "server.hh"

cxxopts::ParseResult parseCLIArgs(int argc, char** argv);
std::string getServerStat(std::map<std::string, lgbm_serving::Model*> models);

int main(int argc, char** argv) {
  //
  // parse cli arguments
  auto args = parseCLIArgs(argc, argv);

  std::string configFilePath = args["config"].as<std::string>();
  if (configFilePath.empty()) {
    std::cerr << "Cannot parse config file." << std::endl;
    return 1;
  }

  //
  // parse model config
  lgbm_serving::ConfigParser parser;
  try {
    parser.parseModelConfig(configFilePath);
  } catch (std::runtime_error& error) {
    std::cerr << "Cannot parse config file: " << error.what() << std::endl;
    return 1;
  }

  //
  // load models
  size_t numModel = parser.getLength();
  if (numModel == 0) {
    std::cerr << "Config file contains nothing." << std::endl;
    return 1;
  }

  std::cout << "Found " << numModel << " configs." << std::endl;
  std::map<std::string, lgbm_serving::Model*> models;
  for (auto iterator = parser.begin(); iterator < parser.end(); iterator++) {
    lgbm_serving::Model* model = new lgbm_serving::Model;

    model->setConfig(&(*iterator));
    try {
      model->load(iterator->path);
    } catch (std::runtime_error& error) {
      std::cerr << "Cannot load model: " << error.what() << std::endl;
      return 1;
    }

    std::cout << "Loaded " << iterator->name << " model from " << iterator->path << "." << std::endl;
    models.insert(std::make_pair(iterator->name, model));
  }

  //
  // launch web server
  std::string host = args["host"].as<std::string>();
  size_t port = args["port"].as<size_t>();

  httplib::Server server;
  server.new_task_queue = [args] { return new httplib::ThreadPool(args["listener-threads"].as<size_t>()); };

  server.Get("/v1/stat", [models](const httplib::Request& req, httplib::Response& res) {
    res.set_content(lgbm_serving::getServerStat(models), "application/json");
  });

  server.Post(R"(/v1/models/([a-zA-Z0-9]+):predict)", [models](const httplib::Request& req, httplib::Response& res) {
    std::string modelName = req.matches[1];

    auto iterator = models.find(modelName);
    if (iterator == models.end()) {
      res.status = 400;
      res.set_content("{\"error\": \"There is no model\"}", "application/json");
      return;
    }

    // parse payload
    std::pair<size_t, std::vector<float*>> features;
    try {
      features = lgbm_serving::parse2DFloatArray(req.body);
    } catch (...) {
      res.status = 400;
      res.set_content("{\"error\": \"Cannot parse json array\"}", "application/json");
      return;
    }
    auto ncols = features.first;
    auto nrows = features.second.size();
    auto nClasses = iterator->second->getNumClasses();

    if (ncols != iterator->second->getNumFeatures()) {
      res.status = 400;
      res.set_content("{\"error\": \"invalid shape\"}", "application/json");
      for (const auto* feat : features.second)
        delete[] feat;
      return;
    }

    // inference
    int64_t outputLength;
    std::vector<double> outResult;
    outResult.resize(nClasses * nrows, 0.0);
    LGBM_BoosterPredictForMats(iterator->second->getHandle(), (const void**)features.second.data(), C_API_DTYPE_FLOAT32, nrows, ncols,
                               C_API_PREDICT_NORMAL, 0, "", &outputLength, outResult.data());

    if (outputLength != nrows * nClasses) {
      res.status = 400;
      res.set_content("{\"error\": \"invalid shape\"}", "application/json");
    } else {
      res.set_content(lgbm_serving::serializeModelOutput(nrows, nClasses, outResult.data()), "application/json");
    }

    for (const auto* feat : features.second)
      delete[] feat;
  });

  std::cout << "Running server on " << host << ":" << port << std::endl;
  server.listen(args["host"].as<std::string>().c_str(), args["port"].as<size_t>());

  // clean up
  for (const auto item : models)
    delete item.second;

  return 0;
}

cxxopts::ParseResult parseCLIArgs(int argc, char** argv) {
  cxxopts::Options options(argv[0], "A lightweight server for LightGBM");

  // clang-format off
  options.add_options()
    ("host", "Host", cxxopts::value<std::string>()->default_value("localhost"))
    ("p,port", "Port", cxxopts::value<size_t>()->default_value("8080"))
    ("c,config", "Model Config File", cxxopts::value<std::string>()->default_value(""))
    ("l,listener-threads", "Num of threads of listener", cxxopts::value<size_t>()->default_value("4"))
    ("h,help", "Print usage");
  // clang-format on

  auto args = options.parse(argc, argv);

  if (args.count("help")) {
    std::cout << options.help() << std::endl;
    std::exit(0);
  }

  return args;
}
