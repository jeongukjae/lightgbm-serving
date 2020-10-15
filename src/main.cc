#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include "config.h"
#include "config_parser.hh"
#include "cxxopts.hpp"
#include "httplib.h"
#include "model.hh"
#include "server.hh"
#include "spdlog/spdlog.h"

cxxopts::ParseResult parseCLIArgs(int argc, char** argv);
std::string getServerStat(std::map<std::string, lgbm_serving::Model*> models);

int main(int argc, char** argv) {
  if (std::getenv("LGBM_DEBUG"))
    spdlog::set_level(spdlog::level::debug);

  //
  // parse cli arguments
  auto args = parseCLIArgs(argc, argv);
  spdlog::debug("CLI Arguments: ");
  for (auto argument : args.arguments())
    spdlog::debug(" - {} : {}", argument.key(), argument.value());

  std::string configFilePath = args["config"].as<std::string>();
  if (configFilePath.empty()) {
    spdlog::error("Cannot parse config file.");
    return 1;
  }

  //
  // parse model config
  lgbm_serving::ConfigParser parser;
  try {
    parser.parseModelConfig(configFilePath);
  } catch (std::runtime_error& error) {
    spdlog::error("Cannot parse config file: {}", error.what());
    return 1;
  }

  //
  // load models
  size_t numModel = parser.getLength();
  if (numModel == 0) {
    spdlog::error("Config file contains nothing.");
    return 1;
  }

  spdlog::info("Found {} configs.", numModel);
  std::map<std::string, lgbm_serving::Model*> models;
  for (auto iterator = parser.begin(); iterator < parser.end(); iterator++) {
    lgbm_serving::Model* model = new lgbm_serving::Model;

    model->setConfig(&(*iterator));
    try {
      model->load(iterator->path);
    } catch (std::runtime_error& error) {
      spdlog::error("Cannot load model: {}", error.what());
      return 1;
    }

    spdlog::info("Loaded {} model from '{}'.", iterator->name, iterator->path);
    models.insert(std::make_pair(iterator->name, model));
  }

  //
  // launch web server
  std::string host = args["host"].as<std::string>();
  size_t port = args["port"].as<size_t>();

  httplib::Server server;
  server.new_task_queue = [args] { return new httplib::ThreadPool(args["listener-threads"].as<size_t>()); };

  server.set_logger([](const httplib::Request& req, const httplib::Response& res) {
    spdlog::debug("{} {} HTTP/{} {} - from {}", req.method, req.path, req.version, res.status, req.remote_addr);
  });

  server.Get("/v1/stat", [models](const httplib::Request& req, httplib::Response& res) {
    res.set_content(lgbm_serving::getServerStat(models), "application/json");
  });

  server.Post(R"(/v1/models/([a-zA-Z0-9-_]+):predict)", [models](const httplib::Request& req, httplib::Response& res) {
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

  spdlog::info("Running server on http://{}:{}", host, port);
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
    ("v,version", "Show version string and infos and exit")
    ("h,help", "Print usage");
  // clang-format on

  auto args = options.parse(argc, argv);

  if (args.count("help")) {
    std::cout << options.help() << std::endl;
    std::exit(0);
  }

  if (args.count("version")) {
    std::cout << "LightGBM Serving " PROJECT_VERSION ", MIT License, https://github.com/jeongukjae/lightgbm-serving" << std::endl;
    std::exit(0);
  }

  return args;
}
