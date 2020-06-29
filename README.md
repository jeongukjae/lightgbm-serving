# lightgbm-serving

A server for lightGBM

usage:

- build:

  ```sh
  $ git clone https://github.com/jeongukjae/lightgbm-serving
  $ cd lightgbm-serving
  $ mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release
  $ make
  ```

- configuration:

  ```json
  {
      "config": [
          {
              "name": "test-model",
              "path": "/path/to/model",
              "nClasses": 1
          }
      ]
  }
  ```

- run:

  ```sh
  $ ./lightgbm-serving --host 0.0.0.0 --port 8080 --config path-to-config.json
  ```

- inference:

  ```sh
  $ curl http://localhost:8080/v1/models/test-model:predict -d "[[1,2,3,4,5], [1,2,3,4,5]]"
  ```
