# lightgbm-serving

A lightweight http server for lightGBM that support multi-model server.

## Installation

### Using docker

```sh
$ docker pull jeongukjae/lightgbm-serving
```

### Build from source code

```sh
$ git clone https://github.com/jeongukjae/lightgbm-serving
$ cd lightgbm-serving
$ mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release
$ make
$ ./lightgbm-server --help
A lightweight server for LightGBM

Usage:
  ./lightgbm-server [OPTION...]

      --host arg              Host (default: localhost)
  -p, --port arg              Port (default: 8080)
  -c, --config arg            Model Config File (default: "")
  -l, --listener-threads arg  Num of threads of listener (default: 4)
  -v, --version               Show version string and infos and exit
  -h, --help                  Print usage
```

## Usage

LightGBM-Serving takes a config file and model files. The config file is passed as a cli argument (`-c` or `--config`).

### Configure Model Config

The config file should be json file, and should be structed like below.

```json
{
    "config": [
        {
            "name": "test-model",
            "path": "/path/to/model"
        },
        ...
        // and more models!!
    ]
}
```

### Run the server

```sh
$ ./lightgbm-serving --host 0.0.0.0 --port 8080 --config path-to-config.json
```

or use docker

```
$ docker run --rm -it \
  -v PATH_TO_MODEL:/models \
  -v PATH_TO_CONFIGL:/config.json \
  jeongukjae/lightgbm-serving -c /config.json
[2020-10-16 01:29:23.200] [info] Found 1 configs.
[2020-10-16 01:29:23.224] [info] Loaded test-model model from '/models/model.lgbm'.
[2020-10-16 01:29:23.224] [info] Running server on http://localhost:8080
```

### Show debug log

Set environment variable `LGBM_DEBUG` to any values. (condition: `if (std::getenv("LGBM_DEBUG"))`)

Example:

```sh
$ LGBM_DEBUG=1 ./lightgbm-server --config ../temp/config.json --host 0.0.0.0
[2020-10-16 01:27:39.549] [debug] CLI Arguments:
[2020-10-16 01:27:39.549] [debug]  - config : ../temp/config.json
[2020-10-16 01:27:39.549] [debug]  - host : 0.0.0.0
[2020-10-16 01:27:39.550] [info] Found 1 configs.
[2020-10-16 01:27:39.571] [info] Loaded test-model model from '../temp/test-model'.
[2020-10-16 01:27:39.571] [info] Running server on http://0.0.0.0:8080
[2020-10-16 01:27:44.966] [debug] POST /v1/models/test-model:predict HTTP/HTTP/1.1 200 - from 127.0.0.1
[2020-10-16 01:27:46.944] [debug] POST /v1/models/test-model:predict HTTP/HTTP/1.1 200 - from 127.0.0.1
[2020-10-16 01:27:47.485] [debug] POST /v1/models/test-model:predict HTTP/HTTP/1.1 200 - from 127.0.0.1
[2020-10-16 01:31:34.928] [debug] POST /v1/models/test-model:predict HTTP/HTTP/1.1 400 - from 127.0.0.1
```

### View server status

```sh
$ curl http://localhost:8080/v1/stat
```

### Inference

```sh
$ curl http://localhost:8080/v1/models/{MODEL_NAME_IN_CONFIG_FILE}:predict -d "[[1,2,3,4,5], [1,2,3,4,5]]"
```

**###Payload should be 2d array that shape of is `(batch size, num features)`.###**

#### Inference Results

If a parameter `num_classes` of model is 1, then server will return 1d array with shape `(batch size, )` like `[0.5,0.3,0.2]`. If not, server will return 2d array with shape `(batch size, num classes)` like `[[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0]]`.

#### When error occured

| code | body                                   | reason                                     |
| ---- | -------------------------------------- | ------------------------------------------ |
| 400  | `{"error": "There is no model"}`       | When the model name key in url is missing. |
| 400  | `{"error": "Cannot parse json array"}` | Cannot parse json array.                   |
| 400  | `{"error": "invalid shape"}`           | Invalid shape when parsing json array.     |
