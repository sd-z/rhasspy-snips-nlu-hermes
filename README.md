# Rhasspy Snips NLU Hermes

MQTT service for Rhasspy that uses [Snips NLU](https://snips-nlu.readthedocs.io/en/latest/) to implement `hermes/nlu` functionality.

## Requirements

* Python 3.7
* [Snips NLU](https://snips-nlu.readthedocs.io/en/latest/)

## Installation

```bash
$ git clone https://github.com/rhasspy/rhasspy-snips-nlu-hermes
$ cd rhasspy-snips-nlu-hermes
$ ./configure
$ make
$ make install
```

## Running

```bash
$ bin/rhasspy-snips-nlu-hermes <ARGS>
```

## Command-Line Options

```
usage: rhasspy-snips-nlu-hermes [-h] --language LANGUAGE
                                [--engine-path ENGINE_PATH]
                                [--dataset-path DATASET_PATH]
                                [--casing {upper,lower,ignore}]
                                [--no-overwrite-train] [--host HOST]
                                [--port PORT] [--username USERNAME]
                                [--password PASSWORD] [--tls]
                                [--tls-ca-certs TLS_CA_CERTS]
                                [--tls-certfile TLS_CERTFILE]
                                [--tls-keyfile TLS_KEYFILE]
                                [--tls-cert-reqs {CERT_REQUIRED,CERT_OPTIONAL,CERT_NONE}]
                                [--tls-version TLS_VERSION]
                                [--tls-ciphers TLS_CIPHERS]
                                [--site-id SITE_ID] [--debug]
                                [--log-format LOG_FORMAT]

optional arguments:
  -h, --help            show this help message and exit
  --language LANGUAGE   Snips language (de, en, es, fr, it, ja, ko, pt_br,
                        pt_pt, zh)
  --engine-path ENGINE_PATH
                        Path to read/write trained engine
  --dataset-path DATASET_PATH
                        Path to write YAML dataset during training
  --casing {upper,lower,ignore}
                        Case transformation for input text (default: ignore)
  --no-overwrite-train  Don't overwrite Snips engine configuration during
                        training
  --host HOST           MQTT host (default: localhost)
  --port PORT           MQTT port (default: 1883)
  --username USERNAME   MQTT username
  --password PASSWORD   MQTT password
  --tls                 Enable MQTT TLS
  --tls-ca-certs TLS_CA_CERTS
                        MQTT TLS Certificate Authority certificate files
  --tls-certfile TLS_CERTFILE
                        MQTT TLS certificate file (PEM)
  --tls-keyfile TLS_KEYFILE
                        MQTT TLS key file (PEM)
  --tls-cert-reqs {CERT_REQUIRED,CERT_OPTIONAL,CERT_NONE}
                        MQTT TLS certificate requirements (default:
                        CERT_REQUIRED)
  --tls-version TLS_VERSION
                        MQTT TLS version (default: highest)
  --tls-ciphers TLS_CIPHERS
                        MQTT TLS ciphers to use
  --site-id SITE_ID     Hermes site id(s) to listen for (default: all)
  --debug               Print DEBUG messages to the console
  --log-format LOG_FORMAT
                        Python logger format
```
