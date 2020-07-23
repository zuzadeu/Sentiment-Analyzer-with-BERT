#!/usr/bin/env bash
NUMBER_OF_PROCS=1
bert-serving-start -model_dir models/uncased_L-12_H-768_A-12 -graph_tmp_dir ${PROJECT_HOME}/tmp -cpu -http_port 8125 -num_worker=${NUMBER_OF_PROCS} -max_seq_len=18 > ${PROJECT_HOME}/logs/bert-server.log
