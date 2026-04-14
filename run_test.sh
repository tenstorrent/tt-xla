#!/bin/bash
docker exec --user 4123:4123 tt-xla-ird-mvasiljev bash -c 'cd /home/mvasiljev/tt-xla && source venv/activate && pytest -sv tests/benchmark/test_llms.py -k test_gpt_oss_20b_tp_galaxy_batch_size_64 --num-layers 6 --max-output-tokens 3 2>&1'
