
DIR="pcc_after";
mkdir $DIR;

pytest -vv --junitxml=$DIR/test1.xml tests/runner/test_models.py::test_all_models_torch[albert/token_classification/pytorch-xlarge_v2-single_device-full-inference]  |& tee $DIR/test1.log
pytest -vv --junitxml=$DIR/test2.xml tests/runner/test_models.py::test_all_models_torch[albert/token_classification/pytorch-xxlarge_v2-single_device-full-inference]  |& tee $DIR/test2.log
pytest -vv --junitxml=$DIR/test3.xml tests/runner/test_models.py::test_all_models_torch[qwen_3/causal_lm/pytorch-0_6b-single_device-full-inference]  |& tee $DIR/test3.log
pytest -vv --junitxml=$DIR/test4.xml tests/runner/test_models.py::test_all_models_torch[qwen_3/causal_lm/pytorch-1_7b-single_device-full-inference]  |& tee $DIR/test4.log
pytest -vv --junitxml=$DIR/test5.xml tests/runner/test_models.py::test_all_models_torch[phi2/causal_lm/pytorch-microsoft/phi-2-single_device-full-inference]  |& tee $DIR/test5.log
pytest -vv --junitxml=$DIR/test6.xml tests/runner/test_models.py::test_all_models_torch[phi3/causal_lm/pytorch-microsoft/Phi-3-mini-128k-instruct-single_device-full-inference]  |& tee $DIR/test6.log
pytest -vv --junitxml=$DIR/test7.xml tests/runner/test_models.py::test_all_models_torch[phi3/causal_lm/pytorch-microsoft/Phi-3-mini-4k-instruct-single_device-full-inference]  |& tee $DIR/test7.log
pytest -vv --junitxml=$DIR/test8.xml tests/runner/test_models.py::test_all_models_torch[llama/causal_lm/pytorch-TinyLlama_v1.1-single_device-full-inference]  |& tee $DIR/test8.log
