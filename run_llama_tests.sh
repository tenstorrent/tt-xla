DIR=llama3b_rope_fix_e60ba14a8_just_first_fix
mkdir -p $DIR

pytest -svv tests/integrations/vllm_plugin/sampling/test_sampling_params.py::test_output_coherence_nongreedy |& tee $DIR/test_output_coherence_nongreedy1.log
pytest -svv tests/integrations/vllm_plugin/sampling/test_sampling_params.py::test_output_coherence_nongreedy |& tee $DIR/test_output_coherence_nongreedy2.log
pytest -svv tests/integrations/vllm_plugin/sampling/test_sampling_params.py::test_output_coherence_nongreedy |& tee $DIR/test_output_coherence_nongreedy3.log
pytest -svv tests/integrations/vllm_plugin/generative/test_llama3_3b_generation.py::test_llama3_3b_generation |& tee $DIR/test_llama3_3b_generation.log
pytest -svv tests/integrations/vllm_plugin/generative/test_llama3_3b_generation.py::test_llama3_3b_generation_opt_level_1[batch1] |& tee $DIR/test_llama3_3b_generation_opt_level_1_b1_1.log
pytest -svv tests/integrations/vllm_plugin/generative/test_llama3_3b_generation.py::test_llama3_3b_generation_opt_level_1[batch1] |& tee $DIR/test_llama3_3b_generation_opt_level_1_b1_2.log
pytest -svv tests/integrations/vllm_plugin/generative/test_llama3_3b_generation.py::test_llama3_3b_generation_opt_level_1[batch1] |& tee $DIR/test_llama3_3b_generation_opt_level_1_b1_3.log
pytest -svv tests/integrations/vllm_plugin/generative/test_llama3_3b_generation.py::test_llama3_3b_generation_opt_level_1[batch2] |& tee $DIR/test_llama3_3b_generation_opt_level_1_b2.log

