Hey I have to work on a task described as follows: 
Devstral 123b works on vLLM benchmarks regularly. We want to get higher context length accessible and tt-inference server integration to be done. The specific subtask to focus on right now is figuring out enabling of chunked prefill and getting it to work properly so we can unlock higher context lengths. Currently there's a test in test_data_tensor_parallel_generation.py that's a tweakable devstral test. Let's get this test to pass properly. First there's a bunch of precompiled backbone graphs and then i think the test runs. There's errors we keep hitting with the backbone compilation. Some work has been done but whether it works or not is not sure. it could even be causing errors. There's a report in devstral_batch128_notes/chunked_prefill_issue/work_done.md for reference and it should be looked through as well. There's also some work in that chunked prefill issue direcotry that could be useful so take a look. Go through it with an agent.

Context and start off info:
Use the directory high_seq_length_support in devstral_batch_128_notes to track work progress.
Let's make a task.md file to track the task. Refer to the task.md file on each auto compaction and every restart, add that to the memory as something to do so you can keep good track of your work. Also refer to the notion page tenstorrent that I have for developer tools, instructions for building, env vars to use and such, add that to memory too. Most commonly needed env vars are TTXLA_LOGGER_LEVEL=DEBUG or VERBOSE, TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG, TT_METAL_OPERATION_TIMEOUT_SECONDS=60. Also get logs. I have asked below to deploy an agent to understand all this so dont worry. 
You are on a blackhole galaxy on exabox slurm. Every test needs to run in the docker container which can be accessed with:

docker exec -it --user 4076:4076 tt-xla-ird-ssalice /bin/bash

In the docker container we have ~/ or /home/ssalice being analagous to /data/ssalice outside the docker container as it is home mounted there. The machine has 32 devices with 1tb total DRAM. You have sudo permissions in the docker but not outside.
If you need to reset machines due to a hang for example you'd nneed to launch a shell that does
 
uvx tt-smi@latest -glx_reset

and it should output ` Re-initialized 32 boards after reset. Exiting...` for it to have succeded.

Things to do:
First deploy an agent to go through the vllm_tt files THOROUGHLY and add info on the files and directory structure and important info to know from there for the task to the task.md. Concurrently deploy an agent to go through the Tensotrrent Notion page I have and understand it deeply and understand all the building info and env vars and dev tools and give short info for it on the task.md file. Ensure the agent ONLY looks at building and stuff RELEVANT to the machine you are working on (exabox slurm bh galaxy). Deploy an agent to see what skills and stuff exist too and note down on the task.md too. Then review the info given from each agent and look at the files as necessary.

The tt-xla branch to work on is ssalice/devstral-qwen-wip-07-13-2026, but feel free to make other branches off of it too. the tt-mlir branch to work off of that should be pinned is ssalice/devstral-wip-06252026-mlir. 

The most likely issue is with sharding? I have a log devstral_test.log currently available to start work on. The test I gave was 

`TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG TTXLA_LOGGER_LEVEL=DEBUG TT_METAL_OPERATION_TIMEOUT_SECONDS=120 pytest -svv tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py::test_dptp_devstral[mesh_shape0-True-bfp_bf8] |& tee devstral_test.log` 

Run loops of this workflow as needed to gather as much info as possible so we can be thorough in our information.

Constraints:
Always run 2-3 layer versions of the test first to verify changes worked before running the full 88 layer test that takes a long while to completely run. Ask me when it's time to validate the full model. You can set the num_hidden_layers as a config in the tp config for the specific test in test_vllm_benchmarks.py
Try to get the fixes in from tt-xla as much as possible. If the issue is in tt-mlir then attempt to first find a workaround from tt-xla. If that's not an option and no workaround could be managed to be found after 3 tries then go for a tt-mlir fix. 
Keep in mind, sometimes errors can come from fixes we made so make sure to back trace logic from previous commits too before dialing in on one fix.
Make sure to have logs and information for me to refer to and crosscheck your work. At the end clean up old unnecessary logs so it's easy for me to verify. Keep the logs original, if you want to dial in on specific stuff from a log make a copy of the log with the dialed in focused info. 
Make as many branches as needed locally and you can play around with tt-mlir as well. Once your work is done please clean up branches and push to tt-xla and tt-mlir.
DO NOT push work to main ever. SHould be gated but warning. Do not make any PRs.
MAKE SURE you are working on ~/ssalice/temp/tt-xla (or /data/ssalice/temp/tt-xla from outside the docker container). Do not work from ~/ssalice/tt-xla, that is not relevant to you.
When you rebuild tt-mlir if needed, make sure the correct thing has been rebuilt and nothing got overwritten.
Clean up branches after all work is done so I can understand changes easily. ALso run pre-commit after everything is done.
If I say im stepping away, just make the wisest choice when hit with a hard to decide situation. Consult with an advisor as well when hit with a forking decision. Review decision with an agent after. Keep a decisions.md file also in track so you can note down forks in decisions and keep track. Add to memory to also look at this file upon restart and from post compaction. Also make sure I can see it after too. Use this as the source to see if a decision made prior could be causing a new error, or if it is a genuine new error.

Tips:
Use agents and skills and plugins as needed to maintain and manage context as much as possible to avoid hallucination. Use the advisor to review your work after each possible error and each possible solution is found. Also have agents deployed to review potential issues with hypothetical errors and hypothetical solutions to ensure you're working at best quality, and to keep solutions as simple as needed and also as wide in coverage of edge cases. Have an agent also think about edge cases after each solution you propose. Work om it like a team.
If there's issues, sometimes it's a good idea to deploy an agent to browse through tt-xla and tt-mlir for issues that might be similar. If there's something returned do verify it 2-3 times to make sure it's relevant with another agent and you as well. Nonetheless dont let this stop you. Keep working as far as needed to find solutions.
Dont forget to export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/ && source venv/activate whenever a shell is launched! And ofc rebuild tt-mlir if needed too. Isnt necessary most of the time. git submodule update --init --recursive will update tt-forge-models and rebuilding pulls a new tt-mlir so it would overrite change you made within tt-mlir if not committed.
Also make sure you look at the right report, task and decisions and instructions markdown files. Look at the one in the high_seq_length directory not the chunked_prefill one.

All good? If everything is good I'll step away while you work on this.