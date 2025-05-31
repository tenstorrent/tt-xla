# The implementation of Mixtral 8x7b, DP + EP model

---

```
tests/jax/models/mixtral/jax_config.py -> Implements the jax configuration
```

```
tests/jax/models/mixtral/requirements.txt -> The required libraries to run the model (pip install -r requirements.txt)
```

```
tests/jax/models/mixtral/singlechip/flaxmixtral.py -> Implements the single-device Mixtral model

                                   /flaxconfigmixtral.py -> Implements the Mixtral Configuration

                                   /convert_weights.py -> Converts the Hugging Face pre-trained weights to a single-device Mixtral model
```

```
tests/jax/models/mixtral/multichip/multichipmixtral.py -> Implements the multi-device Mixtral model using the jax_config file and 1x8 mesh
```

```
tests/jax/models/mixtral/tests/debugger.py -> My file for debugging (can be ignored)

                        /tests/hf_vs_single.py -> A file to compare results between the pre-trained Hugging Face model and my Mixtral single-device model

                        /tests/multi_vs_single.py -> A file to compare results between the single-device and multi-device Mixtral model

```
