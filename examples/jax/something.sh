export JAX_NUM_CPU_DEVICES=8
num_processes=1

range=$(seq 0 $(($num_processes - 1)))

for i in $range; do
  LOGGER_LEVEL=DEBUG python examples/jax/print_stablehlo_multihost.py $i $num_processes > /tmp/toy_$i.out &
done

wait

for i in $range; do
  echo "=================== process $i output ==================="
  cat /tmp/toy_$i.out
  echo
done