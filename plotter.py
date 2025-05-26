import csv
import matplotlib.pyplot as plt

# Read memory values from CSV
memory_chain = []

with open('pytest-memory-usage.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        start = float(row['start_mem'])
        end = float(row['end_mem'])
        if not memory_chain:
            memory_chain.append(start)
        memory_chain.append(end)

# Generate time steps (x-axis)
steps = list(range(len(memory_chain)))

# Plot memory usage
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(steps, memory_chain, marker='o')
plt.title('Chained Memory Usage Over Time')
plt.xlabel('Step')
plt.ylabel('Memory (MB)')
plt.grid(True)

# Compute and plot memory differences
memory_diffs = [memory_chain[i+1] - memory_chain[i] for i in range(len(memory_chain)-1)]
diff_steps = list(range(1, len(memory_chain)))
plt.subplot(2, 1, 2)
plt.bar(diff_steps, memory_diffs)
plt.title('Memory Usage Differences Between Steps')
plt.xlabel('Step')
plt.ylabel('Difference (MB)')
plt.grid(True)

plt.tight_layout()
plt.savefig('memory_usage_plot.png')
