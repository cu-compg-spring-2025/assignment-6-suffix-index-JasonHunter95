import time
from venv import create
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
import tracemalloc
# importing the suffix scripts
import suffix_array
import suffix_tree
import suffix_trie
import ukkonens_suffix_tree
import ukkonens_suffix_array
import resource

soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (hard, hard))

def generate_random_sequence(length, alphabet="ACGT"):
    """Generate a random sequence of specified length from the given alphabet."""
    choices = np.random.choice(list(alphabet), size=length)
    random_sequence = ''.join(choices)
    return random_sequence

def measure_construction(fn, sequence):
    """Measures time and memory taken to construct a structure based on the function passed in"""
    tracemalloc.start()
    
    start_time = time.perf_counter()
    result = fn(sequence)
    elapsed_time = time.perf_counter() - start_time
    
    curr, mem_usage = tracemalloc.get_traced_memory()
    mem_usage = mem_usage / (1024 * 1024)  # convert these to MB from bytes
    curr = curr / (1024 * 1024)
    tracemalloc.stop()
    
    # print(f"Current memory usage: {curr} MB")
    # print(f"Peak memory usage: {mem_usage} MB")
    return result, elapsed_time, mem_usage

# helper function for construction benchmarking
def benchmark_structure(build_fn, structure, sequence, size):
    """Benchmark construction time and memory for a single data structure."""
    try:
        _, elapsed_time, mem_usage = measure_construction(build_fn, sequence)
        return {
            'sequence_size': size,
            'structure': structure,
            'construction_time': elapsed_time,
            'memory_usage_mb': mem_usage
        }
    except Exception as e:
        print(f"Error benchmarking {structure} for size {size}: {e}")
        return None

def benchmark_construction(sizes, repeats, trie_max_size, naive_max_size):
    """Benchmark construction time and memory for different sequence sizes."""
    results = []
    
    # Define the build functions for naive methods and Ukkonen methods
    naive_build_methods = [
        (suffix_array.build_suffix_array, "Suffix Array"),
        (suffix_tree.build_suffix_tree, "Suffix Tree")
    ]
    # Ukkonen-based methods are always run
    ukkonen_build_methods = [
        (lambda seq: ukkonens_suffix_tree.SuffixTree(seq), "Ukkonen's Suffix Tree"),
        (lambda seq: ukkonens_suffix_array.build_suffix_array(seq), "Ukkonen's Suffix Array")
    ]
    
    for size in tqdm(sizes, desc="Benchmarking sequence sizes"):
        for _ in range(repeats):
            sequence = generate_random_sequence(size)
                    
            if size <= naive_max_size:
                for build_fn, name in naive_build_methods:
                    record = benchmark_structure(build_fn, name, sequence, size)
                    if record:
                        results.append(record)
            else:
                print(f"Skipping naive methods for size {size} (exceeds threshold {naive_max_size})")
            
            # Always run Ukkonen methods
            for build_fn, name in ukkonen_build_methods:
                record = benchmark_structure(build_fn, name, sequence, size)
                if record:
                    results.append(record)
            
            # benchmark suffix trie only if size is within limit
            if size <= trie_max_size:
                record = benchmark_structure(suffix_trie.build_suffix_trie, "Suffix Trie", sequence, size)
                if record:
                    results.append(record)
            else:
                print(f"Skipping Suffix Trie for size {size} (exceeds max size {trie_max_size})")
    
    return pd.DataFrame(results)

# helper function for query benchmarking
def measure_and_record_query(search_fn, structure, text, query, requires_text, size, query_length, structure_name):
    """Measure and record the time taken for a single query operation."""
    try:
        start = time.perf_counter()
        if requires_text:
            _ = search_fn(text, structure, query)
        else:
            _ = search_fn(structure, query)
        elapsed = time.perf_counter() - start
        return {
            'sequence_size': size,
            'query_length': query_length,
            'structure': structure_name,
            'query_time': elapsed
        }
    except Exception as e:
        print(f"Error during query on {structure_name}: {e}")
        return None

def benchmark_query(sizes, query_lengths, repeats, trie_max_size):
    """Benchmark query time for different sequence and query sizes."""
    results = []
    
    for size in tqdm(sizes, desc="Benchmarking queries"):
        sequence = generate_random_sequence(size)
        
        # build data structures
        trie = suffix_trie.build_suffix_trie(sequence, show_progress=False) if size <= trie_max_size else None
        tree = suffix_tree.build_suffix_tree(sequence, show_progress=False)
        array = suffix_array.build_suffix_array(sequence)
        ukkonens_tree = ukkonens_suffix_tree.SuffixTree(sequence)
        ukkonens_array = ukkonens_suffix_array.build_suffix_array(sequence)  # build ukkonen's suffix array
        
        # create tuples for each search method to iterate on
        search_methods = [
            (suffix_array.search_array, "Suffix Array", True),
            (suffix_tree.search_tree, "Suffix Tree", False),
            (ukkonens_suffix_array.search_array, "Ukkonen's Suffix Array", True),
            (ukkonens_suffix_tree.search_tree, "Ukkonen's Suffix Tree", False)
            ]
        if trie is not None:
            search_methods.append((suffix_trie.search_trie, "Suffix Trie", False))
        
        for qlen in query_lengths:
            for _ in range(repeats):
                # generate query substring from within the sequence
                if size > qlen:
                    start_index = random.randint(0, size - qlen)
                    query = sequence[start_index:start_index + qlen]
                else:
                    query = sequence
                
                # iterate through each tuple
                for search_fn, name, requires_text in search_methods:
                    structure = (array if name == "Suffix Array" else
                                 tree if name == "Suffix Tree" else
                                 trie if name == "Suffix Trie" else
                                 ukkonens_array if name == "Ukkonen's Suffix Array" else
                                 ukkonens_tree if name == "Ukkonen's Suffix Tree" else None)
                    record = measure_and_record_query(search_fn, structure, sequence, query, requires_text, size, qlen, name)
                    if record:
                        results.append(record)
    
    return pd.DataFrame(results)

# helper function to create and save a log-log plot
def create_loglog_plot(x, y, hue, data, title, xlabel, ylabel, filename, output_dir = 'figures', extra_lines=None):
    plt.figure(figsize=(12, 7))
    ax = sns.lineplot(data=data, x=x, y=y, hue=hue, marker='o', linewidth=2.5)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    
    if extra_lines is not None:
        x_range = np.logspace(np.log10(data[x].min()), np.log10(data[x].max()), 100)
        for style, label, func in extra_lines:
            plt.plot(x_range, func(x_range), style, alpha=0.3, label=label)
        plt.legend(title='Data Structure')
    else:
        plt.legend(title='Data Structure', fontsize=10)
    
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
# def compute_slope(x_values, y_values):
#     """Compute the slope of the line in a log-log plot using linear regression."""
#     log_x = np.log10(x_values)
#     log_y = np.log10(y_values)
#     slope, intercept = np.polyfit(log_x, log_y, 1)
#     return slope, intercept

def plot_construction_results(results, output_dir='figures'):
    """Plot construction benchmark results.
    - uses helper function to create log-log plots.
    - basic error handling.
    """
    if results.empty:
        raise ValueError("Results dataframe is empty. Nothing to plot.")
        
    os.makedirs(output_dir, exist_ok=True)

    # plot construction time with theoretical guidelines: O(n) and O(n log n) and O(n^2)
    scale_factor = results['construction_time'].median() / results['sequence_size'].median()
    extra_lines_time = [
        ('k-.', 'O(n^2)', lambda x: scale_factor * x**2),
        ('k--', 'O(n)', lambda x: scale_factor * x),
        ('k:', 'O(n log n)', lambda x: scale_factor * x * np.log(x))
    ]
    # construction time
    create_loglog_plot(
        x='sequence_size',
        y='construction_time',
        hue='structure',
        data=results,
        title='Construction Time vs Sequence Size',
        xlabel='Sequence Size (bp)',
        ylabel='Time (seconds)',
        filename='construction_time.png',
        output_dir='figures',
        extra_lines=extra_lines_time
    )

    # memory usage
    create_loglog_plot(
        x='sequence_size',
        y='memory_usage_mb',
        hue='structure',
        data=results,
        title='Memory Usage vs Sequence Size',
        xlabel='Sequence Size (bp)',
        ylabel='Memory Usage (MB)',
        filename='memory_usage.png',
        output_dir='figures'
    )

def plot_query_results(results, output_dir='figures'):
    """Plot query benchmark results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # query time by sequence size
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results, x='sequence_size', y='query_time', hue='structure', marker='o')
    plt.title('Query Time vs Sequence Size')
    plt.xlabel('Sequence Size (bp)')
    plt.ylabel('Time (seconds)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Data Structure')
    plt.savefig(os.path.join(output_dir, 'query_time_by_size.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # query time by query length
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results, x='query_length', y='query_time', hue='structure', marker='o')
    plt.title('Query Time vs Query Length')
    plt.xlabel('Query Length (bp)')
    plt.ylabel('Time (seconds)')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Data Structure')
    plt.savefig(os.path.join(output_dir, 'query_time_by_length.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # facet grid for different sequence sizes
    g = sns.FacetGrid(results, col='sequence_size', col_wrap=2, height=4)
    g.map_dataframe(sns.lineplot, x='query_length', y='query_time', hue='structure', marker='o')
    g.add_legend(title='Data Structure')
    g.set_axis_labels('Query Length (bp)', 'Time (seconds)')
    g.set_titles('Sequence Size: {col_name} bp')
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'query_time_facet.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_dashboard(construction_results, query_results, output_dir='figures'):
    """Create a comprehensive performance dashboard with multiple plots."""
    plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2)
    
    # construction time (log-log)
    ax1 = plt.subplot(gs[0, 0])
    sns.lineplot(data=construction_results, x='sequence_size', y='construction_time', 
                 hue='structure', marker='o', ax=ax1)
    ax1.set_title('Construction Time')
    ax1.set_xlabel('Sequence Size (bp)')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # memory usage (log-log)
    ax2 = plt.subplot(gs[0, 1])
    sns.lineplot(data=construction_results, x='sequence_size', y='memory_usage_mb', 
                 hue='structure', marker='o', ax=ax2)
    ax2.set_title('Memory Usage')
    ax2.set_xlabel('Sequence Size (bp)')
    ax2.set_ylabel('Memory (MB)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # query time by sequence size (fixing query length = 20)
    ax3 = plt.subplot(gs[1, 0])
    sns.lineplot(data=query_results[query_results['query_length'] == 100], 
                 x='sequence_size', y='query_time', hue='structure', marker='o', ax=ax3)
    ax3.set_title('Query Time (Fixed Query Length: 100)')
    ax3.set_xlabel('Sequence Size (bp)')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # query time by query length (fixing sequence size = 1000)
    ax4 = plt.subplot(gs[1, 1])
    sns.lineplot(data=query_results[query_results['sequence_size'] == 10000], 
                 x='query_length', y='query_time', hue='structure', marker='o', ax=ax4)
    ax4.set_title('Query Time (Fixed Sequence Size: 10000)')
    ax4.set_xlabel('Query Length (bp)')
    ax4.set_ylabel('Time (seconds)')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # add a single legend for the whole figure
    handles, labels = ax1.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01))
    
    # remove individual legends to avoid duplication
    for ax in [ax1, ax2, ax3, ax4]:
        ax.get_legend().remove()
    
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    plt.suptitle('Performance Comparison of Suffix Data Structures', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run the benchmarking suite."""
    construction_sizes = [10, 100, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 10000000]
    
    # for querying, use a subset of sizes to keep runtime reasonable
    query_sizes = [50, 100, 500, 1000, 5000, 10000]
    query_lengths = [5, 10, 20, 50, 100]
    repeats = 3
    
    # skip sizes that are too large for the trie and naive methods to handle
    trie_max_size = 10000
    naive_max_size = 100000

    print("Starting construction benchmarks...")
    construction_results = benchmark_construction(construction_sizes, repeats, trie_max_size, naive_max_size)
    construction_results.to_csv('figures/construction_benchmark.csv', index=False)
    plot_construction_results(construction_results)
    
    print("Starting query benchmarks...")
    query_results = benchmark_query(query_sizes, query_lengths, repeats, trie_max_size)
    query_results.to_csv('figures/query_benchmark.csv', index=False)
    plot_query_results(query_results)
    
    create_performance_dashboard(construction_results, query_results)
    
    # slope = compute_slope(construction_results['sequence_size'], construction_results['construction_time'])
    # print(f"Construction time slope: {slope[0]:.2f}")
    # slope2 = compute_slope(construction_results['sequence_size'], construction_results['memory_usage_mb'])
    # print(f"Memory usage slope: {slope2[0]:.2f}")
        
    print("Benchmarking complete! Results saved to 'figures' directory.")

if __name__ == "__main__":
    main()