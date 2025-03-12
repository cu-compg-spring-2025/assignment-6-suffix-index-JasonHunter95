import argparse
import os
import tqdm
import psutil
import subprocess
import sys

# try different import approaches based on how the script is being run
try:
    # try relative import first (for use when imported as module)
    from . import utils
except ImportError:
    try:
        # try package import (for tests)
        import src.utils as utils
    except ImportError:
        # try direct import (for direct script execution)
        import utils

def get_args():
    parser = argparse.ArgumentParser(description='Ukkonens Suffix Tree')
    parser.add_argument('--reference',
                        help='Reference sequence file',
                        type=str)
    parser.add_argument('--string',
                        help='Reference sequence',
                        type=str)
    parser.add_argument('--query',
                        help='Query sequences',
                        nargs='+',
                        type=str)
    parser.add_argument('--region',
                        help='Region to visualize (format: start-end)',
                        type=str)
    parser.add_argument('--max-size',
                        help='Maximum sequence length to visualize (default: 200)',
                        type=int,
                        default=100)
    return parser.parse_args()

# Ukkonen's algorithm implementation
class Node:
    def __init__(self, start, end):
        # Children: mapping from character -> Node.
        self.children = {}
        # Suffix link used to speed up tree construction.
        self.suffix_link = None  
        # Start index of the edge label from the parent.
        self.start = start
        # End index for the edge label.
        # For leaves, end is a reference (list) to the global end pointer.
        self.end = end  
        # For leaf nodes, we later set an index indicating the starting position of the suffix.
        self.index = -1

    def edge_length(self, current_index):
        """
        Returns the length of the edge from this node,
        using current_index for leaves (whose end is a reference).
        """
        if isinstance(self.end, list):
            return current_index - self.start + 1
        return self.end - self.start + 1

class SuffixTree:
    def __init__(self, text):
        # Ensure the text ends with a unique terminal symbol.
        self.text = text
        self.size = len(text)
        
        # Create the root node. (Start and end values for root are arbitrary.)
        self.root = Node(-1, -1)
        self.root.suffix_link = self.root  # type: ignore
        
        # Active point for Ukkonen’s algorithm.
        self.active_node = self.root
        self.active_edge = -1
        self.active_length = 0
        
        # Remainder counts the number of suffixes yet to be added.
        self.remaining_suffix_count = 0
        
        # Global end for all leaves; stored as a one-element list so that it acts as a reference.
        self.leaf_end = [-1]
        
        # This variable holds the most recently created internal node
        # during the current extension to set its suffix link.
        self.last_new_node = None

        self.build_suffix_tree()
        # Optionally, assign a suffix index for each leaf by DFS.
        self._set_suffix_index_by_dfs(self.root, 0)

    def build_suffix_tree(self):
        for pos in range(self.size):
            self._extend_suffix_tree(pos)

    def _extend_suffix_tree(self, pos):
        # Update the global end pointer.
        self.leaf_end[0] = pos
        self.remaining_suffix_count += 1
        self.last_new_node = None

        while self.remaining_suffix_count > 0:
            if self.active_length == 0:
                self.active_edge = pos  # The current character position becomes the active edge.
            
            current_char = self.text[self.active_edge]
            # If there is no edge starting with current_char from active_node:
            if current_char not in self.active_node.children:
                # Create a new leaf node.
                self.active_node.children[current_char] = Node(pos, self.leaf_end)
                # If an internal node was waiting for a suffix link, then set it.
                if self.last_new_node is not None:
                    self.last_new_node.suffix_link = self.active_node  # type: ignore
                    self.last_new_node = None
            else:
                # There is an outgoing edge starting with current_char.
                next_node = self.active_node.children[current_char]
                edge_length = next_node.edge_length(pos)
                # If the active_length is greater than or equal to the edge length,
                # walk down the edge.
                if self.active_length >= edge_length:
                    self.active_edge += edge_length
                    self.active_length -= edge_length
                    self.active_node = next_node
                    continue
                # Check if the next character on the edge matches the current character.
                if self.text[next_node.start + self.active_length] == self.text[pos]:
                    # The current character is already on the edge;
                    # we only need to increment active_length.
                    if self.last_new_node is not None and self.active_node != self.root:
                        self.last_new_node.suffix_link = self.active_node  # type: ignore
                        self.last_new_node = None
                    self.active_length += 1
                    break

                # Mismatch found: split the edge.
                split_end = next_node.start + self.active_length - 1
                split = Node(next_node.start, split_end)
                self.active_node.children[current_char] = split

                # Create a new leaf node from the split node.
                split.children[self.text[pos]] = Node(pos, self.leaf_end)
                # Adjust the start of the old node.
                next_node.start += self.active_length
                # Add the old node as a child of the split node.
                split.children[self.text[next_node.start]] = next_node

                # If an internal node was created in a previous extension in the same phase,
                # then set its suffix link to the split node.
                if self.last_new_node is not None:
                    self.last_new_node.suffix_link = split  # type: ignore
                self.last_new_node = split

            self.remaining_suffix_count -= 1

            if self.active_node == self.root and self.active_length > 0:
                self.active_length -= 1
                self.active_edge = pos - self.remaining_suffix_count + 1
            elif self.active_node != self.root:
                self.active_node = self.active_node.suffix_link if self.active_node.suffix_link is not None else self.root

    def _set_suffix_index_by_dfs(self, node, label_height):
        """
        After tree construction, assign suffix indices to leaves by DFS.
        """
        if not node:
            return
        if not node.children:
            node.index = self.size - label_height
            return
        for child in node.children.values():
            self._set_suffix_index_by_dfs(child, label_height + child.edge_length(self.leaf_end[0]))

    def print_edges(self, node=None):
        """
        A helper method to print all edge labels in the suffix tree.
        """
        if node is None:
            node = self.root
        for child in node.children.values():
            end_index = child.end[0] if isinstance(child.end, list) else child.end
            print(self.text[child.start:end_index + 1])
            self.print_edges(child)

def build_suffix_tree_wrapper(text, show_progress=False):
    """
    Build a suffix tree for the input string using Ukkonen's algorithm.
    If the terminal symbol '$' is not present, it is appended.
    """
    if not text.endswith('$'):
        text += '$'
    tree = SuffixTree(text)
    return tree

def search_tree(suffix_tree, pattern):
    """
    Given a suffix_tree (SuffixTree object) and pattern string,
    search for the pattern by walking the tree.
    Returns the number of characters that match the pattern.
    """
    node = suffix_tree.root
    i = 0
    text = suffix_tree.text
    while i < len(pattern):
        if pattern[i] not in node.children:
            return i
        child = node.children[pattern[i]]
        end_index = child.end[0] if isinstance(child.end, list) else child.end
        edge_str = text[child.start:end_index+1]
        j = 0
        while j < len(edge_str) and i+j < len(pattern) and pattern[i+j] == edge_str[j]:
            j += 1
        if j < len(edge_str):
            return i+j
        i += j
        node = child
    return len(pattern)

def main():
    args = get_args()
    T = None
    
    # create script-specific directories for dot and png files
    dot_dir = os.path.join('dots', 'ukkonens_suffix_tree')
    png_dir = os.path.join('pngs', 'ukkonens_suffix_tree')
    os.makedirs(dot_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    if args.string:
        T = args.string
        tree = build_suffix_tree_wrapper(T)
        
        # generate DOT representation and PNG visualization via utils
        dot_output = utils.to_dot(tree)
        dot_file = f'{args.string}_ukkonens_suffix_tree.dot'
        dot_path = os.path.join(dot_dir, dot_file)
        with open(dot_path, 'w') as f:
            f.write(dot_output)
        utils.generate_png_from_dot(dot_path, png_dir)
        
        if args.query:
            for query in args.query:
                match_len = search_tree(tree, query)
                print(f'{query} : {match_len}')
        
    elif args.reference:
        reference = utils.read_fasta(args.reference)
        T = reference[0][1]
        
        if args.region:
            success, message = utils.visualize_region(T, args.region, args.max_size,
                                                      args.reference, output_prefix='ukkonens_suffix_tree',
                                                      use_tree=True)
            print(message)
        else:
            print("No region specified. Skipping visualization for large sequence.")
            
        if args.query:
            print(f"Warning: Building suffix tree for complete sequence ({len(T)} bp)")
            print("This may consume significant memory and time.")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                print("Exiting.")
                return
            else:
                print(f"Attempting to build suffix tree for complete sequence ({len(T)} bp)")
                tree = build_suffix_tree_wrapper(T)
                print("Tree built, processing queries...")
                for query in args.query:
                    match_len = search_tree(tree, query)
                    print(f'{query} : {match_len}')


if __name__ == '__main__':
    main()