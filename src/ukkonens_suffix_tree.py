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
        self.root.suffix_link = self.root # type: ignore
        
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
                    self.last_new_node.suffix_link = self.active_node # type: ignore
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
                        self.last_new_node.suffix_link = self.active_node # type: ignore
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
                    self.last_new_node.suffix_link = split # type: ignore
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
            # Determine the correct end index for printing.
            end_index = child.end[0] if isinstance(child.end, list) else child.end
            print(self.text[child.start:end_index + 1])
            self.print_edges(child)


# Example usage:
if __name__ == '__main__':
    # Append a unique terminal symbol (like '$') if not already present.
    text = "xabxac$"
    stree = SuffixTree(text)
    stree.print_edges()