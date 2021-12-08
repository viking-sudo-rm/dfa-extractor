# modifed from https://albertauyeung.github.io/2020/06/15/python-trie.html/

class TrieNode:
    """A node in the trie structure"""

    def __init__(self, id, char):

        self.char = char
        self.id = id
        self.is_final = False
        self.children = {}

class Trie(object):
    """The trie object"""

    def __init__(self, corpus):
        """
        The trie has at least the root node.
        The root node does not store any character
        """

        self.root = TrieNode(0, "")
        self.count = 0
        self.arcs = []
        # self.strings = []

        for word in corpus:
            self.insert(word)

        self.states = [[i, False] for i in range(self.count + 1)]

        self.dfs(self.root)

    def insert(self, word):
        """Insert a word into the trie"""
        node = self.root

        # Loop through each character in the word
        # Check if there is no child containing the character, create a new child for the current node
        # prefix = ''
        # known_path = False
        for i, char in enumerate(word):
            if char in node.children:
                known_path = True
                node = node.children[char]
                # prefix += char
            else:
                # If a character is not found,
                # create a new node in the trie

                # if (known_path and not node.is_final):
                #     self.strings.append(prefix)
                # known_path = False
                self.count += 1
                new_node = TrieNode(self.count, char)
                node.children[char] = new_node
                node = new_node

        # Mark the end of a word
        # self.strings.append(word)
        node.is_final = True

        # Increment the counter to indicate that we see this word once more
        # node.counter += 1

    def dfs(self, node):
        """Depth-first traversal of the trie

        Args:
            node: the node to start with
        """

        self.states[node.id][1] = node.is_final
        # self.strings[-1] += node.char
        # print(node.id, self.strings[-1])
        # if (len(node.children.values()) > 1):
        #     prefix = self.strings[-1]
        #     for _ in node.children.values():
        #         self.strings.append(prefix)
        for child in node.children.values():
            self.dfs(child)
            self.arcs.append((node.id, child.char, child.id))

# t = Trie(['ab', 'abab'])
# print(t.states)
# print(t.arcs)
