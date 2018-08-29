cimport cython
cimport numpy as np

import numpy as np

cdef class Node:
    '''A node of a tree'''
    
    cdef:
        str name
        bint is_leaf
        object left, right, parent, sister
        double edge_length
        Py_ssize_t post_index
        
    def __cinit__(self,name):
        self.name = name
        self.edge_length = 0.
        
        self.left = None    # node object
        self.right = None   # node object
        self.parent = None  # node object
        self.sister = None  # node object

        self.post_index = 0
        
    @property
    def name(self):
        return self.name
        
    @property
    def is_leaf(self):
        return self.is_leaf
        
    @property
    def edge_length(self):
        return self.edge_length
        
    @property
    def left(self):
        return self.left
    
    @property
    def right(self):
        return self.right
        
    @property
    def children(self):
        return (self.left,self.right)
    
    @property
    def parent(self):
        return self.parent
        
    @property
    def sister(self):
        return self.sister
        
    @sister.setter
    def sister(self,sis):
        assert isinstance(sis,Node), "Must set sister to type ctree.Node"
        self.sister = sis
        
    @property
    def postindex(self):
        return self.post_index
        
cdef class Tree:
    '''A simple Tree class that implements pre- and post-order traces
    
    Initialize with a dendropy Tree object'''
    
    cdef:
        Py_ssize_t _iter_step
        int _n_nodes
        bint _preord_isfilled
        bint _postord_isfilled
        Node root
        Node [:] _preord_node_list
        Node [:] _postord_node_list
        
    def __cinit__(self, object tree):
        self._n_nodes = len(tree.nodes())
        self._iter_step = 0
        self._preord_isfilled = False
        self._postord_isfilled = False
        self.root = None
        self._preord_node_list = np.empty(self._n_nodes, dtype=Node) # holds results of pre-order traces
        self._postord_node_list = np.empty(self._n_nodes, dtype=Node)
        self.fill(tree.seed_node,None)
        self._iter_step = 0

        
    cdef Node fill(self, object node, Node parent):
        '''Fill tree structure from dendropy nodes starting from the root'''
        cdef Node current_node
        if self.root == None: # first iter
            self.root = Node(node.taxon.label)
            current_node = self.root
        else:
            current_node = Node(node.taxon.label)
            current_node.edge_length = node.edge_length
            current_node.parent = parent
        
        children = node.child_nodes()
        if children == []: # is a leaf
            current_node.edge_length = node.edge_length
            current_node.is_leaf = node.is_leaf()
            self._postord_node_list[self._iter_step] = current_node
            current_node.post_index = self._iter_step
            self._iter_step += 1
            return current_node
        
        current_node.left = self.fill(children[0],parent=current_node)
        current_node.right = self.fill(children[1],parent=current_node)
        current_node.left.sister = current_node.right
        current_node.right.sister = current_node.left
        current_node.is_leaf = node.is_leaf()
        current_node.post_index = self._iter_step
        self._postord_node_list[self._iter_step] = current_node
        self._iter_step += 1
        return current_node
        
    cdef void _preord(self,node):
        if node.left == None:
            self._preord_node_list[self._iter_step] = node
            self._iter_step += 1 
            return
        self._preord_node_list[self._iter_step] = node
        self._iter_step += 1
        self._preord(node.left)
        self._preord(node.right)
        
    @property
    def root(self):
        return self.root
    
    @property
    def n_nodes(self):
        return self._n_nodes
    
    @property
    def preorder_nodes(self):
        '''Return a list'''
        if self._preord_isfilled == False:
            self._preord(self.root)
            self._preord_isfilled = True
            self._iter_step = 0
        return np.array(self._preord_node_list) 
     
    @property
    def postorder_nodes(self):
        return np.array(self._postord_node_list)