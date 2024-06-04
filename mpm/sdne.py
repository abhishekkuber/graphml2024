import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import networkx as nx
import numpy as np


class Graph(object):
    def __init__(self):
        self.G = None
        self.look_up_dict = {}
        self.look_back_list = []
        self.node_size = 0

    def encode_node(self):
        look_up = self.look_up_dict
        look_back = self.look_back_list
        for node in self.G.nodes():
            look_up[node] = self.node_size
            look_back.append(node)
            self.node_size += 1
            self.G.nodes[node]['status'] = ''

    def read_adjlist(self, filename):
        """ Read graph from adjacency file in which the edge must be unweighted
            the format of each line: v1 n1 n2 n3 ... nk
            :param filename: the filename of input file
        """
        self.G = nx.read_adjlist(filename, create_using=nx.DiGraph())
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = 1.0
        self.encode_node()

    def add_edgelist(self, mirna_disease, mirna_pcg, disease_pcg, pcg_pcg, n_mirna, n_disease, n_pcg, weighted=False, directed=False):
        self.G = nx.Graph()
        nodes = list(range(n_mirna+n_disease+n_pcg))
        self.G.add_nodes_from(nodes)
        mirna_disease = [[item[0], item[1]+n_mirna] for item in mirna_disease]
        mirna_pcg = [[item[0], item[1] + n_mirna+n_disease] for item in mirna_pcg]
        disease_pcg = [[item[0], item[1] + n_mirna+n_disease] for item in disease_pcg]
        pcg_pcg = [[item[0] + n_mirna+n_disease, item[1] + n_mirna+n_disease] for item in pcg_pcg]
        edges = mirna_disease + mirna_pcg + disease_pcg + pcg_pcg
        self.G.add_edges_from(edges)
        self.encode_node()


    def read_node_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['label'] = vec[1:]
        fin.close()

    def read_node_features(self, filename):
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            self.G.nodes[vec[0]]['feature'] = np.array(
                [float(x) for x in vec[1:]])
        fin.close()

    def read_node_status(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['status'] = vec[1]  # train test valid
        fin.close()

    def read_edge_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G[vec[0]][vec[1]]['label'] = vec[2:]
        fin.close()


def fc_op(input_op, name, n_out, layer_collector, act_func=tf.nn.leaky_relu):
    n_in = input_op.get_shape()[-1]
    with tf.compat.v1.name_scope(name) as scope:
        kernel = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([n_in, n_out]), dtype=tf.float32, name=scope + "w")
        biases = tf.Variable(tf.constant(0, shape=[1, n_out], dtype=tf.float32), name=scope + 'b')

        fc = tf.add(tf.matmul(input_op, kernel), biases)
        activation = act_func(fc, name=scope + 'act')
        layer_collector.append([kernel, biases])
        return activation


class SDNE(object):
    def __init__(self, graph, encoder_layer_list, alpha=1e-6, beta=5., nu=1e-5,
                 batch_size=2000, max_iter=500, learning_rate=0.01, adj_mat=None):

        self.g = graph

        self.node_size = self.g.G.number_of_nodes()
        self.rep_size = encoder_layer_list[-1]

        self.encoder_layer_list = [self.node_size]
        self.encoder_layer_list.extend(encoder_layer_list)
        self.encoder_layer_num = len(encoder_layer_list)+1

        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.bs = batch_size
        self.max_iter = max_iter
        self.lr = learning_rate

        self.sess = tf.compat.v1.Session()
        self.vectors = {}

        self.adj_mat = self.getAdj()
        self.embeddings = self.get_train()

        look_back = self.g.look_back_list

        for i, embedding in enumerate(self.embeddings):
            self.vectors[look_back[i]] = embedding

    def getAdj(self):
        node_size = self.g.node_size
        look_up = self.g.look_up_dict
        adj = np.zeros((node_size, node_size))
        for edge in self.g.G.edges():
            adj[look_up[edge[0]]][look_up[edge[1]]] = 1.0
        return adj

    def get_train(self):
        adj_mat = self.adj_mat

        AdjBatch = tf.compat.v1.placeholder(tf.float32, [None, self.node_size], name='adj_batch')
        Adj = tf.compat.v1.placeholder(tf.float32, [None, None], name='adj_mat')
        B = tf.compat.v1.placeholder(tf.float32, [None, self.node_size], name='b_mat')

        fc = AdjBatch
        scope_name = 'encoder'
        layer_collector = []

        with tf.compat.v1.name_scope(scope_name):
            for i in range(1, self.encoder_layer_num):
                #print(i)
                fc = fc_op(fc,
                           name=scope_name+str(i),
                           n_out=self.encoder_layer_list[i],
                           layer_collector=layer_collector)

        _embeddings = fc

        scope_name = 'decoder'
        with tf.compat.v1.name_scope(scope_name):
            for i in range(self.encoder_layer_num-2, 0, -1):
                #print(i)
                fc = fc_op(fc,
                           name=scope_name+str(i),
                           n_out=self.encoder_layer_list[i],
                           layer_collector=layer_collector)
            fc = fc_op(fc,
                       name=scope_name+str(0),
                       n_out=self.encoder_layer_list[0],
                       layer_collector=layer_collector,)

        _embeddings_norm = tf.reduce_sum(tf.square(_embeddings), 1, keepdims=True)

        L_1st = tf.reduce_sum(
            Adj * (
                    _embeddings_norm - 2 * tf.matmul(
                        _embeddings, tf.transpose(_embeddings)
                    ) + tf.transpose(_embeddings_norm)
            )
        )

        L_2nd = tf.reduce_sum(tf.square((AdjBatch - fc) * B))

        L = L_2nd + self.alpha * L_1st

        for param in layer_collector:
            L += self.nu * (tf.reduce_sum(tf.square(param[0]) + tf.abs(param[0])))

        optimizer = tf.compat.v1.train.AdamOptimizer()

        train_op = optimizer.minimize(L)

        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        for step in range(self.max_iter):
            index = np.random.randint(self.node_size, size=self.bs)
            adj_batch_train = adj_mat[index, :]
            adj_mat_train = adj_batch_train[:, index]
            b_mat_train = 1.*(adj_batch_train <= 1e-10) + self.beta * (adj_batch_train > 1e-10)

            self.sess.run(train_op, feed_dict={AdjBatch: adj_batch_train,
                                               Adj: adj_mat_train,
                                               B: b_mat_train})
            #if step % 50 == 0:
            #    print("step %i: %s" % (step, self.sess.run([L, L_1st, L_2nd],
            #                                               feed_dict={AdjBatch: adj_batch_train,
            #                                                          Adj: adj_mat_train,
            #                                                          B: b_mat_train})))

        return self.sess.run(_embeddings, feed_dict={AdjBatch: adj_mat})

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors)
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()
