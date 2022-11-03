import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from spektral.layers.pooling import GlobalSumPool, GlobalAvgPool
from spektral.layers.convolutional import EdgeConv
from spektral.models.general_gnn import MLP

class NezNet(Model):
    """ Graph Neural Network using EdgeConv """
    def __init__(self, hidden_pre=32, hidden_cn=32, hidden_post=32, n_pre=1, n_post=1, act="relu", bn=True):
        super().__init__()
        self.mlp_pre = MLP(hidden_pre, layers=n_pre, activation=act, final_activation=act, batch_norm=bn)
        self.conv = EdgeConv(hidden_cn,activation=act, aggregate='sum')
        self.global_pooling_label = GlobalSumPool()
        self.mlp_post_label1 = MLP(hidden_post, layers=n_post, activation=act, final_activation=act, batch_norm=bn)
        self.mlp_post_label2 = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x, a, i = inputs
        x = self.mlp_pre(x)
        x = self.conv([x, a])
        x = self.global_pooling_label([x, i])
        x = self.mlp_post_label1(x)
        x = self.mlp_post_label2(x)

        return x

class DenseNet(Model):
    """ Graph Neural Network without message passing, just aggregation """
    def __init__(self, hidden_pre=32, hidden_post=32, n_pre=2, n_post=1, act="relu", bn=True):
        super().__init__()
        self.mlp_pre = MLP(hidden_pre, layers=n_pre, activation=act, final_activation=act, batch_norm=bn)
        self.global_pooling = GlobalSumPool()
        self.mlp_post_label1 = MLP(hidden_post, layers=n_post, activation=act, final_activation=act, batch_norm=bn)
        self.mlp_post_label2 = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x, a, i = inputs
        x = self.mlp_pre(x)
        x = self.global_pooling([x, i])
        x = self.mlp_post_label1(x)
        x = self.mlp_post_label2(x)
        return x
