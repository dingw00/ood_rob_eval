import torch
import torch.nn as nn
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TopicModel(nn.Module):
  def __init__(self, feat_depth, n_concept, thres):
    super(TopicModel, self).__init__()
    
    # C: concept matrix
    self.concept_mat = nn.Linear(feat_depth, n_concept, bias=False)

    # g: 2-layer fully-connected NN with 500 neurons in the hidden layer
    # concept vector size (2048, n_concept)
    self.rec_1 = nn.Sequential(nn.Linear(n_concept, 500), nn.ReLU())
    self.rec_2 = nn.Linear(500, feat_depth)
    self.thres = thres
     
  def forward(self, f_train):
    # f_train size (None,2048,8,8)
    # f_input size (None,8,8,2048)
    concept_mat_n = nn.functional.normalize(self.concept_mat._parameters['weight'], p=2, dim=1)

    # nn normed liner - new class
    f_input = f_train.permute(0, 2, 3, 1)
    f_input_n =  nn.functional.normalize(f_input, p=2, dim=3)

    topic_prob = torch.matmul(f_input, concept_mat_n.T) # concept scores size (None,8,8,n_concept)
    topic_prob_n =  torch.matmul(f_input_n, concept_mat_n.T)
    topic_prob_mask = (topic_prob_n > self.thres).float()
    topic_prob_am = topic_prob * topic_prob_mask
    topic_prob_nn = nn.functional.normalize(topic_prob_am, p=1, dim=3) # (None,8,8,n_concept)

    # rec size is batch_ size * 8 * 8 * 2048
    rec_layer_1 = nn.functional.relu(self.rec_1(topic_prob_nn)) # (None, 8,8, 500)
    rec_layer_2 = self.rec_2(rec_layer_1) # (None, 8,8, 2048)

    # logits = self.predict(rec_layer_2.permute(0, 3, 1, 2))
    return rec_layer_2.permute(0, 3, 1, 2) # logits, self.topic_vector_n
  
  def get_topic_vec(self):
    concept_mat_ = self.concept_mat._parameters['weight']
    return concept_mat_.detach()
  
  def get_concept_scores(self, f_test):

    concept_mat_n = nn.functional.normalize(self.concept_mat._parameters['weight'], p=2, dim=1)

    f_input = f_test.permute(0, 2, 3, 1)
    f_input_n =  nn.functional.normalize(f_input, p=2, dim=3)

    topic_prob_n =  torch.matmul(f_input_n, concept_mat_n.T) # (N, a, b, n_concept)

    return topic_prob_n